#include "cpp_rag/vector_search.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <queue>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#elif defined(__SSE2__)
#include <immintrin.h>
#endif

namespace cpp_rag {

float dot_product(std::span<const float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::invalid_argument("vector dimensions do not match");
    }

    std::size_t i = 0;
    float sum = 0.0F;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t accumulator = vdupq_n_f32(0.0F);
    for (; i + 4 <= lhs.size(); i += 4) {
        accumulator = vmlaq_f32(accumulator, vld1q_f32(lhs.data() + i),
                                vld1q_f32(rhs.data() + i));
    }
#if defined(__aarch64__)
    sum = vaddvq_f32(accumulator);
#else
    const auto pair = vadd_f32(vget_low_f32(accumulator), vget_high_f32(accumulator));
    sum = vget_lane_f32(vpadd_f32(pair, pair), 0);
#endif
#elif defined(__SSE2__)
    __m128 accumulator = _mm_setzero_ps();
    for (; i + 4 <= lhs.size(); i += 4) {
        accumulator = _mm_add_ps(accumulator,
                                 _mm_mul_ps(_mm_loadu_ps(lhs.data() + i),
                                            _mm_loadu_ps(rhs.data() + i)));
    }
    alignas(16) float lanes[4];
    _mm_store_ps(lanes, accumulator);
    sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
#endif
    for (; i < lhs.size(); ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

float cosine_similarity(std::span<const float> lhs, std::span<const float> rhs) {
    const auto product = dot_product(lhs, rhs);
    const auto lhs_norm = std::sqrt(dot_product(lhs, lhs));
    const auto rhs_norm = std::sqrt(dot_product(rhs, rhs));
    if (lhs_norm == 0.0F || rhs_norm == 0.0F) {
        return 0.0F;
    }
    return product / (lhs_norm * rhs_norm);
}

std::vector<SearchResult> search(const MmapIndex& index,
                                 std::span<const float> query,
                                 std::size_t top_k, Metric metric) {
    if (query.size() != index.dimensions()) {
        throw std::invalid_argument("query dimensions do not match index");
    }
    if (top_k == 0 || index.size() == 0) {
        return {};
    }

    float query_norm = 1.0F;
    if (metric == Metric::cosine) {
        query_norm = std::sqrt(dot_product(query, query));
        if (query_norm == 0.0F) {
            return {};
        }
    }

    top_k = std::min(top_k, index.size());

    struct LowerScoreFirst {
        bool operator()(const SearchResult& lhs, const SearchResult& rhs) const {
            return lhs.score > rhs.score;
        }
    };
    using ResultHeap =
        std::priority_queue<SearchResult, std::vector<SearchResult>, LowerScoreFirst>;
    std::map<std::uint32_t, ResultHeap> per_scale;

    for (std::size_t i = 0; i < index.size(); ++i) {
        auto score = dot_product(index.embedding(i), query);
        if (metric == Metric::cosine) {
            const auto item_norm = index.norm(i);
            score = item_norm == 0.0F ? 0.0F : score / (item_norm * query_norm);
        }
        SearchResult candidate{i, score, index.scale(i), index.scale_bytes(i), 0};
        auto& best = per_scale[candidate.scale];
        if (best.size() < top_k) {
            best.push(candidate);
        } else if (score > best.top().score) {
            best.pop();
            best.push(candidate);
        }
    }

    std::map<std::uint32_t, std::vector<SearchResult>> ranked_scales;
    for (auto& [scale, heap] : per_scale) {
        auto& ranked = ranked_scales[scale];
        ranked.reserve(heap.size());
        while (!heap.empty()) {
            ranked.push_back(heap.top());
            heap.pop();
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.score > rhs.score; });
        for (std::size_t rank = 0; rank < ranked.size(); ++rank) {
            ranked[rank].rank_within_scale = rank + 1;
        }
    }

    std::vector<SearchResult> results;
    results.reserve(top_k);
    for (std::size_t rank = 0; results.size() < top_k; ++rank) {
        std::vector<SearchResult> round;
        for (const auto& [scale, ranked] : ranked_scales) {
            (void)scale;
            if (rank < ranked.size()) round.push_back(ranked[rank]);
        }
        if (round.empty()) break;
        std::sort(round.begin(), round.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.score != rhs.score) return lhs.score > rhs.score;
            return lhs.scale < rhs.scale;
        });
        for (const auto& candidate : round) {
            if (results.size() == top_k) break;
            results.push_back(candidate);
        }
    }
    return results;
}

}  // namespace cpp_rag
