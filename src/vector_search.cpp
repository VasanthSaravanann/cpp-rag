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

namespace {

struct LowerScoreFirst {
    bool operator()(const SearchResult& lhs, const SearchResult& rhs) const {
        return lhs.score > rhs.score;
    }
};

using ResultHeap =
    std::priority_queue<SearchResult, std::vector<SearchResult>, LowerScoreFirst>;

// Stronger score first; equal scores fall back to the lower index so results are
// deterministic across runs and platforms.
bool better_result(const SearchResult& lhs, const SearchResult& rhs) {
    if (lhs.score != rhs.score) return lhs.score > rhs.score;
    return lhs.index < rhs.index;
}

}  // namespace

std::vector<SearchResult> search(const MmapIndex& index,
                                 std::span<const float> query,
                                 std::size_t top_k, const SearchOptions& options) {
    if (query.size() != index.dimensions()) {
        throw std::invalid_argument("query dimensions do not match index");
    }
    if (top_k == 0 || index.size() == 0) {
        return {};
    }

    float query_norm = 1.0F;
    if (options.metric == Metric::cosine) {
        query_norm = std::sqrt(dot_product(query, query));
        if (query_norm == 0.0F) {
            return {};
        }
    }

    top_k = std::min(top_k, index.size());
    // The global top-k is always contained in the union of the per-scale top-k, so
    // one heap per scale is enough. In max_per_scale mode only the per-scale leaders
    // can be returned, so the heaps only need to hold that many.
    const auto heap_size = options.max_per_scale == 0
        ? top_k
        : std::min(top_k, options.max_per_scale);

    std::map<std::uint32_t, ResultHeap> per_scale;
    for (std::size_t i = 0; i < index.size(); ++i) {
        auto score = dot_product(index.embedding(i), query);
        if (options.metric == Metric::cosine) {
            const auto item_norm = index.norm(i);
            score = item_norm == 0.0F ? 0.0F : score / (item_norm * query_norm);
        }
        // A non-finite score would break the strict weak ordering the sort relies on.
        if (!std::isfinite(score)) continue;

        SearchResult candidate{i, score, index.scale(i), index.scale_bytes(i), 0};
        auto& best = per_scale[candidate.scale];
        if (best.size() < heap_size) {
            best.push(candidate);
        } else if (score > best.top().score) {
            best.pop();
            best.push(candidate);
        }
    }

    std::vector<SearchResult> results;
    results.reserve(per_scale.size() * heap_size);
    for (auto& [scale, heap] : per_scale) {
        (void)scale;
        while (!heap.empty()) {
            results.push_back(heap.top());
            heap.pop();
        }
    }
    std::sort(results.begin(), results.end(), better_result);

    std::map<std::uint32_t, std::size_t> scale_rank;
    for (auto& result : results) {
        result.rank_within_scale = ++scale_rank[result.scale];
    }

    if (results.size() > top_k) results.resize(top_k);
    return results;
}

std::vector<SearchResult> search(const MmapIndex& index,
                                 std::span<const float> query,
                                 std::size_t top_k, Metric metric) {
    return search(index, query, top_k, SearchOptions{metric, 0});
}

}  // namespace cpp_rag
