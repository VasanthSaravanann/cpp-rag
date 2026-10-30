#pragma once

#include "cpp_rag/mmap_index.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cpp_rag {

enum class Metric { cosine, dot_product };

struct SearchOptions {
    Metric metric{Metric::cosine};
    // When non-zero, at most this many results are returned from any single chunk
    // scale, so the result can contain fewer than top_k entries. Zero searches the
    // whole index and returns the global top-k.
    std::size_t max_per_scale{0};
};

struct SearchResult {
    std::size_t index{};
    float score{};
    std::uint32_t scale{};
    std::uint32_t scale_bytes{};
    // 1-based rank of this result among results of the same scale. Exact for every
    // returned result, including in max_per_scale mode.
    std::size_t rank_within_scale{};
};

float dot_product(std::span<const float> lhs, std::span<const float> rhs);
float cosine_similarity(std::span<const float> lhs, std::span<const float> rhs);

// Returns up to top_k results in descending score order; ties are ordered by index.
// Candidates whose score is not finite are skipped.
std::vector<SearchResult> search(const MmapIndex& index,
                                 std::span<const float> query,
                                 std::size_t top_k,
                                 const SearchOptions& options);
std::vector<SearchResult> search(const MmapIndex& index,
                                 std::span<const float> query,
                                 std::size_t top_k,
                                 Metric metric = Metric::cosine);

}  // namespace cpp_rag
