#pragma once

#include "cpp_rag/mmap_index.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace cpp_rag {

enum class Metric { cosine, dot_product };

struct SearchResult {
    std::size_t index{};
    float score{};
    std::uint32_t scale{};
    std::uint32_t scale_bytes{};
    std::size_t rank_within_scale{};
};

float dot_product(std::span<const float> lhs, std::span<const float> rhs);
float cosine_similarity(std::span<const float> lhs, std::span<const float> rhs);
std::vector<SearchResult> search(const MmapIndex& index,
                                 std::span<const float> query,
                                 std::size_t top_k,
                                 Metric metric = Metric::cosine);

}  // namespace cpp_rag
