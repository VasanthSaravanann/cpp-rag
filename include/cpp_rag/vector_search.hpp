#pragma once

#include "cpp_rag/mmap_index.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace cpp_rag {

enum class Metric { cosine, dot_product };

struct SearchResult {
    std::size_t index{};
    float score{};
};

float dot_product(std::span<const float> lhs, std::span<const float> rhs);
float cosine_similarity(std::span<const float> lhs, std::span<const float> rhs);
std::vector<SearchResult> search(const MmapIndex& index,
                                 std::span<const float> query,
                                 std::size_t top_k,
                                 Metric metric = Metric::cosine);

}  // namespace cpp_rag
