#pragma once

#include "cpp_rag/chunker.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace cpp_rag {

struct IndexItem {
    Chunk chunk;
    std::vector<float> embedding;
};

class MmapIndex {
public:
    MmapIndex() = default;
    explicit MmapIndex(const std::string& path);
    ~MmapIndex();

    MmapIndex(const MmapIndex&) = delete;
    MmapIndex& operator=(const MmapIndex&) = delete;
    MmapIndex(MmapIndex&& other) noexcept;
    MmapIndex& operator=(MmapIndex&& other) noexcept;

    static void write(const std::string& path, std::span<const IndexItem> items);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t dimensions() const noexcept;
    [[nodiscard]] Chunk chunk(std::size_t index) const;
    [[nodiscard]] std::span<const float> embedding(std::size_t index) const;
    [[nodiscard]] float norm(std::size_t index) const;

private:
    void close() noexcept;
    void validate();

    int fd_{-1};
    const std::byte* data_{nullptr};
    std::size_t mapped_size_{0};
};

}  // namespace cpp_rag
