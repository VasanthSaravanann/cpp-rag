#pragma once

#include "cpp_rag/chunker.hpp"

#include <cstddef>
#include <cstdint>
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

    // A default-constructed, closed or moved-from index is not mapped: is_open()
    // returns false and every accessor throws std::logic_error until the object is
    // assigned an open index.
    [[nodiscard]] bool is_open() const noexcept { return data_ != nullptr; }

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t dimensions() const;
    [[nodiscard]] Chunk chunk(std::size_t index) const;
    [[nodiscard]] std::span<const float> embedding(std::size_t index) const;
    [[nodiscard]] float norm(std::size_t index) const;
    [[nodiscard]] std::uint32_t scale(std::size_t index) const;
    [[nodiscard]] std::uint32_t scale_bytes(std::size_t index) const;

private:
    void close() noexcept;
    void validate();
    // Mapped bytes; throws std::logic_error when the index is not open.
    [[nodiscard]] const std::byte* bytes() const;

    int fd_{-1};
    const std::byte* data_{nullptr};
    std::size_t mapped_size_{0};
};

}  // namespace cpp_rag
