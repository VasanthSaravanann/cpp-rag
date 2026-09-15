#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cpp_rag {

struct Chunk {
    std::string source;
    std::string text;
    std::size_t byte_begin{};
    std::size_t byte_end{};
};

struct ChunkOptions {
    std::size_t max_bytes{1000};
    std::size_t overlap_bytes{150};
};

std::vector<Chunk> chunk_text(std::string_view text, std::string source,
                              ChunkOptions options = {});
std::vector<Chunk> chunk_file(const std::string& path, ChunkOptions options = {});

}  // namespace cpp_rag
