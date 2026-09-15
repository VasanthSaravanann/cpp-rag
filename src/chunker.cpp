#include "cpp_rag/chunker.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <stdexcept>

namespace cpp_rag {
namespace {

bool is_utf8_continuation(unsigned char value) {
    return (value & 0xC0U) == 0x80U;
}

std::size_t advance_to_codepoint(std::string_view text, std::size_t position) {
    while (position < text.size() &&
           is_utf8_continuation(static_cast<unsigned char>(text[position]))) {
        ++position;
    }
    return position;
}

std::size_t choose_end(std::string_view text, std::size_t begin, std::size_t max_bytes) {
    const auto hard_end = std::min(text.size(), begin + max_bytes);
    if (hard_end == text.size()) {
        return hard_end;
    }

    auto safe_end = hard_end;
    while (safe_end > begin &&
           is_utf8_continuation(static_cast<unsigned char>(text[safe_end]))) {
        --safe_end;
    }

    const auto earliest_break = begin + max_bytes / 2;
    for (auto cursor = safe_end; cursor > earliest_break; --cursor) {
        const auto ch = static_cast<unsigned char>(text[cursor - 1]);
        if (ch == '\n' || std::isspace(ch) != 0) {
            return cursor;
        }
    }
    return safe_end > begin ? safe_end : hard_end;
}

}  // namespace

std::vector<Chunk> chunk_text(std::string_view text, std::string source,
                              ChunkOptions options) {
    if (options.max_bytes == 0) {
        throw std::invalid_argument("chunk max_bytes must be greater than zero");
    }
    if (options.overlap_bytes >= options.max_bytes) {
        throw std::invalid_argument("chunk overlap_bytes must be smaller than max_bytes");
    }

    std::vector<Chunk> chunks;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto end = choose_end(text, begin, options.max_bytes);
        chunks.push_back(Chunk{source, std::string(text.substr(begin, end - begin)), begin, end});
        if (end == text.size()) {
            break;
        }
        begin = advance_to_codepoint(text, end - std::min(options.overlap_bytes, end));
        if (begin >= end) {
            begin = end;
        }
    }
    return chunks;
}

std::vector<Chunk> chunk_file(const std::string& path, ChunkOptions options) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("unable to open input file: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return chunk_text(text, path, options);
}

}  // namespace cpp_rag
