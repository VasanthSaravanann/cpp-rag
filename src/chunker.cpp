#include "cpp_rag/chunker.hpp"

#include "cpp_rag/document_loader.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
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
    if (options.max_bytes == 0 ||
        options.max_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("chunk max_bytes must fit the index format");
    }
    if (options.overlap_bytes >= options.max_bytes) {
        throw std::invalid_argument("chunk overlap_bytes must be smaller than max_bytes");
    }

    std::vector<Chunk> chunks;
    std::size_t begin = 0;
    while (begin < text.size()) {
        const auto end = choose_end(text, begin, options.max_bytes);
        chunks.push_back(Chunk{source, std::string(text.substr(begin, end - begin)), begin, end,
                               0, static_cast<std::uint32_t>(options.max_bytes)});
        if (end == text.size()) {
            break;
        }
        // choose_end() can return a chunk shorter than the requested overlap. Clamping
        // the overlap to end - 1 keeps the next start strictly after the current one,
        // even after snapping it forward to a UTF-8 code point boundary. Without the
        // clamp the next start can be recomputed as the current one and the loop
        // repeats the same chunk until the process runs out of memory.
        const auto overlap = std::min(options.overlap_bytes, end - begin - 1);
        begin = advance_to_codepoint(text, end - overlap);
    }
    return chunks;
}

std::vector<Chunk> chunk_file(const std::string& path, ChunkOptions options) {
    return chunk_text(load_document_text(path), path, options);
}

std::vector<Chunk> chunk_text_multi_scale(std::string_view text, std::string source,
                                          std::span<const ChunkOptions> scales) {
    if (scales.empty()) throw std::invalid_argument("at least one chunk scale is required");
    if (scales.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("too many chunk scales");
    }

    std::vector<Chunk> result;
    for (std::size_t scale = 0; scale < scales.size(); ++scale) {
        if (scales[scale].max_bytes > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("chunk scale exceeds index format limit");
        }
        auto chunks = chunk_text(text, source, scales[scale]);
        for (auto& chunk : chunks) {
            chunk.scale = static_cast<std::uint32_t>(scale);
            chunk.scale_bytes = static_cast<std::uint32_t>(scales[scale].max_bytes);
            result.push_back(std::move(chunk));
        }
    }
    return result;
}

std::vector<Chunk> chunk_file_multi_scale(const std::string& path,
                                          std::span<const ChunkOptions> scales) {
    const auto text = load_document_text(path);
    return chunk_text_multi_scale(text, path, scales);
}

}  // namespace cpp_rag
