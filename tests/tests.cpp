#include "cpp_rag/chunker.hpp"
#include "cpp_rag/mmap_index.hpp"
#include "cpp_rag/prompt_builder.hpp"
#include "cpp_rag/vector_search.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_chunking() {
    const std::string text = "alpha beta gamma delta epsilon zeta eta theta";
    const auto chunks = cpp_rag::chunk_text(text, "memory", {18, 5});
    check(chunks.size() >= 3, "expected multiple chunks");
    check(chunks.front().byte_begin == 0, "first chunk offset is wrong");
    check(chunks.back().byte_end == text.size(), "last chunk offset is wrong");
    for (const auto& chunk : chunks) {
        check(chunk.text == text.substr(chunk.byte_begin, chunk.byte_end - chunk.byte_begin),
              "chunk text and byte range differ");
        check(chunk.text.size() <= 18, "chunk exceeds configured size");
    }

    const std::string utf8 = "one café 世界 two three";
    const auto utf8_chunks = cpp_rag::chunk_text(utf8, "utf8", {10, 3});
    for (const auto& chunk : utf8_chunks) {
        check(chunk.text.empty() || (static_cast<unsigned char>(chunk.text.front()) & 0xC0U) != 0x80U,
              "chunk starts in a UTF-8 continuation byte");
    }
}

void check_chunk_invariants(const std::string& text,
                            const std::vector<cpp_rag::Chunk>& chunks,
                            std::size_t max_bytes) {
    check(!chunks.empty(), "expected at least one chunk");
    check(chunks.front().byte_begin == 0, "first chunk offset is wrong");
    check(chunks.back().byte_end == text.size(), "chunking did not cover the whole input");
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const auto& chunk = chunks[i];
        check(chunk.byte_begin < chunk.byte_end, "chunk byte range is empty");
        check(chunk.byte_end - chunk.byte_begin <= max_bytes, "chunk exceeds configured size");
        check(chunk.text == text.substr(chunk.byte_begin, chunk.byte_end - chunk.byte_begin),
              "chunk text and byte range differ");
        check((static_cast<unsigned char>(chunk.text.front()) & 0xC0U) != 0x80U,
              "chunk starts in a UTF-8 continuation byte");
        if (i > 0) {
            check(chunk.byte_begin > chunks[i - 1].byte_begin, "chunk offsets must advance");
            check(chunk.byte_begin <= chunks[i - 1].byte_end,
                  "chunk offsets must not skip input");
        }
    }
}

void test_chunking_large_overlap() {
    // An early break can produce a chunk shorter than the requested overlap. The next
    // start must still advance; previously it was recomputed as the same offset and the
    // chunker repeated that chunk until the process ran out of memory.
    std::string early_break(51, 'a');
    early_break += ' ';
    early_break += std::string(300, 'b');
    check_chunk_invariants(early_break, cpp_rag::chunk_text(early_break, "early", {100, 90}),
                           100);

    // The largest accepted overlap must still make progress.
    std::string dense(400, 'x');
    check_chunk_invariants(dense, cpp_rag::chunk_text(dense, "dense", {32, 31}), 32);

    // Overlap close to the chunk size must keep starts on code point boundaries.
    std::string utf8;
    for (int i = 0; i < 40; ++i) utf8 += "世界 ";
    check_chunk_invariants(utf8, cpp_rag::chunk_text(utf8, "utf8", {10, 9}), 10);
}

void test_index_and_search() {
    const auto path = (std::filesystem::temp_directory_path() / "cpp-rag-test.index").string();
    const std::vector<cpp_rag::IndexItem> items{
        {{"a.txt", "first", 0, 5}, {1.0F, 0.0F, 0.0F, 0.0F}},
        {{"b.txt", "second", 10, 16}, {0.0F, 1.0F, 0.0F, 0.0F}},
        {{"c.txt", "third", 20, 25}, {0.7F, 0.7F, 0.0F, 0.0F}},
    };
    cpp_rag::MmapIndex::write(path, items);
    {
        cpp_rag::MmapIndex index(path);
        check(index.size() == 3 && index.dimensions() == 4, "index shape is wrong");
        check(index.chunk(1).text == "second", "mapped metadata is wrong");
        const std::vector<float> query{1.0F, 0.0F, 0.0F, 0.0F};
        const auto results = cpp_rag::search(index, query, 2);
        check(results.size() == 2, "top-k result count is wrong");
        check(results[0].index == 0, "best cosine result is wrong");
        check(std::abs(results[0].score - 1.0F) < 1e-5F, "cosine score is wrong");
    }
    std::filesystem::remove(path);
}

void test_prompt_assembly() {
    constexpr std::string_view system_prompt =
        "You are a strict compliance auditor. Answer precisely using the retrieved chunks. "
        "Always cite the section or chunk ID where you found the information. Do not infer or "
        "extrapolate beyond the written text.";
    const std::vector<cpp_rag::PromptChunk> chunks{
        {"101-A", "\"The company completed the audit on Q3 2024 with zero major non-compliance findings.\""},
        {"101-B", "\"All data encryption standards must follow AES-256 protocols as defined in Section 4.\""},
    };

    const auto prompt = cpp_rag::build_rag_prompt(
        system_prompt, chunks, "What encryption protocol is required, and what is the source?");
    const std::string expected =
        "[SYSTEM PROMPT]\n"
        "You are a strict compliance auditor. Answer precisely using the retrieved chunks. "
        "Always cite the section or chunk ID where you found the information. Do not infer or "
        "extrapolate beyond the written text.\n\n"
        "[RETRIEVED CONTEXT]\n"
        "[Chunk ID: 101-A]\n"
        "\"The company completed the audit on Q3 2024 with zero major non-compliance findings.\"\n\n"
        "[Chunk ID: 101-B]\n"
        "\"All data encryption standards must follow AES-256 protocols as defined in Section 4.\"\n\n"
        "[USER QUERY]\n"
        "What encryption protocol is required, and what is the source?";
    check(prompt == expected, "assembled RAG prompt does not match the required format");
    check(prompt.size() == expected.size(), "assembled prompt size is wrong");
}

void test_search_retains_best_top_k() {
    const auto path = (std::filesystem::temp_directory_path() / "cpp-rag-topk.index").string();
    const std::vector<cpp_rag::IndexItem> items{
        {{"a.txt", "first", 0, 5}, {1.0F, 0.0F, 0.0F, 0.0F}},
        {{"b.txt", "second", 10, 16}, {0.8F, 0.0F, 0.0F, 0.0F}},
        {{"c.txt", "third", 20, 25}, {0.9F, 0.0F, 0.0F, 0.0F}},
        {{"d.txt", "fourth", 30, 36}, {0.95F, 0.0F, 0.0F, 0.0F}},
    };
    cpp_rag::MmapIndex::write(path, items);
    cpp_rag::MmapIndex index(path);

    const std::vector<float> query{1.0F, 0.0F, 0.0F, 0.0F};
    const auto results = cpp_rag::search(index, query, 3, cpp_rag::Metric::dot_product);
    check(results.size() == 3, "top-k should keep exactly three results");
    check(results[0].index == 0, "highest dot-product score should be first");
    check(results[1].index == 3, "second-best result should be the 0.95 item");
    check(results[2].index == 2, "third-best result should be the 0.90 item");
    std::filesystem::remove(path);
}

// Header field offsets in the CPPRAGIX v2 file format.
constexpr std::size_t kVersionOffset = 8;
constexpr std::size_t kItemCountOffset = 16;
constexpr std::size_t kDimensionsOffset = 24;
constexpr std::size_t kRecordsOffsetOffset = 32;
constexpr std::size_t kVectorsOffsetOffset = 40;
constexpr std::size_t kStringsOffsetOffset = 48;
constexpr std::size_t kFileSizeOffset = 56;
constexpr std::size_t kRecordSize = 72;
constexpr std::size_t kFirstRecordOffset = 64;

void patch_field(const std::string& path, std::size_t offset, const void* bytes,
                 std::size_t size) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    check(static_cast<bool>(file), "unable to open index for patching");
    file.seekp(static_cast<std::streamoff>(offset));
    file.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    check(static_cast<bool>(file), "unable to patch index");
}

template <typename T>
void patch_field(const std::string& path, std::size_t offset, T value) {
    patch_field(path, offset, &value, sizeof(value));
}

template <typename T>
T read_field(const std::string& path, std::size_t offset) {
    std::ifstream file(path, std::ios::binary);
    check(static_cast<bool>(file), "unable to open index for reading");
    T value{};
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(&value), sizeof(value));
    check(static_cast<bool>(file), "unable to read index field");
    return value;
}

void expect_rejected(const std::string& path, std::string_view reason) {
    std::string error;
    try {
        cpp_rag::MmapIndex index(path);
        (void)index;
    } catch (const std::exception& exception) {
        error = exception.what();
    }
    check(!error.empty(), "corrupt index was accepted");
    const std::string message = "unexpected rejection reason: " + error;
    check(error.find(reason) != std::string::npos, message.c_str());
}

void test_index_rejects_corrupt_headers() {
    const auto path = (std::filesystem::temp_directory_path() / "cpp-rag-corrupt.index").string();
    const std::vector<cpp_rag::IndexItem> items{
        {{"a.txt", "first", 0, 5}, {1.0F, 0.0F, 0.0F, 0.0F}},
        {{"b.txt", "second", 10, 16}, {0.0F, 1.0F, 0.0F, 0.0F}},
    };
    const auto write_clean = [&] { cpp_rag::MmapIndex::write(path, items); };

    write_clean();
    {
        cpp_rag::MmapIndex index(path);
        check(index.size() == 2 && index.dimensions() == 4, "valid index did not open");
    }

    // Would wrap item_count * sizeof(Record) and item_count * dimensions * sizeof(float),
    // letting the record loop read past the mapping.
    write_clean();
    patch_field(path, kItemCountOffset, std::uint64_t{1} << 63);
    expect_rejected(path, "record section");

    // Record section no longer fits in the file.
    write_clean();
    patch_field(path, kItemCountOffset, std::uint64_t{4});
    expect_rejected(path, "record section");

    // Records must not run into the vector section either.
    write_clean();
    patch_field(path, kItemCountOffset, std::uint64_t{3});
    expect_rejected(path, "vector section");

    // Records overlapping the header.
    write_clean();
    patch_field(path, kRecordsOffsetOffset, std::uint64_t{0});
    expect_rejected(path, "record section");

    // Vectors must not overlap the records, must be float-aligned and must fit.
    write_clean();
    patch_field(path, kVectorsOffsetOffset, read_field<std::uint64_t>(path, kVectorsOffsetOffset) - 1);
    expect_rejected(path, "vector section");

    write_clean();
    patch_field(path, kVectorsOffsetOffset, read_field<std::uint64_t>(path, kVectorsOffsetOffset) + 1);
    expect_rejected(path, "vector section");

    write_clean();
    patch_field(path, kDimensionsOffset, std::uint32_t{0xFFFFFFFFU});
    expect_rejected(path, "vector section");

    // String section must follow the vectors and stay inside the file.
    write_clean();
    patch_field(path, kStringsOffsetOffset, read_field<std::uint64_t>(path, kFileSizeOffset) + 1);
    expect_rejected(path, "string section");

    write_clean();
    patch_field(path, kStringsOffsetOffset, std::uint64_t{0});
    expect_rejected(path, "string section");

    // Record fields must stay inside the string section and the file.
    write_clean();
    patch_field(path, kFirstRecordOffset, std::uint64_t{0xFFFFFFFFU});
    expect_rejected(path, "index record is invalid");

    write_clean();
    patch_field(path, kFirstRecordOffset + 64, std::uint32_t{0});
    expect_rejected(path, "index record is invalid");

    // Header identity checks still apply.
    write_clean();
    patch_field(path, kVersionOffset, std::uint32_t{1});
    expect_rejected(path, "unsupported index version");

    write_clean();
    patch_field(path, kFileSizeOffset, read_field<std::uint64_t>(path, kFileSizeOffset) + 1);
    expect_rejected(path, "invalid or unsupported index header");

    write_clean();
    const char corrupted_magic = 'X';
    patch_field(path, 0, &corrupted_magic, 1);
    expect_rejected(path, "invalid index magic");

    std::filesystem::remove(path);
}

void test_multi_scale_search_ordering() {
    const auto path = (std::filesystem::temp_directory_path() / "cpp-rag-scale.index").string();
    // Two chunk scales; the strongest results overall come from scale 0.
    const std::vector<cpp_rag::IndexItem> items{
        {{"a.txt", "fine 1", 0, 6, 0, 1000}, {1.0F, 0.0F}},
        {{"b.txt", "fine 2", 0, 6, 0, 1000}, {0.94F, 0.0F}},
        {{"c.txt", "coarse 1", 0, 8, 1, 200}, {0.90F, 0.0F}},
        {{"d.txt", "coarse 2", 0, 8, 1, 200}, {0.89F, 0.0F}},
    };
    cpp_rag::MmapIndex::write(path, items);
    cpp_rag::MmapIndex index(path);
    const std::vector<float> query{1.0F, 0.0F};

    // Default search is the global top-k, ordered by descending score. A scale's
    // weaker result must never outrank a stronger result from another scale.
    const auto results = cpp_rag::search(index, query, 3, cpp_rag::Metric::dot_product);
    check(results.size() == 3, "multi-scale top-k should keep exactly three results");
    check(results[0].index == 0 && results[1].index == 1 && results[2].index == 2,
          "multi-scale results are not in descending score order");
    check(results[0].rank_within_scale == 1 && results[1].rank_within_scale == 2,
          "rank within scale 0 is wrong");
    check(results[2].rank_within_scale == 1, "rank within scale 1 is wrong");
    check(results[0].scale == 0 && results[2].scale == 1 && results[2].scale_bytes == 200,
          "result scale metadata is wrong");

    // Opt-in per-scale quota keeps results score-ordered and may return fewer than
    // top_k, because scale 0 may contribute at most one result.
    const auto balanced = cpp_rag::search(index, query, 3,
                                          cpp_rag::SearchOptions{cpp_rag::Metric::dot_product, 1});
    check(balanced.size() == 2, "per-scale quota should cap each scale to one result");
    check(balanced[0].index == 0 && balanced[0].score == 1.0F, "quota result order is wrong");
    check(balanced[1].index == 2 && balanced[1].score == 0.90F, "quota result order is wrong");
    check(balanced[0].scale == 0 && balanced[1].scale == 1, "quota results must span scales");
    // Ranks stay exact in quota mode: index 1 still ranks second within scale 0.
    check(balanced[0].rank_within_scale == 1 && balanced[1].rank_within_scale == 1,
          "quota rank within scale is wrong");

    const auto wide = cpp_rag::search(index, query, 3,
                                      cpp_rag::SearchOptions{cpp_rag::Metric::dot_product, 2});
    check(wide.size() == 3 && wide[0].index == 0 && wide[1].index == 1 && wide[2].index == 2,
          "per-scale quota of two should return the global top three");

    std::filesystem::remove(path);
}

}  // namespace

int main() {
    try {
        test_chunking();
        test_chunking_large_overlap();
        test_index_and_search();
        test_index_rejects_corrupt_headers();
        test_prompt_assembly();
        test_search_retains_best_top_k();
        test_multi_scale_search_ordering();
        std::cout << "all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
