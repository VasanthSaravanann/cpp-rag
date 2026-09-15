#include "cpp_rag/chunker.hpp"
#include "cpp_rag/mmap_index.hpp"
#include "cpp_rag/prompt_builder.hpp"
#include "cpp_rag/vector_search.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
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

}  // namespace

int main() {
    try {
        test_chunking();
        test_index_and_search();
        test_prompt_assembly();
        test_search_retains_best_top_k();
        std::cout << "all tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
