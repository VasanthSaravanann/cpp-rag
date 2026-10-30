#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace cpp_rag {

struct PromptChunk {
    std::string_view id;
    std::string_view text;
};

// Draft prompt-assembly boundary. Compliance wording and delimiter behavior
// require human review before production use.
//
// The retrieved-context section is bounded by max_bytes when it is non-zero, which
// drops whole chunks from the end of the span (pass chunks in descending relevance
// order). The headers and the user query are always emitted. Zero means unbounded.
// An empty system_prompt omits the system section entirely.
[[nodiscard]] std::string build_rag_prompt(
    std::string_view system_prompt,
    std::span<const PromptChunk> chunks,
    std::string_view user_query,
    std::size_t max_bytes = 0);

// ChatML wrapper used by instruction-tuned chat models such as Qwen2.5-Instruct.
// The result ends with the assistant header so generation continues in that turn.
[[nodiscard]] std::string build_chat_prompt(std::string_view system_prompt,
                                            std::string_view user_text);

}  // namespace cpp_rag
