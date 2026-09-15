#pragma once

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
[[nodiscard]] std::string build_rag_prompt(
    std::string_view system_prompt,
    std::span<const PromptChunk> chunks,
    std::string_view user_query);

}  // namespace cpp_rag
