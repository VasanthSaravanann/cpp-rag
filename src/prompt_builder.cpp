#include "cpp_rag/prompt_builder.hpp"

#include <limits>
#include <stdexcept>

namespace cpp_rag {
namespace {

constexpr std::string_view kSystemHeader = "[SYSTEM PROMPT]\n";
constexpr std::string_view kContextHeader = "\n\n[RETRIEVED CONTEXT]\n";
constexpr std::string_view kChunkPrefix = "[Chunk ID: ";
constexpr std::string_view kChunkSuffix = "]\n";
constexpr std::string_view kQueryHeader = "[USER QUERY]\n";
constexpr std::string_view kSectionBreak = "\n\n";

void add_size(std::size_t& total, std::size_t amount) {
    if (amount > std::numeric_limits<std::size_t>::max() - total) {
        throw std::length_error("assembled prompt is too large");
    }
    total += amount;
}

}  // namespace

std::string build_rag_prompt(std::string_view system_prompt,
                             std::span<const PromptChunk> chunks,
                             std::string_view user_query) {
    std::size_t required = 0;
    add_size(required, kSystemHeader.size());
    add_size(required, system_prompt.size());
    add_size(required, kContextHeader.size());
    for (const auto& chunk : chunks) {
        add_size(required, kChunkPrefix.size());
        add_size(required, chunk.id.size());
        add_size(required, kChunkSuffix.size());
        add_size(required, chunk.text.size());
        add_size(required, kSectionBreak.size());
    }
    add_size(required, kQueryHeader.size());
    add_size(required, user_query.size());

    std::string prompt;
    prompt.reserve(required);
    prompt.append(kSystemHeader);
    prompt.append(system_prompt);
    prompt.append(kContextHeader);
    for (const auto& chunk : chunks) {
        prompt.append(kChunkPrefix);
        prompt.append(chunk.id);
        prompt.append(kChunkSuffix);
        prompt.append(chunk.text);
        prompt.append(kSectionBreak);
    }
    prompt.append(kQueryHeader);
    prompt.append(user_query);
    return prompt;
}

}  // namespace cpp_rag
