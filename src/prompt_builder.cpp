#include "cpp_rag/prompt_builder.hpp"

#include <limits>
#include <stdexcept>

namespace cpp_rag {
namespace {

constexpr std::string_view kSystemHeader = "[SYSTEM PROMPT]\n";
constexpr std::string_view kContextHeader = "\n\n[RETRIEVED CONTEXT]\n";
constexpr std::string_view kContextHeaderPlain = "[RETRIEVED CONTEXT]\n";
constexpr std::string_view kChunkPrefix = "[Chunk ID: ";
constexpr std::string_view kChunkSuffix = "]\n";
constexpr std::string_view kQueryHeader = "[USER QUERY]\n";
constexpr std::string_view kSectionBreak = "\n\n";

constexpr std::string_view kChatStart = "<|im_start|>";
constexpr std::string_view kChatEnd = "<|im_end|>";
constexpr std::string_view kChatSystemRole = "system\n";
constexpr std::string_view kChatUserRole = "user\n";
constexpr std::string_view kChatAssistantRole = "assistant\n";

void add_size(std::size_t& total, std::size_t amount) {
    if (amount > std::numeric_limits<std::size_t>::max() - total) {
        throw std::length_error("assembled prompt is too large");
    }
    total += amount;
}

std::size_t chunk_size(const PromptChunk& chunk) {
    std::size_t size = 0;
    add_size(size, kChunkPrefix.size());
    add_size(size, chunk.id.size());
    add_size(size, kChunkSuffix.size());
    add_size(size, chunk.text.size());
    add_size(size, kSectionBreak.size());
    return size;
}

// Keeps the prefix of chunks whose combined context-section size fits the budget.
std::size_t fit_chunks(std::span<const PromptChunk> chunks, std::size_t max_bytes,
                       std::size_t header_size) {
    if (max_bytes == 0) return chunks.size();
    std::size_t used = header_size;
    std::size_t included = 0;
    for (const auto& chunk : chunks) {
        const auto size = chunk_size(chunk);
        // `used` never exceeds max_bytes, so this subtraction cannot wrap.
        if (size > max_bytes - used) break;
        used += size;
        ++included;
    }
    return included;
}

}  // namespace

std::string build_rag_prompt(std::string_view system_prompt,
                             std::span<const PromptChunk> chunks,
                             std::string_view user_query,
                             std::size_t max_bytes) {
    const auto* context_header = system_prompt.empty() ? &kContextHeaderPlain : &kContextHeader;
    const auto included = fit_chunks(chunks, max_bytes, context_header->size());

    std::size_t required = 0;
    if (!system_prompt.empty()) {
        add_size(required, kSystemHeader.size());
        add_size(required, system_prompt.size());
    }
    add_size(required, context_header->size());
    for (std::size_t i = 0; i < included; ++i) add_size(required, chunk_size(chunks[i]));
    add_size(required, kQueryHeader.size());
    add_size(required, user_query.size());

    std::string prompt;
    prompt.reserve(required);
    if (!system_prompt.empty()) {
        prompt.append(kSystemHeader);
        prompt.append(system_prompt);
    }
    prompt.append(*context_header);
    for (std::size_t i = 0; i < included; ++i) {
        prompt.append(kChunkPrefix);
        prompt.append(chunks[i].id);
        prompt.append(kChunkSuffix);
        prompt.append(chunks[i].text);
        prompt.append(kSectionBreak);
    }
    prompt.append(kQueryHeader);
    prompt.append(user_query);
    return prompt;
}

std::string build_chat_prompt(std::string_view system_prompt,
                              std::string_view user_text) {
    std::size_t required = 0;
    add_size(required, kChatStart.size() + kChatSystemRole.size());
    add_size(required, system_prompt.size());
    add_size(required, kChatEnd.size() + 1);
    add_size(required, kChatStart.size() + kChatUserRole.size());
    add_size(required, user_text.size());
    add_size(required, kChatEnd.size() + 1);
    add_size(required, kChatStart.size() + kChatAssistantRole.size());

    std::string prompt;
    prompt.reserve(required);
    prompt.append(kChatStart);
    prompt.append(kChatSystemRole);
    prompt.append(system_prompt);
    prompt.append(kChatEnd);
    prompt.push_back('\n');
    prompt.append(kChatStart);
    prompt.append(kChatUserRole);
    prompt.append(user_text);
    prompt.append(kChatEnd);
    prompt.push_back('\n');
    prompt.append(kChatStart);
    prompt.append(kChatAssistantRole);
    return prompt;
}

}  // namespace cpp_rag
