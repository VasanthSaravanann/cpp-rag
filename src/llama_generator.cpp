#include "cpp_rag/llama_generator.hpp"

#if defined(CPP_RAG_WITH_LLAMA)
#include "llama.h"
#endif

#include <stdexcept>

#if defined(CPP_RAG_WITH_LLAMA)
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <thread>
#endif

namespace cpp_rag {
#if defined(CPP_RAG_WITH_LLAMA)
namespace {

std::string token_piece(const llama_vocab* vocab, llama_token token) {
    std::array<char, 256> buffer{};
    auto count = llama_token_to_piece(vocab, token, buffer.data(),
                                      static_cast<std::int32_t>(buffer.size()), 0, false);
    if (count >= 0) {
        return {buffer.data(), static_cast<std::size_t>(count)};
    }
    std::string piece(static_cast<std::size_t>(-count), '\0');
    count = llama_token_to_piece(vocab, token, piece.data(),
                                 static_cast<std::int32_t>(piece.size()), 0, false);
    if (count < 0) throw std::runtime_error("llama.cpp failed to decode response token");
    piece.resize(static_cast<std::size_t>(count));
    return piece;
}

}  // namespace
#endif

struct LlamaGenerator::Impl {
#if defined(CPP_RAG_WITH_LLAMA)
    llama_model* model{};
    llama_context* context{};
    const llama_vocab* vocab{};
    std::size_t batch_tokens{};
#endif

    Impl(const std::string& model_path, const Options& options) {
#if defined(CPP_RAG_WITH_LLAMA)
        if (options.context_tokens == 0 || options.batch_tokens == 0 ||
            options.context_tokens > std::numeric_limits<std::uint32_t>::max() ||
            options.batch_tokens > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("context and batch token counts must be valid uint32 values");
        }

        if (!std::filesystem::is_regular_file(model_path)) {
            throw std::runtime_error("Qwen GGUF model file does not exist: " + model_path);
        }

        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.gpu_layers;
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (model == nullptr) {
            llama_backend_free();
            throw std::runtime_error("unable to load Qwen GGUF model: " + model_path);
        }

        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(options.context_tokens);
        context_params.n_batch = static_cast<std::uint32_t>(
            std::min(options.batch_tokens, options.context_tokens));
        context_params.n_ubatch = context_params.n_batch;
        const auto thread_count = options.threads > 0
            ? options.threads
            : static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
        context_params.n_threads = thread_count;
        context_params.n_threads_batch = thread_count;

        context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            model = nullptr;
            llama_backend_free();
            throw std::runtime_error("unable to create llama.cpp generation context");
        }
        vocab = llama_model_get_vocab(model);
        batch_tokens = llama_n_batch(context);
#else
        (void)model_path;
        (void)options;
        throw std::runtime_error(
            "llama.cpp support is unavailable; configure with CPP_RAG_WITH_LLAMA=ON "
            "and provide LLAMA_CPP_DIR");
#endif
    }

    ~Impl() {
#if defined(CPP_RAG_WITH_LLAMA)
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
        llama_backend_free();
#endif
    }

#if defined(CPP_RAG_WITH_LLAMA)
    std::vector<llama_token> tokenize(std::string_view prompt) const {
        if (prompt.empty()) throw std::invalid_argument("prompt cannot be empty");
        if (prompt.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::length_error("prompt is too large for llama.cpp tokenization");
        }
        const auto text_size = static_cast<std::int32_t>(prompt.size());
        auto required = llama_tokenize(vocab, prompt.data(), text_size, nullptr, 0, true, false);
        if (required == std::numeric_limits<std::int32_t>::min()) {
            throw std::length_error("tokenized prompt exceeds llama.cpp limits");
        }
        if (required == 0) return {};
        if (required > 0) throw std::runtime_error("unexpected llama.cpp tokenizer response");

        std::vector<llama_token> tokens(static_cast<std::size_t>(-required));
        const auto count = llama_tokenize(vocab, prompt.data(), text_size, tokens.data(),
                                          static_cast<std::int32_t>(tokens.size()), true, false);
        if (count < 0) throw std::runtime_error("llama.cpp prompt tokenization failed");
        tokens.resize(static_cast<std::size_t>(count));
        return tokens;
    }

    void evaluate(std::span<const llama_token> tokens) {
        if (auto memory = llama_get_memory(context); memory != nullptr) {
            llama_memory_clear(memory, false);
        }
        for (std::size_t offset = 0; offset < tokens.size();) {
            const auto count = std::min(batch_tokens, tokens.size() - offset);
            auto batch = llama_batch_init(static_cast<std::int32_t>(count), 0, 1);
            batch.n_tokens = static_cast<std::int32_t>(count);
            for (std::size_t i = 0; i < count; ++i) {
                batch.token[i] = tokens[offset + i];
                batch.pos[i] = static_cast<llama_pos>(offset + i);
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (offset + i + 1 == tokens.size()) ? 1 : 0;
            }
            const auto result = llama_decode(context, batch);
            llama_batch_free(batch);
            if (result != 0) throw std::runtime_error("llama.cpp failed to evaluate prompt tokens");
            offset += count;
        }
    }
#endif
};

LlamaGenerator::LlamaGenerator(const std::string& model_path)
    : LlamaGenerator(model_path, Options{}) {}
LlamaGenerator::LlamaGenerator(const std::string& model_path, Options options)
    : impl_(std::make_unique<Impl>(model_path, options)) {}
LlamaGenerator::~LlamaGenerator() = default;
LlamaGenerator::LlamaGenerator(LlamaGenerator&&) noexcept = default;
LlamaGenerator& LlamaGenerator::operator=(LlamaGenerator&&) noexcept = default;

std::vector<std::int32_t> LlamaGenerator::tokenize(std::string_view prompt) const {
#if defined(CPP_RAG_WITH_LLAMA)
    auto tokens = impl_->tokenize(prompt);
    return {tokens.begin(), tokens.end()};
#else
    (void)prompt;
    throw std::runtime_error("llama.cpp support is unavailable");
#endif
}

std::size_t LlamaGenerator::context_size() const noexcept {
#if defined(CPP_RAG_WITH_LLAMA)
    return llama_n_ctx_seq(impl_->context);
#else
    return 0;
#endif
}

std::string LlamaGenerator::generate(std::string_view prompt, std::size_t max_response_tokens) {
#if defined(CPP_RAG_WITH_LLAMA)
    auto prompt_tokens = impl_->tokenize(prompt);
    if (prompt_tokens.empty()) throw std::runtime_error("prompt produced no tokens");
    if (max_response_tokens > context_size() ||
        prompt_tokens.size() > context_size() - max_response_tokens) {
        throw std::length_error("prompt and requested response exceed the model context window");
    }

    impl_->evaluate(prompt_tokens);
    auto* sampler = llama_sampler_init_greedy();
    if (sampler == nullptr) throw std::runtime_error("unable to create llama.cpp sampler");

    std::string response;
    response.reserve(max_response_tokens * 4);
    auto position = prompt_tokens.size();
    auto response_batch = llama_batch_init(1, 0, 1);
    try {
        for (std::size_t generated = 0; generated < max_response_tokens; ++generated) {
            const auto token = llama_sampler_sample(sampler, impl_->context, -1);
            if (llama_vocab_is_eog(impl_->vocab, token)) break;
            response.append(token_piece(impl_->vocab, token));

            response_batch.n_tokens = 1;
            response_batch.token[0] = token;
            response_batch.pos[0] = static_cast<llama_pos>(position++);
            response_batch.n_seq_id[0] = 1;
            response_batch.seq_id[0][0] = 0;
            response_batch.logits[0] = 1;
            if (llama_decode(impl_->context, response_batch) != 0) {
                throw std::runtime_error("llama.cpp failed to evaluate response token");
            }
        }
    } catch (...) {
        llama_batch_free(response_batch);
        llama_sampler_free(sampler);
        throw;
    }
    llama_batch_free(response_batch);
    llama_sampler_free(sampler);
    return response;
#else
    (void)prompt;
    (void)max_response_tokens;
    throw std::runtime_error("llama.cpp support is unavailable");
#endif
}

}  // namespace cpp_rag
