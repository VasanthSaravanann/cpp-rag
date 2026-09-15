#include "cpp_rag/llama_embedder.hpp"

#if defined(CPP_RAG_WITH_LLAMA)
#include "llama.h"
#endif

#include <stdexcept>

#if defined(CPP_RAG_WITH_LLAMA)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#endif

namespace cpp_rag {

struct LlamaEmbedder::Impl {
#if defined(CPP_RAG_WITH_LLAMA)
    llama_model* model{};
    llama_context* context{};
#endif
    std::size_t dimensions{};

    Impl(const std::string& model_path, const Options& options) {
#if defined(CPP_RAG_WITH_LLAMA)
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = options.gpu_layers;
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        if (model == nullptr) {
            llama_backend_free();
            throw std::runtime_error("unable to load GGUF model: " + model_path);
        }

        auto context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<std::uint32_t>(options.context_tokens);
        context_params.n_batch = static_cast<std::uint32_t>(options.context_tokens);
        context_params.n_ubatch = static_cast<std::uint32_t>(options.context_tokens);
        context_params.embeddings = true;
        context_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
        context_params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        const auto threads = options.threads > 0
            ? options.threads
            : static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;

        context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            model = nullptr;
            llama_backend_free();
            throw std::runtime_error("unable to create llama.cpp embedding context");
        }
        dimensions = static_cast<std::size_t>(llama_model_n_embd_out(model));
        if (dimensions == 0 || llama_pooling_type(context) == LLAMA_POOLING_TYPE_NONE) {
            llama_free(context);
            context = nullptr;
            llama_model_free(model);
            model = nullptr;
            llama_backend_free();
            throw std::runtime_error("model does not provide pooled embeddings");
        }
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
    std::vector<llama_token> tokenize(std::string_view text) const {
        if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::runtime_error("input is too large to tokenize");
        }
        const auto* vocab = llama_model_get_vocab(model);
        auto required = llama_tokenize(vocab, text.data(), static_cast<std::int32_t>(text.size()),
                                       nullptr, 0, true, false);
        if (required == 0) return {};
        if (required > 0) {
            throw std::runtime_error("unexpected llama.cpp tokenizer response");
        }
        std::vector<llama_token> tokens(static_cast<std::size_t>(-required));
        const auto count = llama_tokenize(vocab, text.data(), static_cast<std::int32_t>(text.size()),
                                          tokens.data(), static_cast<std::int32_t>(tokens.size()),
                                          true, false);
        if (count < 0) throw std::runtime_error("llama.cpp tokenization failed");
        tokens.resize(static_cast<std::size_t>(count));
        return tokens;
    }
#endif
};

LlamaEmbedder::LlamaEmbedder(const std::string& model_path)
    : LlamaEmbedder(model_path, Options{}) {}

LlamaEmbedder::LlamaEmbedder(const std::string& model_path, Options options)
    : impl_(std::make_unique<Impl>(model_path, options)) {}
LlamaEmbedder::~LlamaEmbedder() = default;
LlamaEmbedder::LlamaEmbedder(LlamaEmbedder&&) noexcept = default;
LlamaEmbedder& LlamaEmbedder::operator=(LlamaEmbedder&&) noexcept = default;

std::size_t LlamaEmbedder::dimensions() const noexcept { return impl_->dimensions; }

std::vector<float> LlamaEmbedder::embed(std::string_view text) {
#if defined(CPP_RAG_WITH_LLAMA)
    auto tokens = impl_->tokenize(text);
    if (tokens.empty()) throw std::invalid_argument("cannot embed empty text");
    if (tokens.size() > llama_n_batch(impl_->context)) {
        throw std::runtime_error("input exceeds embedding context; reduce chunk size or increase --ctx");
    }

    auto batch = llama_batch_init(static_cast<std::int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<std::int32_t>(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }

    if (auto memory = llama_get_memory(impl_->context); memory != nullptr) {
        llama_memory_clear(memory, false);
    }
    int result = 0;
    if (llama_model_has_encoder(impl_->model) && !llama_model_has_decoder(impl_->model)) {
        result = llama_encode(impl_->context, batch);
    } else {
        result = llama_decode(impl_->context, batch);
    }
    llama_batch_free(batch);
    if (result != 0) throw std::runtime_error("llama.cpp failed to compute embedding");

    const auto* values = llama_get_embeddings_seq(impl_->context, 0);
    if (values == nullptr) throw std::runtime_error("llama.cpp returned no pooled embedding");
    std::vector<float> embedding(values, values + impl_->dimensions);
    double squared_norm = 0.0;
    for (const auto value : embedding) squared_norm += static_cast<double>(value) * value;
    const auto norm = static_cast<float>(std::sqrt(squared_norm));
    if (norm > 0.0F) {
        for (auto& value : embedding) value /= norm;
    }
    return embedding;
#else
    (void)text;
    throw std::runtime_error("llama.cpp support is unavailable");
#endif
}

std::vector<std::vector<float>> LlamaEmbedder::embed_all(std::span<const std::string> texts) {
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(texts.size());
    for (const auto& text : texts) embeddings.push_back(embed(text));
    return embeddings;
}

}  // namespace cpp_rag
