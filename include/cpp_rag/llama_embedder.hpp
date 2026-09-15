#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cpp_rag {

class LlamaEmbedder {
public:
    struct Options {
        std::size_t context_tokens{2048};
        int gpu_layers{0};
        int threads{0};
    };

    explicit LlamaEmbedder(const std::string& model_path);
    LlamaEmbedder(const std::string& model_path, Options options);
    ~LlamaEmbedder();
    LlamaEmbedder(LlamaEmbedder&&) noexcept;
    LlamaEmbedder& operator=(LlamaEmbedder&&) noexcept;
    LlamaEmbedder(const LlamaEmbedder&) = delete;
    LlamaEmbedder& operator=(const LlamaEmbedder&) = delete;

    [[nodiscard]] std::size_t dimensions() const noexcept;
    std::vector<float> embed(std::string_view text);
    std::vector<std::vector<float>> embed_all(std::span<const std::string> texts);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_rag
