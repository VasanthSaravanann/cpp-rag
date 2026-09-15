#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cpp_rag {

class LlamaGenerator {
public:
    struct Options {
        std::size_t context_tokens{4096};
        std::size_t batch_tokens{512};
        int gpu_layers{0};
        int threads{0};
    };

    explicit LlamaGenerator(const std::string& model_path);
    LlamaGenerator(const std::string& model_path, Options options);
    ~LlamaGenerator();
    LlamaGenerator(LlamaGenerator&&) noexcept;
    LlamaGenerator& operator=(LlamaGenerator&&) noexcept;
    LlamaGenerator(const LlamaGenerator&) = delete;
    LlamaGenerator& operator=(const LlamaGenerator&) = delete;

    [[nodiscard]] std::vector<std::int32_t> tokenize(std::string_view prompt) const;
    [[nodiscard]] std::string generate(std::string_view prompt,
                                       std::size_t max_response_tokens = 256);
    [[nodiscard]] std::size_t context_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_rag
