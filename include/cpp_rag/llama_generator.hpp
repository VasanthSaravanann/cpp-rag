#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
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
        // 0 (the default) keeps greedy decoding, which is deterministic. A positive
        // value switches to top-k/top-p/temperature sampling seeded with `seed`;
        // 0xFFFFFFFF asks llama.cpp to pick a random seed.
        float temperature{0.0f};
        float top_p{0.95f};
        std::uint32_t seed{0xFFFFFFFFu};
    };

    explicit LlamaGenerator(const std::string& model_path);
    LlamaGenerator(const std::string& model_path, Options options);
    ~LlamaGenerator();
    LlamaGenerator(LlamaGenerator&&) noexcept;
    LlamaGenerator& operator=(LlamaGenerator&&) noexcept;
    LlamaGenerator(const LlamaGenerator&) = delete;
    LlamaGenerator& operator=(const LlamaGenerator&) = delete;

    [[nodiscard]] std::vector<std::int32_t> tokenize(std::string_view prompt) const;
    // Greedy generation unless Options::temperature is positive. Stops at end of
    // generation or as soon as any stop sequence appears in the response; the
    // sequence and everything after it is dropped.
    [[nodiscard]] std::string generate(std::string_view prompt,
                                       std::size_t max_response_tokens = 256,
                                       std::span<const std::string> stop_sequences = {});
    [[nodiscard]] std::size_t context_size() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace cpp_rag
