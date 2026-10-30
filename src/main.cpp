#include "cpp_rag/prompt_builder.hpp"
#include "cpp_rag/vector_search.hpp"
#ifdef CPP_RAG_WITH_LLAMA
#include "cpp_rag/chunker.hpp"
#include "cpp_rag/llama_embedder.hpp"
#include "cpp_rag/llama_generator.hpp"
#include "cpp_rag/mmap_index.hpp"
#endif

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

// Default instruction for the `ask` command. Evidence-only wording is a draft that
// needs human review before it is treated as policy.
constexpr std::string_view kDefaultSystemPrompt =
    "Answer using only the retrieved context below. If the context does not contain "
    "the answer, say that it is not available. Cite the chunk ids you used.";

[[nodiscard]] std::string default_system_prompt() { return std::string(kDefaultSystemPrompt); }

struct Arguments {
    std::string command;
    std::string model;
    std::string index;
    std::string output;
    std::string query;
    std::string prompt;
    std::string generator;
    std::string system_prompt{default_system_prompt()};
    bool chat{false};
    std::vector<std::string> inputs;
    std::size_t chunk_bytes{1000};
    std::size_t overlap_bytes{150};
    std::size_t context_tokens{2048};
    std::size_t embed_context_tokens{0};
    std::size_t batch_tokens{512};
    std::size_t max_response_tokens{256};
    std::size_t top_k{5};
    std::size_t per_scale{0};
    int gpu_layers{0};
    int threads{0};
    cpp_rag::Metric metric{cpp_rag::Metric::cosine};
};

[[noreturn]] void usage(const std::string& error = {}) {
    if (!error.empty()) std::cerr << "error: " << error << "\n\n";
    std::cerr
        << "Usage:\n"
        << "  cpp-rag index --model MODEL.gguf --output INDEX --input FILE [--input FILE ...]\n"
        << "                [--chunk-bytes N] [--overlap-bytes N] [--ctx N]\n"
        << "  cpp-rag query --model MODEL.gguf --index INDEX --query TEXT [--top-k N]\n"
        << "                [--metric cosine|dot] [--per-scale N] [--ctx N]\n"
        << "  cpp-rag ask --model EMBED.gguf --generator QWEN.gguf --index INDEX --query TEXT\n"
        << "              [--top-k N] [--metric cosine|dot] [--per-scale N]\n"
        << "              [--system TEXT] [--max-tokens N] [--ctx N] [--embed-ctx N]\n"
        << "              [--batch N]\n"
        << "  cpp-rag generate --model QWEN.gguf --prompt TEXT [--chat] [--system TEXT]\n"
        << "                   [--max-tokens N] [--ctx N] [--batch N]\n"
        << "Common model options: [--gpu-layers N] [--threads N]\n";
#ifndef CPP_RAG_WITH_LLAMA
    std::cerr << "\nThis binary was built with CPP_RAG_WITH_LLAMA=OFF: every command needs an\n"
                 "embedding model, so none of them can run. Build with llama.cpp for the CLI.\n";
#endif
    std::exit(error.empty() ? 0 : 2);
}

std::size_t size_value(std::string_view option, const char* value) {
    try {
        std::size_t consumed = 0;
        const auto result = std::stoull(value, &consumed);
        if (consumed != std::string(value).size()) throw std::invalid_argument("trailing");
        return static_cast<std::size_t>(result);
    } catch (...) {
        usage("invalid numeric value for " + std::string(option));
    }
}

Arguments parse(int argc, char** argv) {
    if (argc < 2) usage();
    if (std::string_view(argv[1]) == "--help") usage();
    Arguments args;
    args.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        if (option == "--help") usage();
        if (option == "--chat") {
            args.chat = true;
            continue;
        }
        if (i + 1 >= argc) usage("missing value for " + std::string(option));
        const char* value = argv[++i];
        if (option == "--model") args.model = value;
        else if (option == "--index") args.index = value;
        else if (option == "--output") args.output = value;
        else if (option == "--input") args.inputs.emplace_back(value);
        else if (option == "--query") args.query = value;
        else if (option == "--prompt") args.prompt = value;
        else if (option == "--generator") args.generator = value;
        else if (option == "--system") args.system_prompt = value;
        else if (option == "--chunk-bytes") args.chunk_bytes = size_value(option, value);
        else if (option == "--overlap-bytes") args.overlap_bytes = size_value(option, value);
        else if (option == "--ctx") args.context_tokens = size_value(option, value);
        else if (option == "--embed-ctx") args.embed_context_tokens = size_value(option, value);
        else if (option == "--batch") args.batch_tokens = size_value(option, value);
        else if (option == "--max-tokens") args.max_response_tokens = size_value(option, value);
        else if (option == "--top-k") args.top_k = size_value(option, value);
        else if (option == "--per-scale") args.per_scale = size_value(option, value);
        else if (option == "--gpu-layers") args.gpu_layers = static_cast<int>(size_value(option, value));
        else if (option == "--threads") args.threads = static_cast<int>(size_value(option, value));
        else if (option == "--metric") {
            const std::string_view metric = value;
            if (metric == "cosine") args.metric = cpp_rag::Metric::cosine;
            else if (metric == "dot") args.metric = cpp_rag::Metric::dot_product;
            else usage("--metric must be cosine or dot");
        } else usage("unknown option: " + std::string(option));
    }
    return args;
}

#ifdef CPP_RAG_WITH_LLAMA
cpp_rag::LlamaEmbedder make_embedder(const Arguments& args) {
    if (args.model.empty()) usage("--model is required");
    const auto context = args.embed_context_tokens != 0 ? args.embed_context_tokens
                                                        : args.context_tokens;
    return cpp_rag::LlamaEmbedder(args.model, {context, args.gpu_layers, args.threads});
}

void run_index(const Arguments& args) {
    if (args.output.empty()) usage("--output is required");
    if (args.inputs.empty()) usage("at least one --input is required");
    auto embedder = make_embedder(args);
    std::vector<cpp_rag::IndexItem> items;
    for (const auto& input : args.inputs) {
        auto chunks = cpp_rag::chunk_file(input, {args.chunk_bytes, args.overlap_bytes});
        std::vector<std::string> texts;
        texts.reserve(chunks.size());
        for (const auto& chunk : chunks) texts.push_back(chunk.text);
        auto embeddings = embedder.embed_all(texts);
        if (embeddings.size() != chunks.size()) {
            throw std::runtime_error("embedding count does not match chunk count");
        }
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            items.push_back({std::move(chunks[i]), std::move(embeddings[i])});
        }
    }
    cpp_rag::MmapIndex::write(args.output, items);
    std::cout << "indexed " << items.size() << " chunks (" << embedder.dimensions()
              << " dimensions) into " << args.output << '\n';
}

void run_query(const Arguments& args) {
    if (args.index.empty()) usage("--index is required");
    if (args.query.empty()) usage("--query is required");
    auto embedder = make_embedder(args);
    cpp_rag::MmapIndex index(args.index);
    auto query_embedding = embedder.embed(args.query);
    const auto results = cpp_rag::search(
        index, query_embedding, args.top_k,
        cpp_rag::SearchOptions{args.metric, args.per_scale});
    for (const auto& result : results) {
        const auto chunk = index.chunk(result.index);
        std::cout << "score=" << result.score << " source=" << chunk.source
                  << " bytes=" << chunk.byte_begin << '-' << chunk.byte_end
                  << " scale=" << result.scale << '/' << result.scale_bytes
                  << " rank=" << result.rank_within_scale << '\n'
                  << chunk.text << "\n---\n";
    }
}

std::string chat_prompt(const Arguments& args, std::string_view user_text) {
    return cpp_rag::build_chat_prompt(args.system_prompt, user_text);
}

std::vector<std::string> chat_stop_sequences() {
    // Some GGUFs emit the turn markers as ordinary text, and the predicted text can be
    // split so that "<" belongs to the previous token, so match the bare forms too.
    return {"<|im_end|>", "<|im_start|>", "|im_end|", "|im_start|"};
}

void run_generate(const Arguments& args) {
    if (args.model.empty()) usage("--model is required");
    if (args.prompt.empty()) usage("--prompt is required");
    const auto prompt = args.chat ? chat_prompt(args, args.prompt) : args.prompt;
    const auto stops = chat_stop_sequences();
    cpp_rag::LlamaGenerator generator(
        args.model,
        {args.context_tokens, args.batch_tokens, args.gpu_layers, args.threads});
    std::cout << generator.generate(prompt, args.max_response_tokens, stops) << '\n';
}

void run_ask(const Arguments& args) {
    if (args.model.empty()) usage("--model is required (embedding model)");
    if (args.generator.empty()) usage("--generator is required (generation model)");
    if (args.index.empty()) usage("--index is required");
    if (args.query.empty()) usage("--query is required");

    cpp_rag::MmapIndex index(args.index);
    std::vector<cpp_rag::Chunk> chunks;
    std::vector<std::string> ids;
    {
        auto embedder = make_embedder(args);
        const auto query_embedding = embedder.embed(args.query);
        const auto results = cpp_rag::search(
            index, query_embedding, args.top_k,
            cpp_rag::SearchOptions{args.metric, args.per_scale});
        chunks.reserve(results.size());
        ids.reserve(results.size());
        for (const auto& result : results) {
            auto chunk = index.chunk(result.index);
            ids.push_back(chunk.source + ':' + std::to_string(chunk.byte_begin) + '-' +
                          std::to_string(chunk.byte_end));
            chunks.push_back(std::move(chunk));
            std::cerr << "context score=" << result.score << " id=" << ids.back()
                      << " scale=" << result.scale << '/' << result.scale_bytes
                      << " rank=" << result.rank_within_scale << '\n';
        }
    }
    if (chunks.empty()) {
        std::cerr << "no context retrieved; refusing to answer without evidence\n";
        return;
    }

    cpp_rag::LlamaGenerator generator(
        args.generator,
        {args.context_tokens, args.batch_tokens, args.gpu_layers, args.threads});
    const auto stops = chat_stop_sequences();
    if (args.max_response_tokens >= generator.context_size()) {
        throw std::runtime_error("--max-tokens must be smaller than --ctx");
    }

    // Shrink the retrieved context until prompt plus response fits the context window.
    std::size_t keep = chunks.size();
    std::string prompt;
    while (true) {
        std::vector<cpp_rag::PromptChunk> prompt_chunks;
        prompt_chunks.reserve(keep);
        for (std::size_t i = 0; i < keep; ++i) prompt_chunks.push_back({ids[i], chunks[i].text});
        prompt = chat_prompt(args, cpp_rag::build_rag_prompt("", prompt_chunks, args.query));
        const auto tokens = generator.tokenize(prompt);
        if (!tokens.empty() &&
            tokens.size() <= generator.context_size() - args.max_response_tokens) {
            break;
        }
        if (keep == 0) {
            throw std::runtime_error(
                "query and system prompt exceed the model context window; raise --ctx");
        }
        --keep;
    }
    if (keep < chunks.size()) {
        std::cerr << "context trimmed to " << keep << " of " << chunks.size()
                  << " retrieved chunks to fit --ctx\n";
    }

    std::cout << generator.generate(prompt, args.max_response_tokens, stops) << '\n';
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse(argc, argv);
#ifndef CPP_RAG_WITH_LLAMA
        (void)args;
        throw std::runtime_error("this binary was built with CPP_RAG_WITH_LLAMA=OFF");
#else
        if (args.command == "index") run_index(args);
        else if (args.command == "query") run_query(args);
        else if (args.command == "generate") run_generate(args);
        else if (args.command == "ask") run_ask(args);
        else usage("command must be index, query, generate, or ask");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
