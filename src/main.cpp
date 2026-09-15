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
#include <vector>

namespace {

struct Arguments {
    std::string command;
    std::string model;
    std::string index;
    std::string output;
    std::string query;
    std::string prompt;
    std::vector<std::string> inputs;
    std::size_t chunk_bytes{1000};
    std::size_t overlap_bytes{150};
    std::size_t context_tokens{2048};
    std::size_t batch_tokens{512};
    std::size_t max_response_tokens{256};
    std::size_t top_k{5};
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
        << "                [--metric cosine|dot] [--ctx N]\n"
        << "  cpp-rag generate --model QWEN.gguf --prompt TEXT [--max-tokens N]\n"
        << "                   [--ctx N] [--batch N]\n"
        << "Common model options: [--gpu-layers N] [--threads N]\n";
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
        if (i + 1 >= argc) usage("missing value for " + std::string(option));
        const char* value = argv[++i];
        if (option == "--model") args.model = value;
        else if (option == "--index") args.index = value;
        else if (option == "--output") args.output = value;
        else if (option == "--input") args.inputs.emplace_back(value);
        else if (option == "--query") args.query = value;
        else if (option == "--prompt") args.prompt = value;
        else if (option == "--chunk-bytes") args.chunk_bytes = size_value(option, value);
        else if (option == "--overlap-bytes") args.overlap_bytes = size_value(option, value);
        else if (option == "--ctx") args.context_tokens = size_value(option, value);
        else if (option == "--batch") args.batch_tokens = size_value(option, value);
        else if (option == "--max-tokens") args.max_response_tokens = size_value(option, value);
        else if (option == "--top-k") args.top_k = size_value(option, value);
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
    return cpp_rag::LlamaEmbedder(args.model, {args.context_tokens, args.gpu_layers, args.threads});
}

void run_index(const Arguments& args) {
    if (args.output.empty()) usage("--output is required");
    if (args.inputs.empty()) usage("at least one --input is required");
    auto embedder = make_embedder(args);
    std::vector<cpp_rag::IndexItem> items;
    for (const auto& input : args.inputs) {
        auto chunks = cpp_rag::chunk_file(input, {args.chunk_bytes, args.overlap_bytes});
        for (auto& chunk : chunks) {
            auto embedding = embedder.embed(chunk.text);
            items.push_back({std::move(chunk), std::move(embedding)});
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
    const auto results = cpp_rag::search(index, query_embedding, args.top_k, args.metric);
    for (const auto& result : results) {
        const auto chunk = index.chunk(result.index);
        std::cout << "score=" << result.score << " source=" << chunk.source
                  << " bytes=" << chunk.byte_begin << '-' << chunk.byte_end << '\n'
                  << chunk.text << "\n---\n";
    }
}

void run_generate(const Arguments& args) {
    if (args.model.empty()) usage("--model is required");
    if (args.prompt.empty()) usage("--prompt is required");
    cpp_rag::LlamaGenerator generator(
        args.model,
        {args.context_tokens, args.batch_tokens, args.gpu_layers, args.threads});
    std::cout << generator.generate(args.prompt, args.max_response_tokens) << '\n';
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
        else usage("command must be index, query, or generate");
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
