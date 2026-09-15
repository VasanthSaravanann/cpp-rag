# cpp-rag

A compact local retrieval pipeline using `libllama` embeddings, overlapping C++ text chunks, a versioned memory-mapped vector index, and SIMD cosine/dot-product search.

## Build and link llama.cpp

Use a local checkout (recommended for reproducible/offline builds):

```sh
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
cmake -S . -B build -DLLAMA_CPP_DIR=third_party/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

When `LLAMA_CPP_DIR` is omitted, CMake `FetchContent` downloads the pinned llama.cpp revision. Only `libllama` and its required GGML targets are built; llama.cpp tools, examples, tests, and server are disabled.

To work on storage/search without llama.cpp or network access:

```sh
cmake -S . -B build-core -DCPP_RAG_WITH_LLAMA=OFF
cmake --build build-core -j
ctest --test-dir build-core --output-on-failure
```

## Index documents

Use an embedding GGUF model with pooling support, such as a BGE model converted for llama.cpp:

```sh
./build/cpp-rag index \
  --model models/bge-small-en-v1.5-f16.gguf \
  --input docs/guide.txt --input docs/api.txt \
  --output docs.ragidx \
  --chunk-bytes 1000 --overlap-bytes 150
```

`--gpu-layers N`, `--threads N`, and `--ctx N` configure model execution. Chunk limits are bytes, while `--ctx` is model tokens; if a chunk tokenizes beyond the context, reduce `--chunk-bytes`.

## Query

```sh
./build/cpp-rag query \
  --model models/bge-small-en-v1.5-f16.gguf \
  --index docs.ragidx \
  --query "How is the index stored?" \
  --top-k 5 --metric cosine
```

The index stores fixed-size records, contiguous `float32` vectors, source metadata, and original chunk text. It is atomically written and opened read-only with POSIX `mmap`; vectors are searched without deserialization. The dot-product kernel uses ARM NEON on Apple Silicon, SSE2 on x86-64, and a scalar tail/fallback.

## Generate with Qwen2.5-3B-Instruct

Use a local Qwen2.5-3B-Instruct GGUF file. The command loads the model and causal context, tokenizes the supplied prompt, evaluates prompt tokens in batches, and greedily generates response tokens:

```sh
./build/cpp-rag generate \
  --model models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --prompt "What encryption protocol is required, and what is the source?" \
  --ctx 4096 --batch 512 --max-tokens 256 \
  --gpu-layers 99
```

`--prompt` is passed as already-formatted text. For the RAG layout, first construct it with `build_rag_prompt()` from `include/cpp_rag/prompt_builder.hpp`; the returned `std::string` is contiguous and can be passed directly to `LlamaGenerator::generate()`. Generation uses greedy sampling for deterministic output and fails rather than truncating when prompt plus response capacity exceeds `--ctx`.

## File format

The `CPPRAGIX` v1 header records the item count, dimensions, section offsets, record size, and total file size. Every offset/range is validated before access. Per-item vector norms are stored to avoid recomputing corpus norms during cosine search.
