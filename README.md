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

Results are returned in descending score order; equal scores are ordered by item index, so output is deterministic. Each result reports the chunk scale it came from and its 1-based rank within that scale. Items whose score is not finite are skipped.

`--per-scale N` caps how many results may come from any single chunk scale, which is useful when an index holds overlapping multi-scale chunks. Results stay score-ordered and the query may then return fewer than `--top-k` entries:

```sh
./build/cpp-rag query \
  --model models/bge-small-en-v1.5-f16.gguf \
  --index docs.ragidx \
  --query "How is the index stored?" \
  --top-k 6 --per-scale 2
```

## Ask: retrieve, assemble, generate

`ask` runs the whole pipeline: it embeds the query, searches the index, assembles the retrieved chunks into a prompt, and generates an answer. It needs an embedding model and a generation model, and they are loaded one after the other so only one is resident at a time:

```sh
./build/cpp-rag ask \
  --model models/bge-small-en-v1.5-f16.gguf \
  --generator models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --index docs.ragidx \
  --query "How is the index stored?" \
  --top-k 5 --ctx 4096 --max-tokens 256 --gpu-layers 99
```

stdout is the answer only; retrieved context (score, chunk id, scale, rank) goes to stderr so the answer can be piped. `--system TEXT` replaces the default instruction. The prompt is wrapped in the ChatML template the Instruct model expects, and generation stops at the turn markers, which are stripped from the output.

The retrieved context is trimmed to fit `--ctx`: chunks are dropped from the least relevant end until prompt plus `--max-tokens` fits, and `ask` reports how many survived. `--embed-ctx N` sets the embedding context separately from the generation context (`--ctx`) when a small generation window would not fit a chunk. The prompt assembly itself is a draft: see `include/cpp_rag/prompt_builder.hpp`.

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

`--prompt` is passed as already-formatted text unless `--chat` is given, which wraps it in the ChatML template with an optional `--system TEXT` and stops generation at the turn markers instead of running into the next turn. For the RAG layout, construct it with `build_rag_prompt()` from `include/cpp_rag/prompt_builder.hpp`; the returned `std::string` is contiguous and can be passed directly to `LlamaGenerator::generate()`, whose third argument takes stop sequences. Generation uses greedy sampling for deterministic output and fails rather than truncating when prompt plus response capacity exceeds `--ctx`.

## File format

The `CPPRAGIX` v1 header records the item count, dimensions, section offsets, record size, and total file size. Every offset/range is validated before access. Per-item vector norms are stored to avoid recomputing corpus norms during cosine search.
