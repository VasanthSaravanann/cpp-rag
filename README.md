# cpp-rag

A compact local retrieval pipeline using `libllama` embeddings, overlapping UTF-8-safe C++ text chunks (plain text and PDF), a versioned memory-mapped vector index, and SIMD cosine/dot-product search.

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

Editor tooling reads compile flags through `.clangd` from the `build/` tree, so the llama translation units get `llama.h` and `CPP_RAG_WITH_LLAMA` defined. Configuring only `build-core` still parses the core sources, but clangd then borrows a command from another file for the llama sources and treats their llama-only code as inactive.

## Index documents

Use an embedding GGUF model with pooling support, such as a BGE model converted for llama.cpp:

```sh
./build/cpp-rag index \
  --model models/bge-small-en-v1.5-f16.gguf \
  --input docs/guide.txt --input docs/api.txt \
  --output docs.ragidx \
  --chunk-bytes 1000 --overlap-bytes 150
```

`--input` accepts UTF-8 text files and PDFs. PDF text is extracted per page with PDFKit on Apple platforms; the non-Apple build links a stub that rejects PDF input, so scanned PDFs and non-Apple hosts need their own extraction (no OCR is performed). A PDF with no extractable text is reported as an error rather than indexed empty.

`--gpu-layers N`, `--threads N`, and `--ctx N` configure model execution. Chunk limits are bytes, while `--ctx` is model tokens; if a chunk tokenizes beyond the context, reduce `--chunk-bytes`. Chunks are cut on UTF-8 code point boundaries and prefer the last whitespace within the second half of the window, so a chunk can be shorter than `--chunk-bytes`; `--overlap-bytes` must be smaller than `--chunk-bytes`. Chunks are embedded in order and never lost: the effective overlap shrinks when a short chunk cannot carry the requested overlap.

Every chunk written by this command has `scale = 0` and `scale_bytes = --chunk-bytes`, i.e. a single chunk scale per index. Multi-scale chunking exists in the library (`chunk_text_multi_scale()` / `chunk_file_multi_scale()` in `include/cpp_rag/chunker.hpp`), and `search()` reports the scale of each result, but the CLI has no flag that produces a multi-scale index yet. `query --per-scale` is therefore only useful for indexes built through the library.

## Query

```sh
./build/cpp-rag query \
  --model models/bge-small-en-v1.5-f16.gguf \
  --index docs.ragidx \
  --query "How is the index stored?" \
  --top-k 5 --metric cosine
```

Results are returned in descending score order; equal scores are ordered by item index, so output is deterministic. Each result reports the chunk scale it came from and its 1-based rank within that scale. Items whose score is not finite are skipped.

`--per-scale N` caps how many results may come from any single chunk scale, which is only meaningful for an index built with more than one chunk scale through the library (see above). Results stay score-ordered and the query may then return fewer than `--top-k` entries:

```sh
./build/cpp-rag query \
  --model models/bge-small-en-v1.5-f16.gguf \
  --index docs.ragidx \
  --query "How is the index stored?" \
  --top-k 6 --per-scale 2
```

The index stores fixed-size records, contiguous `float32` vectors, source metadata, and original chunk text. It is written to a temporary file and renamed into place, and opened read-only with POSIX `mmap`; vectors are searched without deserialization. The dot-product kernel uses ARM NEON on Apple Silicon, SSE2 on x86-64, and a scalar tail/fallback.

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

The file starts with a 64-byte `CPPRAGIX` v2 header. The sizes and offsets below are what this implementation writes; validation is slightly more permissive, as noted after the table.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic, `CPPRAGIX` |
| 8 | 4 | version, `2` |
| 12 | 4 | header size, `64` |
| 16 | 8 | item count, non-zero |
| 24 | 4 | dimensions, non-zero |
| 28 | 4 | record size, `72` |
| 32 | 8 | records offset, `64` |
| 40 | 8 | vectors offset, 64-byte aligned |
| 48 | 8 | strings offset |
| 56 | 8 | total file size, which must equal the mapped size |

It is followed by `item_count` fixed 72-byte records, zero padding up to the 64-byte-aligned vectors offset, the contiguous `float32` vectors (`item_count * dimensions`), and the string block. Each record holds the source and chunk-text offsets/sizes into the string block, the chunk's byte range in the source document, the vector index, the precomputed L2 norm, the chunk scale (`scale`, `scale_bytes`) and a reserved word that is currently unused. Because a record is a fixed size, item `i` is `records_offset + i * 72` bytes and its vector is `vectors_offset + vector_index * dimensions * 4` — both mapped in place rather than deserialised.

Sections are validated in order: the record array must follow the header and fit in the file, the vector section must follow the records, be aligned for `float` (4 bytes) and fit, and the string section must follow the vectors and fit. Section sizes are checked by division so a crafted `item_count` or `dimensions` cannot wrap the byte-size arithmetic; each record is then checked to sit inside the file, inside the string block, and to have a valid vector index and byte range. A file that fails any check is rejected with a message naming the section, and a v1 index is rejected with `unsupported index version; rebuild the index for MSI v2`. Per-item norms mean cosine search never recomputes corpus norms.

Writes go to `INDEX.tmp` and are then renamed into place, so a reader observes either the previous index or the new one; the write is not fsynced, so it survives process death but not power loss. The layout is host-endian and carries no checksum, so an index is not portable across byte orders and silent bit rot is not detected.
