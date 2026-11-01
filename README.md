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

To build and test the storage/search core without llama.cpp or network access:

```sh
cmake -S . -B build-core -DCPP_RAG_WITH_LLAMA=OFF
cmake --build build-core -j
ctest --test-dir build-core --output-on-failure
```

That tree builds `cpp_rag_core` and the unit tests, which cover chunking, the index format and search. The `cpp-rag` binary it also produces is inert by design: every command exits with `this binary was built with CPP_RAG_WITH_LLAMA=OFF`, because indexing, querying and generation all need an embedding model. Build the llama.cpp tree for a usable CLI.

Add `-DCPP_RAG_WARNINGS_AS_ERRORS=ON` for a strict build and
`-DCPP_RAG_ENABLE_SANITIZERS=ON` for AddressSanitizer plus UndefinedBehaviorSanitizer
(Clang/GCC). On macOS 26, AppleClang 17 deadlocks inside dyld during ASan initialisation
even for an empty `main()`, so run the sanitizer build on Linux.

`.github/workflows/ci.yml` runs the strict build and the tests on Ubuntu and macOS,
and the sanitizer build on Ubuntu only, for the reason above.

Editor tooling reads compile flags through `.clangd` from the `build/` tree, so the llama translation units get `llama.h` and `CPP_RAG_WITH_LLAMA` defined. Configuring only `build-core` still parses the core sources, but clangd then borrows a command from another file for the llama sources and treats their llama-only code as inactive.

## Demo

With the llama-enabled build and both models present under `models/` (see below):

```sh
./demo.sh          # index this README, retrieve, then answer
./demo.sh query    # retrieval only
```

The script indexes with two chunk scales, runs a `--per-scale` query and then `ask` with `--temp 0.3 --seed 42`. llama.cpp's model-load and scheduling logs are suppressed at `DEBUG`/`INFO`; set `CPP_RAG_VERBOSE=1` to see them.

## Measuring a configuration

`./eval/run --scales 1000:150,300:40` indexes a small fixture corpus and reports hit@k
and MRR per question, plus a summary. `--generate` additionally scores the produced
answers. The bundled cases are mechanically drafted scaffolding over synthetic
documents, **not** a human-labelled gold set, so the numbers are a regression signal
for comparing configurations and nothing more. See `eval/README.md`.

## Models

Model files are not distributed with this repository: `.gitignore` excludes `*.gguf`, so every example below assumes the file already exists under `models/`. Two roles are involved and they are not interchangeable:

| Role | Used by | Needs |
|---|---|---|
| Embedding | `index`, `query`, `ask --model` | a GGUF that reports a pooling type |
| Generation | `generate`, `ask --generator` | an instruct-tuned causal GGUF |

The examples were run against these two artifacts, each verified by downloading it and comparing its SHA-256 with the digest published in the source repository's own file listing:

| File | Source | Bytes | SHA-256 |
|---|---|---|---|
| `models/bge-small-en-v1.5-f16.gguf` | `CompendiumLabs/bge-small-en-v1.5-gguf`, MIT | 67,308,128 | `f0b2fef971e8366438bfd2d9aefea1b0115919389448806d290237f638bae999` |
| `models/Qwen2.5-3B-Instruct-Q4_K_M.gguf` | `Qwen/Qwen2.5-3B-Instruct-GGUF`, Qwen licence | 2,104,932,768 | `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` |

```sh
mkdir -p models
curl -L -o models/bge-small-en-v1.5-f16.gguf \
  https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-f16.gguf
shasum -a 256 models/bge-small-en-v1.5-f16.gguf   # must print the digest above
```

Neither digest is an endorsement: pin the digests you have approved in your own manifest and re-verify after any download. To avoid a third-party re-upload, upstream `BAAI/bge-small-en-v1.5` (`model.safetensors`, 133,466,304 bytes, SHA-256 `3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad`) can be converted with llama.cpp's `convert_hf_to_gguf.py`, at the cost of needing Python and torch.

A causal instruct model can be fed through the embedding path, and `ask` with the same file as `--model` and `--generator` does run, but the pooled mean of a decoder's token states is not a retrieval embedding and ranking quality is poor. Use that combination only as a smoke test.

## Index documents

Use an embedding GGUF model with pooling support, such as a BGE model converted for llama.cpp:

```sh
./build/cpp-rag index \
  --model models/bge-small-en-v1.5-f16.gguf \
  --input docs/guide.txt --input docs/api.txt \
  --output docs.ragidx \
  --chunk-bytes 1000 --overlap-bytes 150
```

The `docs/*.txt` paths are placeholders; pass your own files (`--input README.md` indexes this README). `--input` accepts UTF-8 text files and PDFs. PDF text is extracted per page with PDFKit on Apple platforms; the non-Apple build links a stub that rejects PDF input, so scanned PDFs and non-Apple hosts need their own extraction (no OCR is performed). A PDF with no extractable text is reported as an error rather than indexed empty.

`--gpu-layers N`, `--threads N`, and `--ctx N` configure model execution. Chunk limits are bytes, while `--ctx` is model tokens; if a chunk tokenizes beyond the context, reduce `--chunk-bytes`. Chunks are cut on UTF-8 code point boundaries and prefer the last whitespace within the second half of the window, so a chunk can be shorter than `--chunk-bytes`; `--overlap-bytes` must be smaller than `--chunk-bytes`. Chunks are embedded in order and never lost: the effective overlap shrinks when a short chunk cannot carry the requested overlap.

By default every chunk in the index has `scale = 0` and `scale_bytes = --chunk-bytes`, i.e. one chunk scale per index. Pass `--scales` to index several chunk sizes at once (each embedded once, in order):

```sh
./build/cpp-rag index \
  --model models/bge-small-en-v1.5-f16.gguf \
  --input docs/guide.txt \
  --output docs.ragidx \
  --scales 1000:150,300:40
```

`--scales` and `--chunk-bytes`/`--overlap-bytes` are mutually exclusive in effect: when `--scales` is given it defines the chunking, and the single-scale options are ignored. Each pair must satisfy `0 < overlap < bytes`. Combined with `--per-scale`, this lets one query mix a broad chunk and a narrow one.

## Query

```sh
./build/cpp-rag query \
  --model models/bge-small-en-v1.5-f16.gguf \
  --index docs.ragidx \
  --query "How is the index stored?" \
  --top-k 5 --metric cosine
```

Results are returned in descending score order; equal scores are ordered by item index, so output is deterministic. Each result reports the chunk scale it came from and its 1-based rank within that scale. Items whose score is not finite are skipped.

`--per-scale N` caps how many results may come from any single chunk scale, which is only meaningful for an index built with `--scales` (see above). Results stay score-ordered and the query may then return fewer than `--top-k` entries:

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

The retrieved context is trimmed to fit `--ctx`: chunks are dropped from the least relevant end until prompt plus `--max-tokens` fits, and `ask` reports how many survived. `--embed-ctx N` sets the embedding context separately from the generation context (`--ctx`) when a small generation window would not fit a chunk.

Each retrieved chunk is quoted between `<<<BEGIN RETRIEVED CHUNK>>>` and `<<<END RETRIEVED CHUNK>>>` fences. Runs of identical angle brackets in chunk text are spaced out (`<<<` becomes `< < <`) so a document cannot reproduce a fence and escape the quoted region, and the default system prompt states that fenced text is data rather than instructions. This is containment, not a guarantee: the prompt-assembly layer is still a draft whose wording needs human review — see `include/cpp_rag/prompt_builder.hpp`.

## Generate with Qwen2.5-3B-Instruct

Use a local Qwen2.5-3B-Instruct GGUF file. The command loads the model and causal context, tokenizes the supplied prompt, evaluates prompt tokens in batches, and generates response tokens (greedy by default):

```sh
./build/cpp-rag generate \
  --model models/Qwen2.5-3B-Instruct-Q4_K_M.gguf \
  --prompt "What encryption protocol is required, and what is the source?" \
  --ctx 4096 --batch 512 --max-tokens 256 \
  --gpu-layers 99
```

`--prompt` is passed as already-formatted text unless `--chat` is given, which wraps it in the ChatML template with an optional `--system TEXT` and stops generation at the turn markers instead of running into the next turn. For the RAG layout, construct it with `build_rag_prompt()` from `include/cpp_rag/prompt_builder.hpp`; the returned `std::string` is contiguous and can be passed directly to `LlamaGenerator::generate()`, whose third argument takes stop sequences. Generation fails rather than truncating when prompt plus response capacity exceeds `--ctx`.

Decoding is greedy by default, so output is deterministic. `--temp T` switches to sampling (top-k 40, then `--top-p P`, default `0.95`, then temperature); `--seed N` fixes the sample stream and defaults to a random seed. `--temp 0` restores greedy decoding. Both `generate` and `ask` accept these flags.

## File format

The file starts with a 96-byte `CPPRAGIX` v3 header. The sizes and offsets below are what this implementation writes; validation is slightly more permissive, as noted after the table.

| Offset | Size | Field |
|---|---|---|
| 0 | 8 | magic, `CPPRAGIX` |
| 8 | 4 | version, `3` |
| 12 | 4 | header size, `96` |
| 16 | 4 | endianness marker, `0x01020304` in host order |
| 20 | 4 | record size, `72` |
| 24 | 8 | item count, non-zero |
| 32 | 4 | dimensions, non-zero |
| 36 | 4 | flags, `0` |
| 40 | 8 | records offset, `96` |
| 48 | 8 | vectors offset, 64-byte aligned |
| 56 | 8 | strings offset |
| 64 | 8 | total file size, which must equal the mapped size |
| 72 | 4 | checksum algorithm, `1` (FNV-1a 64) |
| 76 | 4 | reserved, `0` |
| 80 | 8 | payload checksum over everything after the header |
| 88 | 8 | reserved, `0` |

It is followed by `item_count` fixed 72-byte records, zero padding up to the 64-byte-aligned vectors offset, the contiguous `float32` vectors (`item_count * dimensions`), and the string block. Each record holds the source and chunk-text offsets/sizes into the string block, the chunk's byte range in the source document, the vector index, the precomputed L2 norm, the chunk scale (`scale`, `scale_bytes`) and a reserved word that is currently unused. Because a record is a fixed size, item `i` is `records_offset + i * 72` bytes and its vector is `vectors_offset + vector_index * dimensions * 4` — both mapped in place rather than deserialised.

Sections are validated in order: the record array must follow the header and fit in the file, the vector section must follow the records, be aligned for `float` (4 bytes) and fit, and the string section must follow the vectors and fit. Section sizes are checked by division so a crafted `item_count` or `dimensions` cannot wrap the byte-size arithmetic; each record is then checked to sit inside the file, inside the string block, and to have a valid vector index and byte range. A file that fails any check is rejected with a message naming the section.

The checksum is verified last, so structural damage keeps its specific message and everything the section and record checks cannot see — a flipped text, embedding, or norm byte — is reported as `index payload checksum mismatch; the file is corrupt, rebuild the index` instead of being served as a plausible-looking answer. FNV-1a detects accidental corruption, not tampering: an attacker who edits the payload can recompute the digest. Verifying it reads the whole payload once on open, which trades startup latency for corruption detection; `mmap` still avoids copying vectors into heap memory.

The endianness marker is written in host order. Reading it back swapped means the file came from the opposite byte order, and the index is rejected with `index was written on a host with the opposite byte order; rebuild the index` rather than being misinterpreted. A v1 or v2 index is rejected with `unsupported index version; rebuild the index for CPPRAGIX v3`. Per-item norms mean cosine search never recomputes corpus norms.

Writes go to `INDEX.tmp` and are then renamed into place, so a reader observes either the previous index or the new one; the write is still not fsynced, so it survives process death but not power loss.
