#!/usr/bin/env bash
# One-command prototype demo: index this README, retrieve, then answer.
#
#   ./demo.sh              # index, query, ask
#   ./demo.sh query        # retrieval only (no 2 GB generation model needed)
#
# Override paths with CPP_RAG_BIN, EMBED_MODEL, GEN_MODEL.
set -euo pipefail
cd "$(dirname "$0")"

BIN=${CPP_RAG_BIN:-build/cpp-rag}
EMBED=${EMBED_MODEL:-models/bge-small-en-v1.5-f16.gguf}
GEN=${GEN_MODEL:-models/Qwen2.5-3B-Instruct-Q4_K_M.gguf}
INDEX=${INDEX:-/tmp/cpp-rag-demo.ragidx}
MODE=${1:-all}

[[ -x $BIN ]] || {
    echo "no binary at $BIN; build the llama-enabled tree first:" >&2
    echo "  cmake -S . -B build && cmake --build build -j" >&2
    exit 1
}
[[ -f $EMBED ]] || { echo "missing embedding model: $EMBED (see README 'Models')" >&2; exit 1; }

echo "== index (two chunk scales) =="
"$BIN" index --model "$EMBED" --input README.md --output "$INDEX" --scales 1000:150,300:40

echo
echo "== retrieve =="
"$BIN" query --model "$EMBED" --index "$INDEX" \
    --query "How is the index stored on disk?" --top-k 4 --per-scale 1

if [[ $MODE == query ]]; then exit 0; fi

[[ -f $GEN ]] || { echo "missing generation model: $GEN (see README 'Models')" >&2; exit 1; }

echo
echo "== answer =="
"$BIN" ask --model "$EMBED" --generator "$GEN" --index "$INDEX" \
    --query "How is the index stored on disk?" --top-k 4 --per-scale 1 \
    --ctx 4096 --max-tokens 160 --gpu-layers 99 --temp 0.3 --seed 42
