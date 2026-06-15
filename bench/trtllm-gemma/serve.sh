#!/usr/bin/env bash
# serve.sh — serve Gemma-4-31B-it (dense, bf16) single-GPU as a local in-process
# TensorRT-LLM (PyTorch backend) engine on the benchbox CVM, so the bench client
# hits it over localhost (no TLS/shim) for a clean CC-overhead measurement.
# The TRT-LLM analog of gemma-probe/gemma_serve.sh (vLLM). Serves :8001. nospec.
#
#   ./serve.sh           # serve TP=1 (mirrors the vLLM Gemma single-GPU arm)
#   SMOKE=1 ./serve.sh   # dummy weights: validate arch load + graph capture in ~3 min
#
# CLI flags verified against TensorRT-LLM 1.3.0rc19 (tensorrt_llm/commands/serve.py),
# same pattern as trtllm-benchbox/bench/serve.sh. Read-only rootfs -> scratch in /dev/shm.
set -euo pipefail

# Gemma-4-31B-it bf16 mpk — the EXACT dir the vLLM arm served (custom tokenizer,
# trust_remote_code). Same hash on inf8 (H200) and inf14 (B300). Apples-to-apples.
MODEL=${MODEL:-/tinfoil/mpk/mpk-d2f38032f5faaa8a45b433157d0f25da1e533cf233784c7438fe9f46d1fbc3f4}
PORT=${PORT:-8001}
LOG=${LOG:-/dev/shm/serve.log}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML=${YAML:-$HERE/extra-llm-api.yaml}

export TMPDIR=${TMPDIR:-/dev/shm}
export HF_HOME=${HF_HOME:-/dev/shm/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/.cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/dev/shm/triton}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/dev/shm/fi}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"

if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[serve] already healthy on :${PORT}"; exit 0
fi
[ -e "$MODEL" ] || { echo "[serve] ERROR: model not mounted at $MODEL"; exit 1; }

# SMOKE: random weights — skips the real load but still builds the model + captures
# CUDA graphs, so the Gemma4 arch load + the head_dim=512 triton-attn path + CC
# graph-capture surface in ~3 min before committing to the full run.
EFFECTIVE_YAML="$YAML"
if [ "${SMOKE:-0}" = "1" ]; then
  EFFECTIVE_YAML=/dev/shm/extra-llm-api.smoke.yaml
  { cat "$YAML"; echo "load_format: dummy"; } > "$EFFECTIVE_YAML"
  echo "[serve] SMOKE: random weights (load_format: dummy)"
fi

echo "[serve] launching trtllm-serve $(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null) (Gemma TP=1, bf16, default KV, nospec)…"
nohup trtllm-serve "$MODEL" \
  --backend pytorch \
  --host 0.0.0.0 --port "$PORT" \
  --tp_size 1 --pp_size 1 \
  --trust_remote_code \
  --config "$EFFECTIVE_YAML" \
  >"$LOG" 2>&1 &
disown
echo "[serve] pid $! -> $LOG; waiting for /health (timeout 20m)"

for i in $(seq 1 240); do          # 240 * 5s = 20 min (bf16 ~58 GB load is faster than NVFP4)
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[serve] healthy after ~$((i * 5))s"
    echo "[serve] served model id(s):"; curl -s "http://localhost:${PORT}/v1/models" | jq -r '.data[].id' 2>/dev/null || true
    exit 0
  fi
  if ! pgrep -f "trtllm-serve" >/dev/null; then
    echo "[serve] ERROR: trtllm-serve exited early — last 50 log lines:"; tail -50 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[serve] ERROR: not healthy after 20m — see $LOG"; tail -50 "$LOG"; exit 1
