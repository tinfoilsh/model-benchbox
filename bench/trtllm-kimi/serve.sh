#!/usr/bin/env bash
# bench/serve.sh — serve Kimi-K2.6-NVFP4 as a local in-process TensorRT-LLM
# (PyTorch backend) engine on the benchbox CVM's GPUs, so the bench client hits
# it over localhost (no TLS/shim/host-net) for a clean CC-overhead measurement.
# The TRT-LLM analog of model-benchbox/bench/serve.sh (which is vLLM). Serves :8001.
#
#   TP=8 ./serve.sh            # serve TP=8 + EP=8 (default; mirrors the vLLM Kimi arm)
#   TP=4 ./serve.sh            # the TP=4 parallelism-degree arm (§1g)
#   SMOKE=1 ./serve.sh         # dummy weights: validate arch loads + graph capture in ~3 min
#
# CLI flags verified against the local TensorRT-LLM 1.3.0rc19 source
# (tensorrt_llm/commands/serve.py). The CVM rootfs is read-only; all scratch goes
# to /dev/shm (large via ipc:host). Real NVFP4 load takes ~20 min; blocks until
# /health or a 40-minute timeout.
set -euo pipefail

# The Kimi-K2.6 NVFP4 mpk staged on the host (same dir the vLLM arm served — incl.
# the custom tiktoken tokenizer, loaded with trust_remote_code). Apples-to-apples.
MODEL=${MODEL:-/tinfoil/mpk/mpk-23653f0bad86c7ad6d4994a2607858662c56ed3ba205252f5714ba8636db35f8}
PORT=${PORT:-8001}
TP=${TP:-8}
EP=${EP:-$TP}                 # expert-parallel = TP (the vLLM arm used --enable-expert-parallel)
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

# SMOKE mode: random weights — skips the ~20-min real load but still builds the
# model + captures CUDA graphs, so kimi_k25 load + the CUTLASS NVFP4 MoE path +
# CC graph-capture surface in ~3 min. (load_format=dummy injected via a temp YAML.)
EFFECTIVE_YAML="$YAML"
if [ "${SMOKE:-0}" = "1" ]; then
  EFFECTIVE_YAML=/dev/shm/extra-llm-api.smoke.yaml
  { cat "$YAML"; echo "load_format: dummy"; } > "$EFFECTIVE_YAML"
  echo "[serve] SMOKE: random weights (load_format: dummy)"
fi

echo "[serve] launching trtllm-serve $(python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' 2>/dev/null) (TP=$TP, EP=$EP, NVFP4, fp8 KV)…"
nohup trtllm-serve "$MODEL" \
  --backend pytorch \
  --host 0.0.0.0 --port "$PORT" \
  --tp_size "$TP" --ep_size "$EP" --pp_size 1 \
  --kv_cache_dtype fp8 \
  --trust_remote_code \
  --config "$EFFECTIVE_YAML" \
  >"$LOG" 2>&1 &
disown
echo "[serve] pid $! -> $LOG; waiting for /health (timeout 40m)"

for i in $(seq 1 480); do          # 480 * 5s = 40 min
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
echo "[serve] ERROR: not healthy after 40m — see $LOG"; tail -50 "$LOG"; exit 1
