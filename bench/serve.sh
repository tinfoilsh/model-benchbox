#!/usr/bin/env bash
# bench/serve.sh — serve Kimi K2.6 (NVFP4) as a local in-process vLLM engine on
# the benchbox CVM's 8 GPUs, so the bench client hits it over localhost (no TLS,
# no shim, no host networking) for a clean CC-overhead measurement.
#
# Production-faithful (vLLM 0.21.0): FlashInfer NVFP4 MoE + expert parallelism +
# CUDA graphs, mirroring confidential-kimi-k2-6-b200, MINUS eagle3 speculative
# decoding (left out of the first baseline). Serves on :8001.
#
# The CVM rootfs is read-only; all scratch/caches go to /dev/shm (large via
# ipc:host). Model load (~0.5 TB NVFP4 into 8 GPUs) takes ~20 min; this blocks
# until /health passes or a 35-minute timeout.
set -euo pipefail

MPK=${MPK:-/tinfoil/models/kimi-k2-6}
PORT=${PORT:-8001}
LOG=${LOG:-/dev/shm/serve.log}

export TMPDIR=${TMPDIR:-/dev/shm}
export HF_HOME=${HF_HOME:-/dev/shm/hf}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/.cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/dev/shm/vllm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/dev/shm/triton}
export VLLM_USE_FLASHINFER_MOE_FP4=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[serve] already healthy on :${PORT}"; exit 0
fi
[ -e "$MPK" ] || { echo "[serve] ERROR: model not mounted at $MPK"; exit 1; }

echo "[serve] launching vLLM $(vllm --version 2>/dev/null || echo '?') (TP=8, NVFP4, fp8 KV, EP); load takes ~20 min…"
nohup vllm serve "$MPK" \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 48 \
  --mm-encoder-tp-mode data \
  --trust-remote-code \
  --served-model-name kimi-k2-6 \
  --port "$PORT" \
  >"$LOG" 2>&1 &
disown
echo "[serve] pid $! -> logging to $LOG; waiting for /health (timeout 35m)"

for i in $(seq 1 420); do          # 420 * 5s = 35 min
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[serve] healthy after ~$((i * 5))s"; exit 0
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "[serve] ERROR: vllm exited early — last 40 log lines:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[serve] ERROR: not healthy after 35m — see $LOG"; tail -40 "$LOG"; exit 1
