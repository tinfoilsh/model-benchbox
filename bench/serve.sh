#!/usr/bin/env bash
# bench/serve.sh — launch Kimi K2.6 (NVFP4) as a local in-process vLLM engine on
# the benchbox CVM's 8 GPUs, so the bench client can hit it over localhost (no
# TLS, no shim, no host networking) for a clean CC-overhead measurement.
#
# Serving args mirror the proven confidential-kimi-k2-6-b200 config, MINUS the
# eagle3 speculative-decoding block (left out of the first baseline — it changes
# ITL materially and we want a clean reference first). Serves on :8001.
#
# Model load (~0.5 TB NVFP4 into 8 GPUs + FlashInfer MoE warmup) takes minutes;
# this blocks until /health passes or a 30-minute timeout.
set -euo pipefail

MPK=${MPK:-/tinfoil/mpk/mpk-23653f0bad86c7ad6d4994a2607858662c56ed3ba205252f5714ba8636db35f8}
PORT=${PORT:-8001}
LOG=${LOG:-/workspace/serve.log}

if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[serve] already healthy on :${PORT}"; exit 0
fi
[ -e "$MPK" ] || { echo "[serve] ERROR: model not mounted at $MPK"; exit 1; }

echo "[serve] launching vLLM (TP=8, NVFP4, fp8 KV); model load takes several minutes…"
VLLM_USE_FLASHINFER_MOE_FP4=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
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
echo "[serve] pid $! -> logging to $LOG; waiting for /health (timeout 30m)"

for i in $(seq 1 360); do          # 360 * 5s = 30 min
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[serve] healthy after ~$((i * 5))s"; exit 0
  fi
  if ! kill -0 "$!" 2>/dev/null; then
    echo "[serve] ERROR: vllm process exited early — last 40 log lines:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[serve] ERROR: not healthy after 30m — see $LOG"; tail -40 "$LOG"; exit 1
