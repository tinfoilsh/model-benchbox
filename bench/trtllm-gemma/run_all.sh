#!/usr/bin/env bash
# run_all.sh <cc-on|cc-off> — SMOKE-test, then serve Gemma-4-31B (bf16, nospec, TP=1)
# and run the full bench matrix. Run INSIDE the deployed TRT-LLM benchbox CVM over
# debug-SSH (after injecting this dir's scripts). Results -> /dev/shm (volatile: tar out).
set -euo pipefail
COND=${1:?usage: run_all.sh <cc-on|cc-off>}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[run_all] (1/3) SMOKE: validate Gemma4 load + CUDA-graph capture (dummy weights, ~3 min)…"
SMOKE=1 bash "$HERE/serve.sh" || { echo "[run_all] SMOKE failed — fix before the real load"; exit 1; }
pkill -9 -f trtllm-serve 2>/dev/null || true
for i in $(seq 1 30); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null | head -1)
  [ "${u:-0}" -lt 2000 ] && break; sleep 5
done

echo "[run_all] (2/3) serve Gemma (real bf16 weights, ~5-10 min)…"
bash "$HERE/serve.sh"

echo "[run_all] (3/3) bench (nospec, full matrix)…"
LOG=/dev/shm/serve.log bash "$HERE/bench.sh" "$COND" nospec full

echo "[run_all] done -> /dev/shm/results/${COND}/nospec"
echo "[run_all] TAR IT OUT (ramdisk is volatile):  tar czf /tmp/${COND}-nospec.tgz -C /dev/shm/results ${COND}"
