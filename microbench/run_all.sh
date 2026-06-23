#!/bin/bash
# Orchestrates the microbench suite inside the benchbox container.
# Usage: bash run_all.sh <cc-on|cc-off>   (invoke via bash — /dev/shm is noexec)
set -uo pipefail
COND=${1:?usage: run_all.sh cc-on|cc-off}
M=/dev/shm/micro
OUT=/dev/shm/results/micro-$COND
mkdir -p "$OUT"

echo "[run_all] provenance"
bash $M/provenance.sh > "$OUT/provenance.txt" 2>&1

echo "[run_all] membench (H2D/D2H/D2D sweep)"
python3 $M/membench.py > "$OUT/membench.json" 2> "$OUT/membench.log" \
  || echo "[run_all] membench FAILED rc=$?"

echo "[run_all] collectives (8-GPU all-reduce + all-to-all, NCCL_DEBUG=INFO)"
NCCL_DEBUG=INFO NCCL_DEBUG_FILE="$OUT/nccl.%h.%p.log" \
  torchrun --standalone --nproc-per-node=8 $M/collectives.py \
  > "$OUT/collectives.json" 2> "$OUT/collectives.log" \
  || echo "[run_all] collectives FAILED rc=$?"

echo "[run_all] launchbench (launch latency + graph capture/replay)"
python3 $M/launchbench.py > "$OUT/launchbench.json" 2> "$OUT/launchbench.log" \
  || echo "[run_all] launchbench FAILED rc=$?"

touch "$OUT/done.marker"
echo "[run_all] DONE -> $OUT"
ls -la "$OUT"
