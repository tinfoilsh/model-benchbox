#!/bin/bash
# Round-2 microbench orchestrator. Usage: bash run_all2.sh <cc-on|cc-off>
set -uo pipefail
COND=${1:?usage: run_all2.sh cc-on|cc-off}
M=/dev/shm/micro
OUT=/dev/shm/results/micro2-$COND
mkdir -p "$OUT"

echo "[run_all2] membench2 (knee/multistream/overlap/cpu-per-byte)"
python3 $M/membench2.py > "$OUT/membench2.json" 2> "$OUT/membench2.log" \
  || echo "[run_all2] membench2 FAILED rc=$?"

echo "[run_all2] mprocbench (multi-process pinned H2D scaling)"
python3 $M/mprocbench.py > "$OUT/mprocbench.json" 2> "$OUT/mprocbench.log" \
  || echo "[run_all2] mprocbench FAILED rc=$?"

echo "[run_all2] cpubench (pure-CPU, isolates TDX tax)"
python3 $M/cpubench.py > "$OUT/cpubench.json" 2> "$OUT/cpubench.log" \
  || echo "[run_all2] cpubench FAILED rc=$?"

echo "[run_all2] collectives round 2 (variance check)"
NCCL_DEBUG=WARN torchrun --standalone --nproc-per-node=8 $M/collectives.py \
  > "$OUT/collectives-r2.json" 2> "$OUT/collectives-r2.log" \
  || echo "[run_all2] collectives-r2 FAILED rc=$?"

if [ "$COND" = "cc-off" ]; then
  echo "[run_all2] collectives NVLS ablation (NCCL_NVLS_ENABLE=0)"
  NCCL_NVLS_ENABLE=0 NCCL_DEBUG=INFO NCCL_DEBUG_FILE="$OUT/nccl-nonvls.%h.%p.log" \
    torchrun --standalone --nproc-per-node=8 $M/collectives.py \
    > "$OUT/collectives-nonvls.json" 2> "$OUT/collectives-nonvls.log" \
    || echo "[run_all2] collectives-nonvls FAILED rc=$?"
fi

touch "$OUT/done.marker"
echo "[run_all2] DONE -> $OUT"
ls -la "$OUT"
