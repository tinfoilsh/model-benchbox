#!/bin/bash
# Round-3 single-GPU mechanism cells. Usage: bash run_all3.sh <cond>
set -uo pipefail
COND=${1:?usage: run_all3.sh <cond>}
M=/dev/shm/micro
OUT=/dev/shm/results/micro3-$COND
mkdir -p "$OUT"

echo "[run_all3] provenance"
bash $M/provenance.sh > "$OUT/provenance.txt" 2>&1

echo "[run_all3] membench3 (mechanism cells)"
python3 $M/membench3.py > "$OUT/membench3.json" 2> "$OUT/membench3.log" \
  || echo "[run_all3] membench3 FAILED rc=$?"

echo "[run_all3] mprocbench2 (d2h + core-pinned scaling)"
python3 $M/mprocbench2.py > "$OUT/mprocbench2.json" 2> "$OUT/mprocbench2.log" \
  || echo "[run_all3] mprocbench2 FAILED rc=$?"

touch "$OUT/done.marker"
echo "[run_all3] DONE -> $OUT"
ls -la "$OUT"
