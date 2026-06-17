#!/usr/bin/env bash
set -uo pipefail

OUT=${1:?usage: collect_gpu_telemetry.sh <csv-out> [interval-sec]}
INTERVAL=${2:-2}

echo "timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu" > "$OUT"
while true; do
  nvidia-smi \
    --query-gpu=timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu \
    --format=csv,noheader,nounits >> "$OUT" 2>/dev/null || true
  sleep "$INTERVAL"
done
