#!/usr/bin/env bash
# bench/run_cc_experiment.sh — measure CC overhead for the locally-served TRT-LLM
# engine, mirroring the FULL B300 multi-GPU vLLM Kimi matrix so results diff 1:1:
#   short    128-in /128-out, conc {1,4}    -> <out>/short/guidellm-c<c>.json   (compare_cc.py, vs results/cc-*)
#   long    8192-in /512-out, conc {16,32}  -> <out>/long/guidellm-c<c>.json    (compare_cc.py, vs results/cc-*)
#   frontier 1024-in /256-out, conc {1,4,16,64,128} -> <out>/frontier/c<c>.json (compare_sweep.py, vs results/kimi-tp-cc-*)
#
#   Usage: ./run_cc_experiment.sh <cc-on|cc-off> [label]   (label e.g. tp8 / tp4 / tp8r2)
#          POINTS="short long frontier"  to run a subset.
#
# Run AFTER serve.sh reports healthy, in the SAME shell. Identical guidellm
# invocation to the vLLM arm (same instrument => the engine is the only variable).
set -uo pipefail

COND=${1:?usage: run_cc_experiment.sh <cc-on|cc-off> [label]}
LABEL=${2:-$(cut -c1-8 /proc/sys/kernel/random/uuid)}
PORT=${PORT:-8001}
MODEL=${MODEL:-/tinfoil/mpk/mpk-23653f0bad86c7ad6d4994a2607858662c56ed3ba205252f5714ba8636db35f8}
TOKENIZER=${TOKENIZER:-$MODEL}                 # local custom Kimi tokenizer (trust_remote_code)
GUIDELLM_SECS=${GUIDELLM_SECS:-60}             # vLLM B300 short/long used 60s; frontier used 75s — see run_point
SEED=${SEED:-0}                                # vLLM B300 arm used random-seed 0
POINTS=${POINTS:-"short long frontier"}
BASE="http://localhost:${PORT}"
OUT=${OUT:-/dev/shm/results/${COND}/${LABEL}}
export TMPDIR=${TMPDIR:-/dev/shm} HF_HOME=${HF_HOME:-/dev/shm/hf} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/.cache}
mkdir -p "$OUT"

curl -sf "${BASE}/health" >/dev/null || { echo "[run] server not healthy on ${BASE}; run serve.sh first"; exit 1; }
SERVED=$(curl -s "${BASE}/v1/models" | jq -r '.data[0].id' 2>/dev/null); SERVED=${SERVED:-$MODEL}
echo "[run] served model id = $SERVED"

echo "[run] === provenance -> $OUT ==="
nvidia-smi -q > "$OUT/nvidia-smi.q.txt" 2>&1 || true
nvidia-smi    > "$OUT/nvidia-smi.txt"   2>&1 || true
# Confirm the deploy flag took effect. On B300 `conf-compute -f` reports correctly;
# capture `-mgm` too (it is the authoritative readout, and it does NOT lie on Hopper).
{ echo "## conf-compute -f";   nvidia-smi conf-compute -f   2>&1
  echo "## conf-compute -mgm"; nvidia-smi conf-compute -mgm 2>&1
  echo "## tdx_guest"; grep -qi tdx_guest /proc/cpuinfo && echo present || echo absent; } > "$OUT/cc-mode.txt"
echo "[run] declared=$COND  cc-mode.txt:"; sed 's/^/[run]   /' "$OUT/cc-mode.txt"
python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' > "$OUT/trtllm-version.txt" 2>&1 || true
pip freeze 2>/dev/null | grep -iE "^(tensorrt-llm|tensorrt_llm|guidellm|flashinfer|torch)" > "$OUT/pkgs.txt" || true
cp "${LOG:-/dev/shm/serve.log}" "$OUT/serve.log" 2>/dev/null || true
curl -s "${BASE}/metrics" > "$OUT/metrics.before.txt" 2>&1 || true

bench_one () {  # subdir prompt out conc flat_naming secs
  local sub=$1 pin=$2 pout=$3 c=$4 flat=$5 secs=$6
  local pdir="$OUT/$sub"; mkdir -p "$pdir"
  local name; [ "$flat" = "flat" ] && name="c${c}" || name="guidellm-c${c}"
  echo "[run]   $sub concurrent=$c (in=$pin out=$pout ${secs}s)"
  guidellm benchmark run \
    --target "$BASE" --model "$SERVED" \
    --processor "$TOKENIZER" --processor-args '{"trust_remote_code": true}' \
    --rate-type concurrent --rate "$c" \
    --data "prompt_tokens=${pin},output_tokens=${pout}" \
    --max-seconds "$secs" --random-seed "$SEED" --disable-progress \
    --output-path "$pdir/${name}.json" > "$pdir/${name}.log" 2>&1 \
    && echo "[run]   ok -> $sub/${name}.json" \
    || { echo "[run]   WARN $sub c=$c failed (tail):"; tail -6 "$pdir/${name}.log"; }
}

for P in $POINTS; do
  case "$P" in
    short)    for c in 1 4;            do bench_one short    128  128 "$c" nested 60; done ;;
    long)     for c in 16 32;          do bench_one long     8192 512 "$c" nested 60; done ;;
    frontier) for c in 1 4 16 64 128;  do bench_one frontier 1024 256 "$c" flat   75; done ;;
    *) echo "[run] unknown point: $P" ;;
  esac
done

curl -s "${BASE}/metrics" > "$OUT/metrics.after.txt" 2>&1 || true
cat > "$OUT/manifest.json" <<EOF
{
  "condition": "${COND}",
  "label": "${LABEL}",
  "engine": "tensorrt-llm",
  "model": "${SERVED}",
  "tokenizer": "local:mpk",
  "served_via": "localhost in-process trtllm-serve (pytorch backend, no shim/TLS)",
  "points_run": "${POINTS}",
  "operating_points": {
    "short":    {"in": 128,  "out": 128, "concurrency": [1, 4]},
    "long":     {"in": 8192, "out": 512, "concurrency": [16, 32]},
    "frontier": {"in": 1024, "out": 256, "concurrency": [1, 4, 16, 64, 128]}
  }
}
EOF
echo "[run] done -> $OUT"
