#!/usr/bin/env bash
# bench/run_cc_experiment.sh — measure NVIDIA confidential-computing overhead for
# the locally-served model, reporting TTFT and inter-token latency separately.
# Run AFTER bench/serve.sh reports healthy.
#
#   ./run_cc_experiment.sh <cc-on|cc-off> [run-label]
#
# Deploy the SAME benchbox release twice — once normally (CC on) and once with
# `tinfoil container create … --disable-cc-mode --yes` (CC off) — and run this in
# each. On this Intel/B300 host that flag flips BOTH the GPU CC/PPCIe data path
# AND the CPU TDX memory encryption, so on-vs-off = full confidential-stack
# overhead. (Isolating the GPU-only part needs a third arm.)
#
# Operating points:
#   short — 128-in / 128-out, low concurrency   (I/O-bound worst case; TTFT-heavy)
#   long  — 8192-in / 512-out, higher concurrency (compute-bound best case)
#
# Primary instrument: guidellm. Tokenizer is loaded locally from the mpk dir
# (custom Kimi tokenizer, trust_remote_code) so no HF dependency. Results +
# /metrics + nvidia-smi (incl. CC-mode confirmation) land under
# /dev/shm/results/<cond>/<label>/ — save them out before teardown.
set -uo pipefail

COND=${1:?usage: run_cc_experiment.sh <cc-on|cc-off> [label]}
LABEL=${2:-$(cut -c1-8 /proc/sys/kernel/random/uuid)}     # no date/clock deps
PORT=${PORT:-8001}
MODEL=${MODEL:-kimi-k2-6}
MPK=${MPK:-/tinfoil/mpk/mpk-23653f0bad86c7ad6d4994a2607858662c56ed3ba205252f5714ba8636db35f8}
TOKENIZER=${TOKENIZER:-$MPK}                              # local custom tokenizer
GUIDELLM_SECS=${GUIDELLM_SECS:-60}
BASE="http://localhost:${PORT}"
OUT=${OUT:-/dev/shm/results/${COND}/${LABEL}}
export TMPDIR=${TMPDIR:-/dev/shm} HF_HOME=${HF_HOME:-/dev/shm/hf} XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/.cache}
mkdir -p "$OUT"

curl -sf "${BASE}/health" >/dev/null || { echo "[run] server not healthy on ${BASE}; run serve.sh first"; exit 1; }

echo "[run] === provenance -> $OUT ==="
nvidia-smi -q > "$OUT/nvidia-smi.q.txt" 2>&1 || true
nvidia-smi    > "$OUT/nvidia-smi.txt"   2>&1 || true
# Confirm the deploy flag took effect — the experiment is meaningless if the
# GPUs aren't in the CC mode we think they are.
CC_DETECT=$( (nvidia-smi conf-compute -f 2>/dev/null || grep -i "Confidential Compute" "$OUT/nvidia-smi.q.txt" | head -1) | tr -d '\n')
echo "${CC_DETECT:-unknown}" > "$OUT/cc-mode.txt"
echo "[run] declared=$COND  detected=[${CC_DETECT:-unknown}]"
vllm --version > "$OUT/vllm-version.txt" 2>&1 || true
pip freeze 2>/dev/null | grep -iE "^(vllm|guidellm|flashinfer|torch)" > "$OUT/pkgs.txt" || true
curl -s "${BASE}/metrics" > "$OUT/metrics.before.txt" 2>&1 || true

run_point () {
  local name=$1 pin=$2 pout=$3; shift 3; local concs=("$@")
  local pdir="$OUT/$name"; mkdir -p "$pdir"
  echo "[run] --- operating point: $name (in=$pin out=$pout, conc=${concs[*]}) ---"
  for c in "${concs[@]}"; do
    echo "[run]   guidellm concurrent=$c"
    guidellm benchmark run \
      --target "$BASE" --model "$MODEL" \
      --processor "$TOKENIZER" --processor-args '{"trust_remote_code": true}' \
      --rate-type concurrent --rate "$c" \
      --data "prompt_tokens=${pin},output_tokens=${pout}" \
      --max-seconds "$GUIDELLM_SECS" --random-seed 0 \
      --output-path "$pdir/guidellm-c${c}.json" \
      --disable-progress >"$pdir/guidellm-c${c}.log" 2>&1 \
      && echo "[run]   ok -> guidellm-c${c}.json" \
      || { echo "[run]   WARN guidellm c=$c failed (tail):"; tail -6 "$pdir/guidellm-c${c}.log"; }
  done
}

run_point short 128  128 1 4
run_point long  8192 512 16 32

curl -s "${BASE}/metrics" > "$OUT/metrics.after.txt" 2>&1 || true

cat > "$OUT/manifest.json" <<EOF
{
  "condition": "${COND}",
  "label": "${LABEL}",
  "model": "${MODEL}",
  "tokenizer": "local:mpk",
  "tensor_parallel": 8,
  "expert_parallel": true,
  "served_via": "localhost in-process vllm 0.21.0 (no shim/TLS)",
  "operating_points": {
    "short": {"in": 128, "out": 128, "concurrency": [1, 4]},
    "long":  {"in": 8192, "out": 512, "concurrency": [16, 32]}
  },
  "detected_cc": "$(printf '%s' "${CC_DETECT:-unknown}" | sed 's/"/\\"/g')"
}
EOF
echo "[run] done -> $OUT  (save it out before teardown; ramdisk is volatile)"
