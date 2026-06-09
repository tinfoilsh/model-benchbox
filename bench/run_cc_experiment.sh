#!/usr/bin/env bash
# bench/run_cc_experiment.sh — measure NVIDIA confidential-computing overhead for
# a locally-served model, reporting TTFT and inter-token latency separately.
# Run this AFTER bench/serve.sh reports healthy.
#
#   ./run_cc_experiment.sh <cc-on|cc-off> [run-label]
#
# Deploy the SAME benchbox release twice — once normally (CC on) and once with
# `tinfoil container create … --disable-cc-mode --yes` (CC off) — and run this in
# each. On this Intel/B300 host that flag flips BOTH the GPU CC/PPCIe data path
# AND the CPU TDX memory encryption, so the on-vs-off delta is the full
# confidential-stack overhead. (Isolating the GPU-only part needs a third arm.)
#
# Two operating points:
#   short — 128-in / 128-out, low concurrency   (I/O-bound worst case; TTFT-heavy)
#   long  — 8192-in / 512-out, higher concurrency (compute-bound best case)
#
# Primary instrument: guidellm. Independent cross-check: vllm bench serve. Plus
# /metrics + nvidia-smi provenance (incl. CC-mode confirmation). All under
# results/<cond>/<label>/.
set -euo pipefail

COND=${1:?usage: run_cc_experiment.sh <cc-on|cc-off> [label]}
LABEL=${2:-$(cut -c1-8 /proc/sys/kernel/random/uuid)}     # no date/clock deps
PORT=${PORT:-8001}
MODEL=${MODEL:-kimi-k2-6}
TOKENIZER=${TOKENIZER:-moonshotai/Kimi-K2.6}              # NVFP4 shares this tokenizer; pulled via HF_TOKEN
GUIDELLM_SECS=${GUIDELLM_SECS:-120}
BASE="http://localhost:${PORT}"
OUT="/workspace/results/${COND}/${LABEL}"
mkdir -p "$OUT"

curl -sf "${BASE}/health" >/dev/null || { echo "[run] server not healthy on ${BASE}; run serve.sh first"; exit 1; }

echo "[run] === provenance -> $OUT ==="
nvidia-smi -q > "$OUT/nvidia-smi.q.txt" 2>&1 || true
nvidia-smi    > "$OUT/nvidia-smi.txt"   2>&1 || true
# Confirm the deploy flag actually took effect — the experiment is meaningless
# if the GPUs aren't in the CC mode we think they are.
CC_DETECT=$( (nvidia-smi conf-compute -f 2>/dev/null || grep -i "Confidential Compute" "$OUT/nvidia-smi.q.txt" | head -1) | tr -d '\n' || echo "unknown")
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
    guidellm benchmark \
      --target "$BASE" --rate-type concurrent --rate "$c" --max-seconds "$GUIDELLM_SECS" \
      --processor "$TOKENIZER" \
      --data "prompt_tokens=${pin},output_tokens=${pout}" \
      --output-path "$pdir/guidellm-c${c}.json" \
      >"$pdir/guidellm-c${c}.log" 2>&1 || echo "[run]   WARN guidellm c=$c failed (see log)"
  done
  local ctop=${concs[${#concs[@]}-1]}
  echo "[run]   vllm bench serve cross-check (max-conc=$ctop)"
  vllm bench serve --base-url "$BASE" --model "$MODEL" --tokenizer "$TOKENIZER" \
    --dataset-name random --random-input-len "$pin" --random-output-len "$pout" \
    --max-concurrency "$ctop" --num-prompts $((ctop * 4)) --seed 0 \
    --save-result --result-dir "$pdir" \
    >"$pdir/vllm-bench-c${ctop}.log" 2>&1 || echo "[run]   WARN vllm bench failed (see log)"
}

run_point short 128  128 1 4
run_point long  8192 512 16 32

curl -s "${BASE}/metrics" > "$OUT/metrics.after.txt" 2>&1 || true

cat > "$OUT/manifest.json" <<EOF
{
  "condition": "${COND}",
  "label": "${LABEL}",
  "model": "${MODEL}",
  "tokenizer": "${TOKENIZER}",
  "tensor_parallel": 8,
  "served_via": "localhost in-process vllm (no shim/TLS)",
  "operating_points": {
    "short": {"in": 128, "out": 128, "concurrency": [1, 4]},
    "long":  {"in": 8192, "out": 512, "concurrency": [16, 32]}
  },
  "detected_cc": "$(printf '%s' "${CC_DETECT:-unknown}" | sed 's/"/\\"/g')"
}
EOF
echo "[run] done -> $OUT"
echo "[run] save it out before teardown:  gh release upload  OR  scp from your laptop (ramdisk is volatile)"
