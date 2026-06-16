#!/usr/bin/env bash
# bench.sh — bench the running single-GPU Gemma TRT-LLM server with TRT-LLM's
# benchmark_serving (a FORK of `vllm bench serve` — identical TPOT/ITL/TTFT/
# throughput math), so results diff 1:1 against the vLLM Gemma baseline
# (results/{,hopper-}gemma-cc-*/nospec/). SAME operating points + seed as
# gemma-probe/gemma_bench.sh; the engine is the only variable.
#
#   Usage: bench.sh <cc-on|cc-off> [label=nospec] [full|fixed]
#   Run AFTER serve.sh reports healthy, in the SAME shell.
set -uo pipefail

COND=${1:?usage: bench.sh <cc-on|cc-off> [label] [full|fixed]}
LABEL=${2:-nospec}
PROG=${3:-full}
PORT=${PORT:-8001}
MODEL=${MODEL:-/tinfoil/mpk/mpk-d2f38032f5faaa8a45b433157d0f25da1e533cf233784c7438fe9f46d1fbc3f4}
TOKENIZER=${TOKENIZER:-$MODEL}                  # local Gemma tokenizer (trust_remote_code)
SEED=${SEED:-0}                                 # vLLM Gemma arm used seed 0
BASE="http://localhost:${PORT}"
OUT=${OUT:-/dev/shm/results/${COND}/${LABEL}}
export TMPDIR=${TMPDIR:-/dev/shm} HF_HOME=${HF_HOME:-/dev/shm/hf}
mkdir -p "$OUT"

curl -sf "${BASE}/health" >/dev/null || { echo "[bench] server not healthy on ${BASE}; run serve.sh first"; exit 1; }
SERVED=$(curl -s "${BASE}/v1/models" | jq -r '.data[0].id' 2>/dev/null); SERVED=${SERVED:-$MODEL}
echo "[bench] served model id = $SERVED"

# --- provenance (mirrors trtllm-benchbox/bench/run_cc_experiment.sh) ---
echo "[bench] === provenance -> $OUT ==="
nvidia-smi -q > "$OUT/nvidia-smi.q.txt" 2>&1 || true
nvidia-smi    > "$OUT/nvidia-smi.txt"   2>&1 || true
{ echo "## conf-compute -f";   nvidia-smi conf-compute -f   2>&1
  echo "## conf-compute -mgm"; nvidia-smi conf-compute -mgm 2>&1
  echo "## tdx_guest"; grep -qi tdx_guest /proc/cpuinfo && echo present || echo absent; } > "$OUT/cc-mode.txt"
echo "[bench] declared=$COND  cc-mode.txt:"; sed 's/^/[bench]   /' "$OUT/cc-mode.txt"
python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' > "$OUT/trtllm-version.txt" 2>&1 || true
pip freeze 2>/dev/null | grep -iE "^(tensorrt-llm|tensorrt_llm|torch|flashinfer|transformers)" > "$OUT/pkgs.txt" || true
cp "${LOG:-/dev/shm/serve.log}" "$OUT/serve.log" 2>/dev/null || true
curl -s "${BASE}/metrics" > "$OUT/metrics.before.txt" 2>&1 || true

run() {  # in out conc nprompts label
  local pin=$1 pout=$2 c=$3 n=$4 name=$5
  echo "[bench] $name (in=$pin out=$pout c=$c n=$n)"
  # --backend openai => /v1/completions (NOT openai-chat: completions matches the
  # vLLM Gemma arm's endpoint, so prompt token counts are identical).
  python3 -m tensorrt_llm.serve.scripts.benchmark_serving \
    --backend openai --base-url "$BASE" --model "$SERVED" \
    --tokenizer "$TOKENIZER" --trust-remote-code \
    --dataset-name random --random-range-ratio 0.0 --random-ids --ignore-eos \
    --random-input-len "$pin" --random-output-len "$pout" \
    --max-concurrency "$c" --num-prompts "$n" \
    --request-rate inf --burstiness 1.0 --seed "$SEED" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
    --save-result --result-filename "$OUT/$name.json" \
    > "$OUT/$name.log" 2>&1 \
    && echo "[bench]   ok -> $name.json" \
    || { echo "[bench]   WARN $name failed (tail):"; tail -8 "$OUT/$name.log"; }
}

# Operating points IDENTICAL to gemma-probe/gemma_bench.sh (so the vLLM baseline diffs 1:1).
if [ "$PROG" = "full" ]; then
  for rep in 1 2 3; do
    run 128  128 1  32  "short-c1-r$rep"
    run 128  128 8  64  "short-c8-r$rep"
    run 8192 512 8  24  "long-c8-r$rep"
    run 8192 512 32 64  "long-c32-r$rep"
  done
  for c in 1 4 16 32 64 128; do
    run 1024 256 "$c" $(( c*6 > 24 ? c*6 : 24 )) "frontier-c$c"
  done
else
  run 128  128 1  32  "short-c1-r1"
  run 128  128 8  64  "short-c8-r1"
  run 8192 512 8  24  "long-c8-r1"
  run 8192 512 32 64  "long-c32-r1"
fi

curl -s "${BASE}/metrics" > "$OUT/metrics.after.txt" 2>&1 || true
cat > "$OUT/manifest.json" <<EOF
{
  "condition": "${COND}", "label": "${LABEL}", "engine": "tensorrt-llm",
  "model": "${SERVED}", "quant": "bf16", "kv_dtype": "bf16(default)", "tp": 1,
  "instrument": "tensorrt_llm.serve.scripts.benchmark_serving (fork of vllm bench serve)",
  "served_via": "localhost in-process trtllm-serve (pytorch backend, no shim/TLS)",
  "program": "${PROG}",
  "operating_points": {
    "short":    {"in": 128,  "out": 128, "concurrency": [1, 8]},
    "long":     {"in": 8192, "out": 512, "concurrency": [8, 32]},
    "frontier": {"in": 1024, "out": 256, "concurrency": [1, 4, 16, 32, 64, 128]}
  }
}
EOF
echo "[bench] done -> $OUT"
