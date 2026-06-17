#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: train/run_gemma_training.sh <cc-on|cc-off> <smoke-e4b-peft|gemma31b-peft|gemma31b-sft> [label]

Environment overrides:
  NPROC_PER_NODE       torchrun worker count (default: 8)
  SMOKE_STEPS          default 50
  GEMMA31B_PEFT_STEPS  default 200
  GEMMA31B_SFT_STEPS   default 100
  GEMMA4_31B_MODEL     default: local MPK if mounted, else google/gemma-4-31B-it
  GEMMA4_E4B_MODEL     default: google/gemma-4-E4B-it
  PIN_MEMORY           dataloader pin_memory override, true/false (default: true)
  WARMUP_STEPS         step records to discard in summary (default: 10)
EOF
}

COND=${1:-}
SCENARIO=${2:-}
LABEL=${3:-$(date -u +%Y%m%dT%H%M%SZ)}
[ -n "$COND" ] && [ -n "$SCENARIO" ] || { usage; exit 2; }
case "$COND" in cc-on|cc-off) ;; *) usage; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$ROOT/configs"
SCRIPTS_DIR="$ROOT/scripts"
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
PIN_MEMORY=${PIN_MEMORY:-true}
WARMUP_STEPS=${WARMUP_STEPS:-10}

LOCAL_31B="/tinfoil/mpk/mpk-d2f38032f5faaa8a45b433157d0f25da1e533cf233784c7438fe9f46d1fbc3f4"
if [ -e "$LOCAL_31B" ]; then
  DEFAULT_31B="$LOCAL_31B"
else
  DEFAULT_31B="google/gemma-4-31B-it"
fi
GEMMA4_31B_MODEL=${GEMMA4_31B_MODEL:-$DEFAULT_31B}
GEMMA4_E4B_MODEL=${GEMMA4_E4B_MODEL:-google/gemma-4-E4B-it}

case "$SCENARIO" in
  smoke-e4b-peft)
    CONFIG="$CONFIG_DIR/gemma4_e4b_peft.yaml"
    MODEL="$GEMMA4_E4B_MODEL"
    STEPS=${SMOKE_STEPS:-50}
    ;;
  gemma31b-peft)
    CONFIG="$CONFIG_DIR/gemma4_31b_peft.yaml"
    MODEL="$GEMMA4_31B_MODEL"
    STEPS=${GEMMA31B_PEFT_STEPS:-200}
    ;;
  gemma31b-sft)
    CONFIG="$CONFIG_DIR/gemma4_31b_sft.yaml"
    MODEL="$GEMMA4_31B_MODEL"
    STEPS=${GEMMA31B_SFT_STEPS:-100}
    ;;
  *)
    usage; exit 2
    ;;
esac

export TMPDIR=${TMPDIR:-/dev/shm/tmp}
export HF_HOME=${HF_HOME:-/dev/shm/hf}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/dev/shm/hf/hub}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/dev/shm/.cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/dev/shm/inductor}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/dev/shm/triton}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

OUT=${OUT:-/dev/shm/results/train/${COND}/${LABEL}/${SCENARIO}}
mkdir -p "$OUT"

AUTOMODEL_DIR=${AUTOMODEL_DIR:-/opt/Automodel}
if [ ! -f "$AUTOMODEL_DIR/examples/vlm_finetune/finetune.py" ]; then
  AUTOMODEL_DIR=/dev/shm/Automodel
  if [ ! -f "$AUTOMODEL_DIR/examples/vlm_finetune/finetune.py" ]; then
    git clone --depth 1 https://github.com/NVIDIA-NeMo/Automodel.git "$AUTOMODEL_DIR"
  fi
fi

cp "$CONFIG" "$OUT/config.yaml"
"$SCRIPTS_DIR/provenance.sh" > "$OUT/provenance.txt" 2>&1 || true

CC_MGM="$(nvidia-smi conf-compute -mgm 2>&1 || true)"
CC_F="$(nvidia-smi conf-compute -f 2>&1 || true)"
printf '%s\n\n%s\n' "$CC_MGM" "$CC_F" > "$OUT/cc-mode.txt"

cat > "$OUT/manifest.json" <<EOF
{
  "condition": "$COND",
  "scenario": "$SCENARIO",
  "label": "$LABEL",
  "model": "$MODEL",
  "config": "$CONFIG",
  "steps": $STEPS,
  "nproc_per_node": $NPROC_PER_NODE,
  "pin_memory": "$PIN_MEMORY",
  "automodel_dir": "$AUTOMODEL_DIR",
  "output_dir": "$OUT",
  "cc_mgm": $(CC_MGM="$CC_MGM" python3 -c 'import json,os; print(json.dumps(os.environ.get("CC_MGM", "")))')
}
EOF

echo "[train] condition=$COND scenario=$SCENARIO label=$LABEL"
echo "[train] model=$MODEL"
echo "[train] steps=$STEPS nproc=$NPROC_PER_NODE out=$OUT"

"$SCRIPTS_DIR/collect_gpu_telemetry.sh" "$OUT/gpu-telemetry.csv" "${TELEMETRY_INTERVAL:-2}" &
TELEM_PID=$!
cleanup() {
  kill "$TELEM_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

START_EPOCH=$(date +%s)
set +e
torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" \
  "$AUTOMODEL_DIR/examples/vlm_finetune/finetune.py" \
  -c "$CONFIG" \
  --model.pretrained_model_name_or_path "$MODEL" \
  --step_scheduler.max_steps "$STEPS" \
  --step_scheduler.num_epochs 1 \
  --step_scheduler.ckpt_every_steps 1000000 \
  --step_scheduler.val_every_steps 1000000 \
  --checkpoint.enabled false \
  --checkpoint.checkpoint_dir "/dev/shm/checkpoints/${COND}/${LABEL}/${SCENARIO}" \
  --dataloader.pin_memory "$PIN_MEMORY" \
  2>&1 | tee "$OUT/train.log"
RC=${PIPESTATUS[0]}
set -e
END_EPOCH=$(date +%s)
cleanup
trap - EXIT

python3 "$SCRIPTS_DIR/summarize_log.py" "$OUT/train.log" "$OUT/summary.json" "$WARMUP_STEPS" > "$OUT/summary.stdout.txt" 2>&1 || true

python3 - "$OUT/manifest.json" "$RC" "$START_EPOCH" "$END_EPOCH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
data["returncode"] = int(sys.argv[2])
data["start_epoch"] = int(sys.argv[3])
data["end_epoch"] = int(sys.argv[4])
data["wall_seconds"] = data["end_epoch"] - data["start_epoch"]
path.write_text(json.dumps(data, indent=2) + "\n")
PY

echo "[train] returncode=$RC wall=$((END_EPOCH - START_EPOCH))s -> $OUT"
exit "$RC"
