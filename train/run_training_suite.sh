#!/usr/bin/env bash
set -euo pipefail

COND=${1:?usage: train/run_training_suite.sh <cc-on|cc-off> [label]}
LABEL=${2:-$(date -u +%Y%m%dT%H%M%SZ)}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$ROOT/run_gemma_training.sh" "$COND" smoke-e4b-peft "$LABEL"
"$ROOT/run_gemma_training.sh" "$COND" gemma31b-peft "$LABEL"
"$ROOT/run_gemma_training.sh" "$COND" gemma31b-sft "$LABEL"
