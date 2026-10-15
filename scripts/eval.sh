#!/usr/bin/env bash
# Teacher-forced + autoregressive evaluation (paper metrics) on a freshly
# generated test split (100k trajectories by default).
#
#   bash scripts/eval.sh 1d-det-v2k3-shared                   # released checkpoint from HF
#   CKPT=checkpoints/<run>/<run>_best.pt bash scripts/eval.sh 1d-det-v2k3-shared
#   N=2000 bash scripts/eval.sh 2d-stoch-v2k5-shared           # quick check
#
# Results go to results/<model>.json; compare with configs/paper_results.json.
set -euo pipefail

MODEL=${1:?usage: bash scripts/eval.sh MODEL_KEY [extra src.evaluate flags]}
shift
OUT_DIR=${OUT_DIR:-results}
WANDB_MODE=${WANDB_MODE:-disabled}      # online | offline | disabled
mkdir -p "$OUT_DIR"

ARGS=(--model "$MODEL" --out "$OUT_DIR/$MODEL.json" --wandb_mode "$WANDB_MODE")
[[ -n "${CKPT:-}" ]] && ARGS+=(--ckpt "$CKPT")
[[ -n "${N:-}" ]] && ARGS+=(--n "$N")
[[ -n "${WANDB_ENTITY:-}" ]] && ARGS+=(--wandb_entity "$WANDB_ENTITY")

python -m src.evaluate "${ARGS[@]}" "$@"
