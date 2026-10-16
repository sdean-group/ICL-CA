#!/usr/bin/env bash
# Teacher-forced + autoregressive evaluation (paper metrics) on a freshly
# generated test split (100k trajectories by default).
#
#   bash scripts/eval.sh 1d-det-v2k3-shared                    # released model (downloaded from HF)
#   CKPT=checkpoints/<run>/<run>_best.pt bash scripts/eval.sh  # a checkpoint you trained (any architecture)
#   N=2000 bash scripts/eval.sh 2d-stoch-v2k5-shared            # quick check
#
# Results go to results/<model or run>.json; compare with configs/paper_results.json.
# All flags: python -m src.evaluate --help
set -euo pipefail

MODEL=""
if [[ $# -gt 0 && "$1" != -* ]]; then MODEL=$1; shift; fi
if [[ -z "$MODEL" && -z "${CKPT:-}" ]]; then
    echo "usage: bash scripts/eval.sh MODEL_KEY [flags]   or   CKPT=path.pt bash scripts/eval.sh [flags]" >&2
    exit 1
fi
OUT_DIR=${OUT_DIR:-results}
WANDB_MODE=${WANDB_MODE:-disabled}      # online | offline | disabled
mkdir -p "$OUT_DIR"
NAME=${MODEL:-$(basename "${CKPT%.pt}")}

ARGS=(--out "$OUT_DIR/$NAME.json" --wandb_mode "$WANDB_MODE")
[[ -n "$MODEL" ]] && ARGS+=(--model "$MODEL")
[[ -n "${CKPT:-}" ]] && ARGS+=(--ckpt "$CKPT")
[[ -n "${N:-}" ]] && ARGS+=(--n "$N")
[[ -n "${WANDB_ENTITY:-}" ]] && ARGS+=(--wandb_entity "$WANDB_ENTITY")

python -m src.evaluate "${ARGS[@]}" "$@"
