#!/usr/bin/env bash
# Train one of the released models from scratch (data is generated on the fly).
#
#   bash scripts/train.sh 1d-det-v2k3-shared                  # all visible GPUs
#   NGPU=1 bash scripts/train.sh 2d-stoch-v2k5-dedicated
#   WANDB_MODE=online WANDB_ENTITY=<you> bash scripts/train.sh 1d-det-v3k3-shared
#   bash scripts/train.sh 1d-det-v2k3-shared --epochs 1 --lr 5e-4   # extra flags override configs/
#
# Hyperparameters come from configs/pretrained_models.yaml; the global batch size
# is fixed there, so the number of GPUs does not change the optimization.
# List the model keys with: python -c "import src; print(src.list_pretrained())"
set -euo pipefail

MODEL=${1:?usage: bash scripts/train.sh MODEL_KEY [extra src.train flags]}
shift
if ! python -c "import torch, sys; sys.exit(not torch.cuda.is_available())"; then
    echo "ERROR: training needs a CUDA GPU, but PyTorch sees none." >&2
    echo "       Check that your torch build matches your NVIDIA driver (see README)." >&2
    exit 1
fi
NGPU=${NGPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
SAVE_DIR=${SAVE_DIR:-checkpoints}
WANDB_MODE=${WANDB_MODE:-disabled}      # online | offline | disabled
WANDB_PROJECT=${WANDB_PROJECT:-CA}

ARGS=(--model "$MODEL" --save_dir "$SAVE_DIR" --wandb_mode "$WANDB_MODE"
      --project_name "$WANDB_PROJECT")
[[ -n "${WANDB_ENTITY:-}" ]] && ARGS+=(--wandb_entity "$WANDB_ENTITY")

if [[ "$NGPU" -gt 1 ]]; then
    python -m torch.distributed.run --standalone --nproc_per_node "$NGPU" \
        -m src.train "${ARGS[@]}" "$@"
else
    python -m src.train "${ARGS[@]}" "$@"
fi
