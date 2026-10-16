#!/usr/bin/env bash
# Train a model from scratch (training data is generated on the fly).
#
# The first argument is either
#   * a released model key, e.g. 1d-det-v2k3-shared: its architecture and training
#     hyperparameters come from configs/pretrained_models.yaml, or
#   * a task name, e.g. 1d-det-v2k3: you choose the architecture; training
#     hyperparameters come from training_defaults in the same file.
# Any extra flag is passed to src.train and overrides the config.
#
#   bash scripts/train.sh 1d-det-v2k3-shared                        # reproduce a released model
#   bash scripts/train.sh 1d-det-v2k3-shared --lr 5e-4 --epochs 1   # ... with your own hyperparameters
#   bash scripts/train.sh 2d-det-v2k5 --hidden_size 512 \
#       --num_heads_per_layer 4 1 --use_mlp_per_layer 0 1           # your own architecture
#   NGPU=1 bash scripts/train.sh 2d-stoch-v2k5-dedicated            # number of GPUs (default: all)
#   WANDB_MODE=online WANDB_ENTITY=<you> bash scripts/train.sh 1d-det-v3k3-shared
#
# The batch size in the config is global, so the number of GPUs does not change
# the optimization. All flags: python -m src.train --help
set -euo pipefail

WHAT=${1:?usage: bash scripts/train.sh MODEL_KEY|TASK [extra src.train flags]}
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

if python -c "import sys; from src.registry import task_names; sys.exit('$WHAT' not in task_names())"; then
    WHICH=--task
else
    WHICH=--model
fi
ARGS=($WHICH "$WHAT" --save_dir "$SAVE_DIR" --wandb_mode "$WANDB_MODE"
      --project_name "$WANDB_PROJECT")
[[ -n "${WANDB_ENTITY:-}" ]] && ARGS+=(--wandb_entity "$WANDB_ENTITY")

if [[ "$NGPU" -gt 1 ]]; then
    # Direct GPU-to-GPU (P2P) transfers hang on some PCIe-only multi-GPU machines
    # (no NVLink): the first collective times out after 10 minutes. The models are
    # small, so routing through host memory costs little. Set NCCL_P2P_DISABLE=0
    # to re-enable P2P on machines where it works.
    export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
    python -m torch.distributed.run --standalone --nproc_per_node "$NGPU" \
        -m src.train "${ARGS[@]}" "$@"
else
    python -m src.train "${ARGS[@]}" "$@"
fi
