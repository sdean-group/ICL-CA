"""Train a CA-Transformer with training data generated on the fly.

Ported from the repo-root train.py. The training loop, DDP, torch.compile,
gradient accumulation, resume, step-granular eval/save, best-ckpt selection,
patience and final test evaluation are unchanged; what changed:
  * no data files: the train split is simulated on the fly and val/test splits
    are generated in memory from a fixed seed (src/data/streaming.py);
  * the task and the paper hyperparameters come from configs/ via --model
    (any flag given on the command line overrides them);

Reproduce a released model (single GPU / multi-GPU):
    python -m src.train --model 1d-det-v2k3-shared --save_dir ckpts
    python -m torch.distributed.run --nproc_per_node=4 -m src.train \\
        --model 2d-stoch-v2k5-shared --save_dir ckpts

New architecture on an existing task:
    python -m src.train --task 1d-stoch-v3k3 --hidden_size 512 \\
        --num_heads_per_layer 2 1 --use_mlp_per_layer 0 1 --save_dir ckpts

Quick smoke test (CPU ok):
    python -m src.train --model 1d-det-v2k3-shared --max_train_steps 20 \\
        --samples_per_epoch 20480 --val_size 512 --test_size 512 \\
        --eval_steps 10 --no_compile --wandb_mode disabled --save_dir /tmp/ca_smoke
"""

import argparse
import dataclasses
import glob
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .data.streaming import (EvalSplit, OnTheFlyTrainDataset, default_select_split,
                             default_val_splits)
from .evaluators import evaluate
from .model import CATransformer
from .registry import get_spec, specs, task_names, task_spec, training_config, _raw_registry


# ---------------------------------------------------------------------------
# DDP helpers. Single-GPU (no torchrun env vars) returns single-process defaults.
# ---------------------------------------------------------------------------
def is_ddp():
    return all(v in os.environ for v in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def get_rank():
    return int(os.environ["RANK"]) if is_ddp() else 0


def get_world_size():
    return int(os.environ["WORLD_SIZE"]) if is_ddp() else 1


def get_local_rank():
    return int(os.environ["LOCAL_RANK"]) if is_ddp() else 0


def is_rank0():
    return get_rank() == 0


_WANDB = None


def wlog(d):
    """Rank-0-only wandb.log; no-op when wandb is disabled."""
    if is_rank0() and _WANDB is not None:
        _WANDB.log(d)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def prune_ckpts(save_dir, run_name, keep_last):
    """Keep only the `keep_last` most-recently-written _ep*.pt / _step*.pt
    ckpts; _best.pt / _final.pt are never pruned. No-op when keep_last <= 0."""
    if keep_last <= 0:
        return
    d = f"{save_dir}/{run_name}"
    rotating = (glob.glob(f"{d}/{run_name}_ep*.pt")
                + glob.glob(f"{d}/{run_name}_step*.pt"))
    if len(rotating) <= keep_last:
        return
    rotating.sort(key=os.path.getmtime)
    for p in rotating[:-keep_last]:
        try:
            os.remove(p)
            print(f"  [prune] removed {os.path.basename(p)}")
        except OSError as e:
            print(f"  [prune] failed to remove {p}: {e}")


def get_scheduler(name, optimizer, num_warmup_steps, num_training_steps):
    """Same lr curves as transformers.get_scheduler for these names."""
    def warm(step):
        return step / max(1, num_warmup_steps)

    if name == "cosine":
        def f(step):
            if step < num_warmup_steps:
                return warm(step)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    elif name == "linear":
        def f(step):
            if step < num_warmup_steps:
                return warm(step)
            return max(0.0, (num_training_steps - step) / max(1, num_training_steps - num_warmup_steps))
    elif name == "constant_with_warmup":
        def f(step):
            return warm(step) if step < num_warmup_steps else 1.0
    elif name == "constant":
        def f(step):
            return 1.0
    else:
        raise ValueError(f"unknown scheduler '{name}'")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def get_temperature(epoch, total_epochs, init=1.0, final=1.0, anneal_frac=1.0):
    """Linear anneal from init to final over the first anneal_frac x total_epochs."""
    if init == final:
        return init
    anneal_epochs = max(1, int(total_epochs * anneal_frac))
    progress = min(epoch / anneal_epochs, 1.0)
    return init + (final - init) * progress


def train_epoch(model, train_loader, optimizer, criterion, device, epoch,
                output_size, scheduler=None, attn_temperature=1.0, global_step=0,
                eval_steps=0, eval_callback=None, accum_steps=1,
                save_steps=0, save_callback=None, max_train_steps=0):
    model.train()
    last_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    amp = device.type == "cuda"

    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        # On-the-fly train batches are (X, y): every target position is scored
        # (the original on-disk train shards had an all-ones mask too).
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            output = model(X, attn_temperature=attn_temperature)
        output = output.float()
        loss = criterion(output.view(-1, output_size), y.view(-1)).mean()
        (loss / accum_steps).backward()
        accum_loss += loss.detach()

        if (batch_idx + 1) % accum_steps != 0:
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        last_loss = (accum_loss / accum_steps).item()
        accum_loss = 0.0
        wlog({"train/minibatch_loss": last_loss, "train/step": global_step})
        global_step += 1

        if eval_steps and eval_callback is not None and global_step % eval_steps == 0:
            eval_callback(global_step, attn_temperature)
            model.train()

        if save_steps and save_callback is not None and global_step % save_steps == 0:
            save_callback(global_step, epoch)

        if max_train_steps and global_step >= max_train_steps:
            break

    return last_loss, global_step


def get_model(args, spec):
    if args.use_pos_bias:
        assert spec.dimension == 2, "--use_pos_bias is for 2D tasks"
        pos_bias = dict(pos_bias_T=spec.T, pos_bias_L1=spec.grid_shape[0],
                        pos_bias_L2=spec.grid_shape[1])
    else:
        pos_bias = dict(pos_bias_T=None, pos_bias_L1=None, pos_bias_L2=None)
    return CATransformer(
        vocab_size=spec.vocab_size,
        hidden_size=args.hidden_size,
        output_size=spec.vocab_size,
        seq_len=spec.seq_len,
        num_heads_per_layer=args.num_heads_per_layer,
        use_mlp_per_layer=args.use_mlp_per_layer,
        ffn_dim_per_layer=args.ffn_dim_per_layer,
        emb_dropout=args.emb_dropout,
        dropout=args.dropout,
        pos_bias_use_row_sep=True,
        use_rope=args.use_rope,
        mlp_depth_per_layer=args.mlp_depth_per_layer,
        **pos_bias,
    )


# Defaults for every hyperparameter that configs/ may set. Flags left at None
# on the command line are filled from: model's `training:` block, then
# `training_defaults`, then this table.
_FALLBACK = dict(
    samples_per_epoch=50_000_000, seed=42, lr=1e-3, weight_decay=0.2, beta1=0.9,
    beta2=0.95, scheduler="cosine", warmup_ratio=0.004, dropout=0.0,
    global_batch_size=1024, accum_steps=1, eval_steps=5000, val_size=100_000,
    test_size=500_000, selection_metric="auto", patience=0, min_delta=1e-3,
    early_stop_threshold=None, epochs=3,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--model", choices=sorted(specs()),
                       help="Reproduce a released model: its task, architecture and "
                            "training hyperparameters from configs/.")
    which.add_argument("--task", choices=task_names(),
                       help="Train a new architecture on one of the paper's tasks.")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, default=None,
                   help="Default: '<model or task>-<arch>-lr..-gbs..-ep..'.")
    p.add_argument("--seed", type=int, default=None, help="Model init + train stream seed.")
    p.add_argument("--eval_seed", type=int, default=42,
                   help="Seed of the generated val/test splits (fixed across runs).")

    # data
    p.add_argument("--samples_per_epoch", type=int, default=None,
                   help="Freshly generated train sequences per epoch (paper: 50M).")
    p.add_argument("--val_splits", type=str, nargs="*", default=None,
                   help="Default: val_in_nofilter val_in_filter val_out (det) / val (stoch).")
    p.add_argument("--select_split", type=str, default=None,
                   help="Val split driving best-ckpt / early stop. Default val_out / val.")
    p.add_argument("--val_size", type=int, default=None, help="Sequences per val split.")
    p.add_argument("--test_size", type=int, default=None, help="Sequences in the final test split.")
    p.add_argument("--train_eval_size", type=int, default=10_000,
                   help="Train-distribution sequences evaluated under train/*.")
    p.add_argument("--sim_batch", type=int, default=256,
                   help="Trajectories simulated per numpy call inside the train stream.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)

    # architecture (filled from the registry with --model)
    p.add_argument("--hidden_size", type=int, default=None)
    p.add_argument("--num_heads_per_layer", "--heads_list", dest="num_heads_per_layer",
                   type=int, nargs="+", default=None)
    p.add_argument("--use_mlp_per_layer", "--use_mlp_list", dest="use_mlp_per_layer",
                   type=int, nargs="+", default=None)
    p.add_argument("--ffn_dim_per_layer", "--ffn_dim_list", dest="ffn_dim_per_layer",
                   type=int, nargs="+", default=None, help="Default 4*hidden_size.")
    p.add_argument("--mlp_depth_per_layer", "--mlp_depth_list", dest="mlp_depth_per_layer",
                   type=int, nargs="+", default=None)
    p.add_argument("--emb_dropout", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--use_pos_bias", action="store_true",
                   help="2D learned relative positional bias (2D tasks only).")
    p.add_argument("--use_rope", action="store_true",
                   help="Rotary position embedding instead of learned absolute PE.")

    # optimization
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--beta1", type=float, default=None)
    p.add_argument("--beta2", type=float, default=None)
    p.add_argument("--scheduler", type=str, default=None,
                   choices=["cosine", "linear", "constant", "constant_with_warmup"])
    p.add_argument("--warmup_ratio", type=float, default=None)
    p.add_argument("--global_batch_size", type=int, default=None,
                   help="Across all ranks and accumulation steps.")
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--max_train_steps", type=int, default=0,
                   help="Hard cap on optimizer steps (0 = follow --epochs).")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--temperature_init", type=float, default=1.0)
    p.add_argument("--temperature_final", type=float, default=1.0)
    p.add_argument("--temperature_anneal_frac", type=float, default=1.0)
    p.add_argument("--load_weights", type=str, default=None,
                   help="Initialize weights from a checkpoint (fresh optimizer).")

    # eval / selection / saving
    p.add_argument("--eval_steps", type=int, default=None,
                   help="Eval every N optimizer steps (0 = epoch-granular via --eval_every).")
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--selection_metric", type=str, default=None,
                   choices=["auto", "sequence_accuracy", "kl_true_model", "kl_kgram_model",
                            "kl_model_kgram", "cell_acc", "loss"])
    p.add_argument("--patience", type=int, default=None, help="0 disables early stopping.")
    p.add_argument("--min_delta", type=float, default=None)
    p.add_argument("--early_stop_threshold", type=float, default=None)
    p.add_argument("--save_every", type=int, default=1, help="Epoch ckpt every N epochs.")
    p.add_argument("--save_steps", type=int, default=0)
    p.add_argument("--keep_last", type=int, default=5)

    # logging
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--project_name", type=str, default="CA")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_notes", type=str, default="")
    return resolve_args(p.parse_args(argv))


def resolve_args(args):
    """Fill unset flags from configs/ and attach the task spec as args.spec."""
    if args.model:
        spec = get_spec(args.model)
        cfg = training_config(args.model)
        arch = dict(hidden_size=spec.d_model, num_heads_per_layer=list(spec.heads),
                    use_mlp_per_layer=list(spec.mlp))
    else:
        spec = task_spec(args.task)
        cfg = dict(_raw_registry().get("training_defaults", {}))
        arch = dict(hidden_size=256, num_heads_per_layer=[1, 1], use_mlp_per_layer=[1, 1])
    for k, v in arch.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    for k, v in _FALLBACK.items():
        if getattr(args, k, None) is None:
            setattr(args, k, cfg.get(k, v))
    if args.val_splits is None:
        args.val_splits = default_val_splits(spec)
    if args.select_split is None:
        args.select_split = default_select_split(spec)
    if args.select_split not in args.val_splits:
        raise ValueError(f"--select_split {args.select_split} is not in --val_splits {args.val_splits}")
    # Keep the task fields; record the architecture actually trained.
    args.spec = dataclasses.replace(
        spec, key=args.model or args.task,
        architecture="custom" if args.task else spec.architecture,
        d_model=args.hidden_size, heads=tuple(args.num_heads_per_layer),
        mlp=tuple(args.use_mlp_per_layer or [1] * len(args.num_heads_per_layer)))
    return args


def _fmt_lr(lr):
    if lr >= 1:
        return str(lr)
    return f"{lr:.0e}".replace("e-0", "e-")


def build_run_name(args):
    heads = "-".join(map(str, args.num_heads_per_layer))
    mlp = "-".join(map(str, args.use_mlp_per_layer or [1] * len(args.num_heads_per_layer)))
    name = (f"{args.task or args.model.rsplit('-', 1)[0]}-otf"
            f"-layers{len(args.num_heads_per_layer)}-heads{heads}-mlp{mlp}-d{args.hidden_size}"
            f"-lr{_fmt_lr(args.lr)}-wd{args.weight_decay}-gbs{args.global_batch_size}"
            f"-b2{args.beta2}-ep{args.epochs}")
    if args.seed != 42:
        name += f"-seed{args.seed}"
    return name


def build_model_config(args):
    # Same keys as the original train.py checkpoints (load_pretrained reads them).
    return {
        "hidden_size": args.hidden_size,
        "heads_list": args.num_heads_per_layer,
        "use_mlp_list": args.use_mlp_per_layer,
        "ffn_dim_list": args.ffn_dim_per_layer,
        "emb_dropout": args.emb_dropout,
        "dropout": args.dropout,
        "mixed_L": False,
        "pos_bias_T": args.spec.T if args.use_pos_bias else None,
        "pos_bias_L1": args.spec.grid_shape[0] if args.use_pos_bias else None,
        "pos_bias_L2": args.spec.grid_shape[1] if args.use_pos_bias else None,
        "pos_bias_use_row_sep": True,
        "use_rope": args.use_rope,
        "mlp_depth_list": args.mlp_depth_per_layer,
    }


def ckpt_metadata(args):
    """Task description stored in every checkpoint so it is self-describing."""
    return {"task": dataclasses.asdict(args.spec), "registry_key": args.model}


def log_metrics(metrics, prefix, log_dict):
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                log_dict[f"{prefix}/{k}/{sub_k}"] = sub_v
        else:
            log_dict[f"{prefix}/{k}"] = v


# ---- Selection-metric plumbing ---------------------------------------------

def resolve_selection(sel_arg, stochastic):
    """(metric_name, mode). 'auto': kl_true_model (min) for stochastic tasks,
    sequence_accuracy (max) for deterministic ones."""
    modes = {"sequence_accuracy": "max", "cell_acc": "max", "kl_true_model": "min",
             "kl_kgram_model": "min", "kl_model_kgram": "min", "loss": "min"}
    if sel_arg == "auto":
        return ("kl_true_model", "min") if stochastic else ("sequence_accuracy", "max")
    if sel_arg not in modes:
        raise ValueError(f"unknown selection_metric={sel_arg!r}")
    return sel_arg, modes[sel_arg]


def selection_improved(current, best, mode, min_delta):
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


def selection_worst(mode):
    return float("inf") if mode == "min" else -float("inf")


def selection_passed_threshold(best, threshold, mode):
    """threshold=None means 'no gate'."""
    if threshold is None:
        return True
    return (best <= threshold) if mode == "min" else (best >= threshold)


def main(args):
    global _WANDB
    spec = args.spec
    if not torch.cuda.is_available():
        raise SystemExit("src.train needs a CUDA GPU, but PyTorch sees none. Check that your "
                         "torch build matches your NVIDIA driver (see README).")
    ddp = is_ddp()
    if ddp:
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        world_size = get_world_size()
        print(f"[DDP] rank {get_rank()}/{world_size} on cuda:{local_rank}")
    else:
        device = torch.device("cuda")
        world_size = 1

    denom = world_size * args.accum_steps
    if args.global_batch_size % denom != 0:
        raise ValueError(
            f"--global_batch_size ({args.global_batch_size}) must be divisible by "
            f"world_size ({world_size}) * accum_steps ({args.accum_steps}) = {denom}.")
    per_gpu_batch_size = args.global_batch_size // denom
    print(f"[BATCH] global={args.global_batch_size}  per_gpu={per_gpu_batch_size} "
          f" world_size={world_size}  accum_steps={args.accum_steps}")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32

    if args.run_name is None:
        args.run_name = build_run_name(args)
    run_dir = f"{args.save_dir}/{args.run_name}"
    if is_rank0():
        os.makedirs(run_dir, exist_ok=True)
    print(f"Task: {spec.describe()}")
    print(f"Run:  {args.run_name}")

    # ---- Data: train is an endless on-the-fly stream; eval splits are generated
    # in memory on rank 0 only (only rank 0 evaluates).
    train_ds = OnTheFlyTrainDataset(spec, args.samples_per_epoch, seed=args.seed,
                                    rank=get_rank(), world_size=world_size,
                                    sim_batch=args.sim_batch)
    train_kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    if args.num_workers > 0:
        train_kw.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=per_gpu_batch_size, **train_kw)

    def make_eval_loader(ds):
        return DataLoader(ds, batch_size=per_gpu_batch_size, shuffle=False, num_workers=0)

    val_dss, val_loaders, train_eval_loader = {}, {}, None
    if is_rank0():
        t0 = time.time()
        for name in args.val_splits:
            val_dss[name] = EvalSplit(spec, name, args.val_size, seed=args.eval_seed)
            val_loaders[name] = make_eval_loader(val_dss[name])
        train_eval_ds = EvalSplit(spec, "train_eval", args.train_eval_size, seed=args.eval_seed)
        train_eval_loader = make_eval_loader(train_eval_ds)
        print(f"Generated val splits {args.val_splits} ({args.val_size} each) + "
              f"train_eval ({args.train_eval_size}) in {time.time() - t0:.1f}s "
              f"(eval_seed={args.eval_seed})")
    select_tag = args.select_split
    print(f"Train: {train_ds.total_samples()} fresh samples/epoch | "
          f"val splits: {args.val_splits} (select='{select_tag}')")

    model = get_model(args, spec).to(device)
    output_size = model.fc.out_features
    if not args.no_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[get_local_rank()])
    core_model = model.module if ddp else model

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(args.beta1, args.beta2),
                                  weight_decay=args.weight_decay)

    micro_per_epoch = (train_ds.rank_total_samples() + per_gpu_batch_size - 1) // per_gpu_batch_size
    steps_per_epoch = micro_per_epoch // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(total_steps * args.warmup_ratio)
    print(f"optimizer_steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
          f"warmup={warmup_steps}")
    scheduler = get_scheduler(args.scheduler, optimizer, warmup_steps, total_steps)

    selection_metric, selection_mode = resolve_selection(args.selection_metric, spec.stochastic)
    best_selection_value = selection_worst(selection_mode)
    print(f"Selection: metric={selection_metric} mode={selection_mode} "
          f"(min_delta={args.min_delta})", flush=True)

    patience_counter = 0
    start_epoch = 0

    # Resume from the most recent _ep{N}.pt / _step{N}.pt in this run dir.
    ckpt_files = (glob.glob(f"{run_dir}/{args.run_name}_ep*.pt")
                  + glob.glob(f"{run_dir}/{args.run_name}_step*.pt"))
    if not ckpt_files and args.load_weights:
        print(f"Loading initial weights from {args.load_weights} (fresh optimizer/scheduler)")
        sd = torch.load(args.load_weights, map_location=device, weights_only=False)["model_state_dict"]
        target_prefix = any(k.startswith("_orig_mod.") for k in core_model.state_dict())
        source_prefix = any(k.startswith("_orig_mod.") for k in sd)
        if target_prefix and not source_prefix:
            sd = {f"_orig_mod.{k}": v for k, v in sd.items()}
        elif source_prefix and not target_prefix:
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        missing, unexpected = core_model.load_state_dict(sd, strict=False)
        real_missing = [k for k in missing if not k.endswith(("position_ids", "causal_mask"))]
        if real_missing or unexpected:
            print(f"  WARNING: missing={real_missing[:5]} unexpected={unexpected[:5]}")
    if ckpt_files:
        latest = max(ckpt_files, key=os.path.getmtime)
        print(f"Resuming from {latest}")
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        core_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = ckpt["epoch"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if (ckpt.get("selection_metric"), ckpt.get("selection_mode")) != (selection_metric, selection_mode):
            raise RuntimeError(
                f"Resume mismatch: ckpt uses {ckpt.get('selection_metric')}/"
                f"{ckpt.get('selection_mode')}, this run uses {selection_metric}/{selection_mode}.")
        best_selection_value = ckpt["best_selection_value"]
        print(f"Resumed at epoch {start_epoch}, best[{selection_metric}]={best_selection_value:.4f}")

    if is_rank0() and args.wandb_mode != "disabled":
        import wandb
        _WANDB = wandb
        cfg = {k: v for k, v in vars(args).items() if k != "spec"}
        cfg["task"] = dataclasses.asdict(spec)
        wandb.init(entity=args.wandb_entity, project=args.project_name, config=cfg,
                   name=args.run_name, id=args.run_name, notes=args.wandb_notes or None,
                   resume="allow", mode=args.wandb_mode)
        wandb.define_metric("train/step")
        wandb.define_metric("train/minibatch_loss", step_metric="train/step")
        for ns in ["train", "test", "test_best"] + list(args.val_splits):
            wandb.define_metric(f"{ns}/*", step_metric="train/step")
        for k in ("learning_rate", "epoch", "train/epoch_time_sec", "train/attn_temperature"):
            wandb.define_metric(k, step_metric="train/step")

    global_step = start_epoch * steps_per_epoch

    def save_best():
        torch.save({"model_state_dict": core_model.state_dict(),
                    "model_config": build_model_config(args),
                    "selection_metric": selection_metric,
                    "selection_mode": selection_mode,
                    "best_selection_value": best_selection_value,
                    **ckpt_metadata(args)},
                   f"{run_dir}/{args.run_name}_best.pt")

    def run_val_eval(gstep, attn_temp):
        """Eval train_eval + every val split, log, update best-ckpt + patience.
        Rank 0 only; other ranks wait at the barrier."""
        nonlocal best_selection_value, patience_counter
        if ddp and not is_rank0():
            dist.barrier()
            return selection_worst(selection_mode)
        d = {"train/step": gstep}
        tm = evaluate(core_model, train_eval_loader, spec, criterion, device,
                      rules=train_eval_ds.rules, attn_temperature=attn_temp)
        tm["loss_eval"] = tm.pop("loss")
        log_metrics(tm, "train", d)
        by_split = {}
        for tag, loader in val_loaders.items():
            m = evaluate(core_model, loader, spec, criterion, device,
                         rules=val_dss[tag].rules, attn_temperature=attn_temp)
            log_metrics(m, tag, d)
            by_split[tag] = m
        sel = by_split[select_tag]
        if selection_metric not in sel:
            raise RuntimeError(f"selection_metric='{selection_metric}' not available in "
                               f"'{select_tag}'. Present: {sorted(sel)}")
        current = float(sel[selection_metric])
        if selection_improved(current, best_selection_value, selection_mode, args.min_delta):
            best_selection_value = current
            patience_counter = 0
            save_best()
        else:
            patience_counter += 1
        wlog(d)
        print(f"  [step {gstep}] [{select_tag}] cell_acc {sel['cell_acc']:.4f} "
              f"seq_acc {sel.get('sequence_accuracy', float('nan')):.4f} "
              f"kl_true_model {sel.get('kl_true_model', float('nan')):.4f} "
              f"| sel[{selection_metric}]={current:.4f} (best {best_selection_value:.4f}, "
              f"patience {patience_counter}/{args.patience or '-'})", flush=True)
        if ddp:
            dist.barrier()
        return current

    def full_ckpt(epoch, gstep=None):
        ckpt = {
            "model_state_dict": core_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "selection_metric": selection_metric,
            "selection_mode": selection_mode,
            "best_selection_value": best_selection_value,
            "model_config": build_model_config(args),
            "epoch": epoch,
            "rng_state": {"torch": torch.get_rng_state(),
                          "numpy": np.random.get_state(),
                          "python": random.getstate()},
            **ckpt_metadata(args),
        }
        if torch.cuda.is_available():
            ckpt["rng_state"]["torch_cuda"] = torch.cuda.get_rng_state_all()
        if gstep is not None:
            ckpt["global_step"] = gstep
        return ckpt

    def run_step_save(gstep, cur_epoch):
        if ddp and not is_rank0():
            dist.barrier()
            return
        path = f"{run_dir}/{args.run_name}_step{gstep}.pt"
        torch.save(full_ckpt(cur_epoch, gstep), path)
        print(f"  [step {gstep}] saved step ckpt -> {path}")
        prune_ckpts(args.save_dir, args.run_name, args.keep_last)
        if ddp:
            dist.barrier()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        train_ds.set_epoch(epoch)   # fresh samples every epoch
        attn_temp = get_temperature(epoch, args.epochs, args.temperature_init,
                                    args.temperature_final, args.temperature_anneal_frac)
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            output_size=output_size, scheduler=scheduler,
            attn_temperature=attn_temp, global_step=global_step,
            eval_steps=args.eval_steps,
            eval_callback=run_val_eval if args.eval_steps > 0 else None,
            accum_steps=args.accum_steps,
            save_steps=args.save_steps,
            save_callback=run_step_save if args.save_steps > 0 else None,
            max_train_steps=args.max_train_steps,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        if ddp:
            dist.barrier()
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1}/{args.epochs} train time: {epoch_time:.1f}s "
              f"| attn_temp: {attn_temp:.4f} | loss: {train_loss:.4f} "
              f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        stop_steps = bool(args.max_train_steps and global_step >= args.max_train_steps)
        early_stop = False
        if args.eval_steps > 0 or (epoch + 1) % args.eval_every == 0 or stop_steps:
            run_val_eval(global_step, attn_temp)
            early_stop = (args.patience > 0 and patience_counter >= args.patience
                          and selection_passed_threshold(best_selection_value,
                                                         args.early_stop_threshold,
                                                         selection_mode))
        wlog({"train/step": global_step, "train/epoch_time_sec": epoch_time,
              "train/attn_temperature": attn_temp,
              "learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch})

        if (epoch + 1) % args.save_every == 0 and is_rank0():
            torch.save(full_ckpt(epoch + 1), f"{run_dir}/{args.run_name}_ep{epoch + 1}.pt")
            prune_ckpts(args.save_dir, args.run_name, args.keep_last)

        if ddp:
            flag = torch.tensor([1 if early_stop else 0], device=device)
            dist.broadcast(flag, src=0)
            early_stop = bool(flag.item())
            dist.barrier()
        if stop_steps:
            print(f"Reached --max_train_steps={args.max_train_steps}.")
            break
        if early_stop:
            print(f"Early stopping: {select_tag} {selection_metric} stalled "
                  f"({args.patience} cycles).")
            break

    # ===== Final test evaluation (rank 0 only) =====
    if is_rank0():
        print("\n" + "=" * 50)
        t0 = time.time()
        test_ds = EvalSplit(spec, "test", args.test_size, seed=args.eval_seed)
        test_loader = make_eval_loader(test_ds)
        print(f"Generated test split ({args.test_size}) in {time.time() - t0:.1f}s. Evaluating...",
              flush=True)
        test_metrics = evaluate(core_model, test_loader, spec, criterion, device,
                                rules=test_ds.rules, attn_temperature=args.temperature_final)
        test_log = {}
        log_metrics(test_metrics, "test", test_log)
        wlog(test_log)
        print("Final model test:", {k: round(v, 6) for k, v in test_metrics.items()})

        # Save the final weights before the best ckpt is loaded for test_best.
        torch.save({"model_state_dict": core_model.state_dict(),
                    "model_config": build_model_config(args), **ckpt_metadata(args)},
                   f"{run_dir}/{args.run_name}_final.pt")
        best_path = f"{run_dir}/{args.run_name}_best.pt"
        if os.path.exists(best_path):
            best = torch.load(best_path, map_location=device, weights_only=False)
            core_model.load_state_dict(best["model_state_dict"])
            best_metrics = evaluate(core_model, test_loader, spec, criterion, device,
                                    rules=test_ds.rules, attn_temperature=args.temperature_final)
            best_log = {}
            log_metrics(best_metrics, "test_best", best_log)
            wlog(best_log)
            print("Best model test: ", {k: round(v, 6) for k, v in best_metrics.items()})

        print(f"\nBest val {selection_metric}: {best_selection_value:.4f}")
        if _WANDB is not None:
            _WANDB.finish()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
