"""Teacher-forced + autoregressive evaluation of one checkpoint (paper metrics).

Ported from the repo-root run_ckpt_eval.py: same evaluators (src/evaluators.py)
called with the same arguments, same metrics and W&B layout. What changed:
  * the model comes from the registry (--model) or any checkpoint (--ckpt);
  * the test split is generated on the fly by default (same rule pools,
    coverage filter and masks as the original; statistically, not bit-for-bit,
    the same samples);
  * W&B is optional (--wandb_mode, default disabled) and results go to --out.

Examples:
    # Released model, 100k freshly generated test trajectories (the default)
    python -m src.evaluate --model 1d-det-v2k3-shared --out results.json
    # A checkpoint you trained with src.train
    python -m src.evaluate --ckpt ckpts/<run>/<run>_best.pt
    # Quick check
    python -m src.evaluate --model 1d-stoch-v3k3-dedicated --n 2000
"""

import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from .data.streaming import EvalSplit
from .evaluators import (
    evaluate_standard_accuracy,
    evaluate_autoregressive_cell_accuracy,
    evaluate_stochastic_teacher_forced,
    evaluate_stochastic_autoregressive,
)
from .pretrained import load_checkpoint, load_pretrained
from .registry import specs


def _load(args, device):
    if args.ckpt and not args.model:
        return load_checkpoint(args.ckpt, device)
    return load_pretrained(args.model, device=device, checkpoint_path=args.ckpt)


def _build_test_set(args, spec):
    """Return (dataset, rules aligned with it, description)."""
    n = args.n or 100_000
    t0 = time.time()
    ds = EvalSplit(spec, args.split, n, seed=args.eval_seed)
    return ds, (ds.rules if spec.stochastic else None), \
        f"generated '{args.split}' split, {n} samples, eval_seed={args.eval_seed} " \
        f"({time.time() - t0:.1f}s)"


def run(args) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[eval] WARNING: no CUDA GPU visible to PyTorch, running on CPU (check that "
              "your torch build matches your NVIDIA driver).")
        args.device = "cpu"
    device = torch.device(args.device)
    bundle = _load(args, device)
    model, spec = bundle.model, bundle.spec
    print(f"[eval] Model: {spec.describe()}  ({bundle.checkpoint_path})")
    mode = f"{spec.dimension}d"
    T, M, V, K = spec.T, spec.M, spec.V, spec.K
    if args.split is None:
        args.split = "test"
    ds, rules_np, desc = _build_test_set(args, spec)
    print(f"[eval] Test set: {desc}")
    n_examples = len(ds)

    def loader():
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=device.type == "cuda")

    run_wandb = None
    if args.wandb_mode != "disabled":
        import wandb
        run_wandb = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                               name=args.wandb_run_name or f"eval-{spec.key}",
                               group=args.wandb_group, mode=args.wandb_mode,
                               notes=args.wandb_notes, job_type="eval", reinit=True,
                               config={"model": spec.key, "ckpt": bundle.checkpoint_path,
                                       "test_set": desc, "n_test_examples": n_examples,
                                       "num_rollouts": args.num_rollouts, "seed": args.seed})

    log = {}
    geom = (dict(L=spec.grid_shape[0]) if spec.dimension == 1
            else dict(L1=spec.grid_shape[0], L2=spec.grid_shape[1]))
    tokens = (dict(sep_token=spec.time_sep) if spec.dimension == 1 else
              dict(use_row_sep=True, row_sep_token=spec.row_sep, time_sep_token=spec.time_sep))

    if not spec.stochastic:
        print("\n[eval] [DET] Running teacher-forced evaluation ...")
        t0 = time.time()
        std = evaluate_standard_accuracy(model, loader(), device, v_states=V, use_ddp=False)
        print(f"[eval] done in {time.time() - t0:.1f}s. tf/cell_acc = {std['cell_acc']:.6f}  "
              f"tf/seq_acc = {std['seq_acc']:.6f}")
        log["tf/cell_acc"] = std["cell_acc"]
        log["tf/seq_acc"] = std["seq_acc"]

        print("\n[eval] [DET] Running autoregressive rollout evaluation ...")
        t0 = time.time()
        ar_kwargs = dict(mode=mode, num_context_states=M, first_predicted_timestamp=M + 1,
                         last_predicted_timestamp=T, use_ddp=False, **geom, **tokens)
        if spec.dimension == 1:
            ar_kwargs["V"] = V
        ar = evaluate_autoregressive_cell_accuracy(model, loader(), device, **ar_kwargs)
        overall = ar["total_correct_overall"] / max(ar["total_cells_overall"], 1)
        print(f"[eval] done in {time.time() - t0:.1f}s. ar/overall/cell_acc = {overall:.6f}")
        log["ar/overall/cell_acc"] = overall
        by_t = {int(t): float(a) for t, a in sorted(ar["cell_acc_by_timestamp"].items())}
        for t, a in by_t.items():
            print(f"       AR t={t:2d}: cell_acc={a:.6f}")
        log["ar/cell_acc_by_timestamp"] = by_t
    else:
        print("\n[eval] [STOCH] Running teacher-forced stochastic evaluation ...")
        t0 = time.time()
        tf = evaluate_stochastic_teacher_forced(model, loader(), device, mode=mode, T=T, M=M,
                                                V=V, K=K, rules_np=rules_np, use_ddp=False, **geom)
        print(f"[eval] done in {time.time() - t0:.1f}s (n_scored={tf['n_scored_cells']}).")
        for k in ("cell_acc", "kgram_cell_acc", "kl_kgram_model", "kl_true_model", "kl_true_kgram"):
            print(f"       tf/{k:15s} = {tf[k]:.6f}")
            log[f"tf/{k}"] = tf[k]

        print("\n[eval] [STOCH] Running two-stream sampled AR evaluation ...")
        t0 = time.time()
        ar = evaluate_stochastic_autoregressive(model, loader(), device, mode=mode, T=T, M=M,
                                                V=V, K=K, rules_np=rules_np,
                                                num_rollouts=args.num_rollouts, seed=args.seed,
                                                use_ddp=False, **geom, **tokens)
        print(f"[eval] done in {time.time() - t0:.1f}s.")
        for k in ("model_cell_acc", "kgram_cell_acc", "kl_kgram_model", "kl_true_model",
                  "kl_true_kgram"):
            print(f"       ar/overall/{k:15s} = {ar[k]:.6f}")
            log[f"ar/overall/{k}"] = ar[k]
        log["ar/by_timestamp"] = {int(t): {k: float(v) for k, v in ar["by_timestamp"][t].items()
                                           if isinstance(v, (int, float))}
                                  for t in ar["evaluated_timestamps"]}

    result = {"model": spec.key, "checkpoint": bundle.checkpoint_path, "test_set": desc,
              "n_test_examples": n_examples, "metrics": log}
    if run_wandb is not None:
        flat = {k: v for k, v in log.items() if not isinstance(v, dict)}
        run_wandb.log(flat)
        run_wandb.summary.update(flat)
        run_wandb.finish()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[eval] wrote {args.out}")
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=sorted(specs()), default=None,
                   help="Registry key. Alone: download the released checkpoint.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Local checkpoint. With --model it is checked against the "
                        "registry; alone it must come from src.train.")
    p.add_argument("--split", type=str, default=None,
                   help="Split name (default 'test'; e.g. val_out / val also work).")
    p.add_argument("--n", type=int, default=0,
                   help="Number of test trajectories (default: 100000).")
    p.add_argument("--eval_seed", type=int, default=42,
                   help="Seed of the generated split (the same default as src.train).")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_rollouts", type=int, default=1,
                   help="Stochastic AR: independent rollouts per test example.")
    p.add_argument("--seed", type=int, default=42, help="Stochastic AR sampling seed.")
    p.add_argument("--out", type=str, default=None, help="Write results as JSON.")
    p.add_argument("--wandb_mode", type=str, default="disabled",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default="CA-eval")
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_notes", type=str, default=None)
    args = p.parse_args(argv)
    if not args.model and not args.ckpt:
        p.error("give --model and/or --ckpt")
    return args


if __name__ == "__main__":
    run(parse_args())
