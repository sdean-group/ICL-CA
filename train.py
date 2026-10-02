"""Training loop for ECA transformer models.

Usage:
    python train.py --train_path /path/to/train.npz --test_path /path/to/test.npz
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from transformers import get_scheduler

from data_generate import ECADataset
from eval import evaluate
from models import VSimpleTransformer
from utils import log_sample_analysis, set_seed, unpack_batch

# Try to import mixed-L dataset; fall back gracefully for fixed-L usage
try:
    from data_generate_mixed_L import ECADatasetMixedL
except ImportError:
    ECADatasetMixedL = None


def train_epoch(model: VSimpleTransformer, train_loader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                device: torch.device, epoch: int, output_size: int = 3,
                scheduler: object | None = None,
                spike_threshold: float = 3.0) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch_idx, batch in enumerate(train_loader):
        X, y, mask, lengths = unpack_batch(batch)
        X, y, mask = X.to(device), y.to(device), mask.to(device)
        if lengths is not None:
            lengths = lengths.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(X, lengths=lengths)
        B = output.shape[0]
        loss_raw = criterion(output.view(-1, output_size), y.view(-1))
        loss_raw = loss_raw.view(B, -1)

        loss = (loss_raw * mask).sum() / mask.sum()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.detach()
        total_count += 1

    return (total_loss / total_count).item()


def get_model(args: argparse.Namespace, seq_len: int) -> VSimpleTransformer:
    vocab_size = args.vocab_size
    output_size = vocab_size

    if args.model_type == "transformer":
        return VSimpleTransformer(
            vocab_size=vocab_size,
            hidden_size=args.hidden_size,
            output_size=output_size,
            seq_len=seq_len,
            heads_list=args.heads_list,
            use_mlp_list=args.use_mlp_list,
            ffn_dim_list=args.ffn_dim_list,
            emb_dropout=args.emb_dropout,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model_type: {args.model_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--model_type", type=str, default="transformer",
        choices=["transformer", "cnn_transformer", "linear", "mlp", "clstransformer"])
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, default=None,
                        help="Validation set path. If not provided, uses test_path for val.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Test set path. Only evaluated at the end of training.")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--heads_list", type=int, nargs='+', default=[2, 4])
    parser.add_argument("--emb_dropout", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.004)
    parser.add_argument("--project_name", type=str, default="xtx_v0")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=16)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--cell_width", type=int, default=None)
    parser.add_argument("--analyze_every", type=int, default=32)
    parser.add_argument("--use_mlp_list", type=int, nargs='+', default=None)
    parser.add_argument("--ffn_dim_list", type=int, nargs='+', default=None,
                        help="Per-layer MLP hidden dim. Defaults to 4*hidden_size.")
    parser.add_argument("--mixed_L", action="store_true",
                        help="Enable mixed grid-width training (vocab=4, expects 4-tuple batches)")
    parser.add_argument("--vocab_size", type=int, default=3,
                    help="3 for 1D ECA or 2D no_row_sep, 4 for 2D with row_sep")
    parser.add_argument("--eval_train", action="store_true",
                        help="Also evaluate on train set (slow for large datasets)")
    return parser.parse_args()


def build_model_config(args: argparse.Namespace) -> dict:
    return {
        "model_type": args.model_type,
        "hidden_size": args.hidden_size,
        "heads_list": args.heads_list,
        "use_mlp_list": args.use_mlp_list,
        "ffn_dim_list": args.ffn_dim_list,
        "emb_dropout": args.emb_dropout,
        "dropout": args.dropout,
        "mixed_L": args.mixed_L,
    }


def log_metrics(metrics: dict, prefix: str, log_dict: dict) -> None:
    """Log metrics to wandb dict. Skip per_rule_acc to avoid chart spam."""
    for k, v in metrics.items():
        if k == "per_rule_acc":
            continue  # skip per-rule breakdown
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                log_dict[f"{prefix}/{k}/{sub_k}"] = sub_v
        else:
            log_dict[f"{prefix}/{k}"] = v


def load_dataset(path: str, mixed_L: bool = False) -> ECADataset:
    if mixed_L:
        if ECADatasetMixedL is None:
            raise ImportError("data_generate_mixed_L.py not found.")
        return ECADatasetMixedL(path)
    else:
        return ECADataset(path)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    set_seed(args.seed)

    # Load datasets
    train_ds = load_dataset(args.train_path, args.mixed_L)

    # Val set: use --val_path if provided, otherwise fall back to --test_path
    val_path = args.val_path if args.val_path else args.test_path
    val_ds = load_dataset(val_path, args.mixed_L)

    # Test set: only evaluated at the end
    test_ds = load_dataset(args.test_path, args.mixed_L)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f"Train: {len(train_ds)} samples")
    print(f"Val:   {len(val_ds)} samples (from {val_path})")
    print(f"Test:  {len(test_ds)} samples (from {args.test_path})")

    val_rules = val_ds.rules
    test_rules = test_ds.rules
    train_Ls = getattr(train_ds, 'Ls', None)
    val_Ls = getattr(val_ds, 'Ls', None)
    test_Ls = getattr(test_ds, 'Ls', None)

    first_sample = train_ds[0]
    seq_len = first_sample[0].shape[0]
    print(f"Sequence length: {seq_len}")

    model = get_model(args, seq_len)
    model.to(device)
    model = torch.compile(model)

    output_size = model.fc.out_features
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_acc = -float("inf")
    start_epoch = 0

    # Resume from checkpoint if available
    last_ckpt_path = f"{args.save_dir}/{args.run_name}/{args.run_name}_last.pt"
    if os.path.exists(last_ckpt_path):
        print(f"Resuming from {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_val_acc = ckpt.get("best_val_acc", -float("inf"))
        print(f"Resumed at epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    wandb.init(
        project=args.project_name,
        config=vars(args),
        name=args.run_name or f"{args.model_type}_seed{args.seed}",
        id=args.run_name,
        resume="allow",
    )

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            output_size=output_size,
            scheduler=scheduler if args.scheduler in ["cosine", "onecycle"] else None,
        )

        log_dict = {
            "train/loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        }

        if (epoch + 1) % args.eval_every == 0:
            # Evaluate on val set (no per-rule breakdown, rules=None)
            val_metrics = evaluate(model, val_loader, criterion, device,
                                   cell_width=args.cell_width, rules=None, Ls=val_Ls)
            log_metrics(val_metrics, "val", log_dict)

            # Optionally evaluate on train set (slow for large datasets)
            if args.eval_train:
                train_metrics = evaluate(model, train_loader, criterion, device,
                                         cell_width=args.cell_width, rules=None, Ls=train_Ls)
                log_metrics(train_metrics, "train_eval", log_dict)

            if args.scheduler == "reduce_on_plateau":
                scheduler.step(val_metrics["cell_acc"])

            val_acc = val_metrics["cell_acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_config": build_model_config(args),
                }
                torch.save(checkpoint, f"{args.save_dir}/{args.run_name}/{args.run_name}_best.pt")

            zer = val_metrics.get("zero_error_rate", -1)
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} "
                  f"| Val cell_acc: {val_acc:.4f} zer: {zer:.4f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} "
                  f"| LR: {optimizer.param_groups[0]['lr']:.6f}")

        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "model_config": build_model_config(args),
                "epoch": epoch + 1,
            }
            ckpt_name = f"{args.run_name}_ep{epoch+1}.pt"
            torch.save(ckpt, f"{args.save_dir}/{args.run_name}/{ckpt_name}")

        if args.cell_width is not None and (epoch + 1) % args.analyze_every == 0:
            log_sample_analysis(model, val_loader, device, args.cell_width, epoch,
                                rules=val_rules)

        wandb.log(log_dict)

    # ===== Final test evaluation =====
    print("\n" + "=" * 50)
    print("Final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device,
                            cell_width=args.cell_width, rules=test_rules, Ls=test_Ls)

    test_log = {}
    log_metrics(test_metrics, "test", test_log)
    wandb.log(test_log)

    test_acc = test_metrics["cell_acc"]
    test_zer = test_metrics.get("zero_error_rate", -1)
    print(f"Test cell_acc: {test_acc:.4f} | Test zer: {test_zer:.4f}")

    # Also evaluate best model on test set
    best_ckpt_path = f"{args.save_dir}/{args.run_name}/{args.run_name}_best.pt"
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])
        best_test_metrics = evaluate(model, test_loader, criterion, device,
                                     cell_width=args.cell_width, rules=test_rules, Ls=test_Ls)
        best_test_log = {}
        log_metrics(best_test_metrics, "test_best", best_test_log)
        wandb.log(best_test_log)

        best_test_acc = best_test_metrics["cell_acc"]
        best_test_zer = best_test_metrics.get("zero_error_rate", -1)
        print(f"Best model test cell_acc: {best_test_acc:.4f} | Test zer: {best_test_zer:.4f}")

    # Save final model
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": build_model_config(args),
    }
    torch.save(checkpoint, f"{args.save_dir}/{args.run_name}/{args.run_name}_final.pt")
    print(f"\nBest Val Cell Acc: {best_val_acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main(parse_args())