"""Evaluation metrics for ECA transformer models.

Usage:
    python eval.py --checkpoint_path /path/to/best.pt --test_path /path/to/test.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_generate import ECADataset
from models import VSimpleTransformer
from utils import unpack_batch

# Try to import mixed-L dataset
try:
    from data_generate_mixed_L import ECADatasetMixedL
except ImportError:
    ECADatasetMixedL = None


def evaluate(model: VSimpleTransformer, loader: DataLoader, criterion: nn.Module,
             device: torch.device, cell_width: int | None = None,
             rules: np.ndarray | None = None,
             Ls: np.ndarray | None = None) -> dict:
    """Return dict of metrics.

    Always: loss, cell_acc, zero_error_rate
    If cell_width set (fixed-L): per_timestep_acc, per_rule_acc
    If Ls set (mixed-L): per_timestep_acc, per_rule_acc (computed per-sample)

    Compatible with both fixed-L and mixed-L dataloaders.
    """
    model.eval()
    total_correct = 0
    total_mask = 0
    total_loss = 0.0

    zero_error_samples = 0
    total_samples = 0
    row_correct: dict[int, int] = {}
    row_total: dict[int, int] = {}
    rule_correct: dict[int, int] = {}
    rule_total: dict[int, int] = {}

    output_size = model.fc.out_features
    # Fixed-L mode: global stride (only used when Ls is None)
    stride = cell_width + 1 if cell_width else None
    sample_offset = 0

    with torch.no_grad():
        for batch in loader:
            X, y, mask, lengths = unpack_batch(batch)
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            if lengths is not None:
                lengths = lengths.to(device)

            output = model(X, lengths=lengths)
            B, T = X.shape[0], X.shape[1]

            loss_raw = criterion(output.view(-1, output_size), y.view(-1))
            loss_raw = loss_raw.view(B, -1)

            preds = output.argmax(dim=-1)
            eval_mask = mask > 0

            total_loss += (loss_raw * eval_mask).sum().item()

            correct = (preds == y) & eval_mask
            total_correct += correct.sum().item()
            total_mask += eval_mask.sum().item()

            # --- Zero-error rate (always, works for both fixed/mixed-L) ---
            is_error = (preds != y) & eval_mask
            has_error = is_error.any(dim=1)
            has_eval = eval_mask.any(dim=1)

            zero_error_samples += ((~has_error) & has_eval).sum().item()
            total_samples += has_eval.sum().item()

            # --- Per-rule accuracy (always, if rules provided) ---
            if rules is not None:
                has_error_cpu = has_error.cpu().numpy()
                has_eval_cpu = has_eval.cpu().numpy()
                for b in range(B):
                    gidx = sample_offset + b
                    if gidx >= len(rules):
                        continue
                    if not has_eval_cpu[b]:
                        continue
                    r = int(rules[gidx])
                    if r not in rule_correct:
                        rule_correct[r] = 0
                        rule_total[r] = 0
                    rule_total[r] += 1
                    if not has_error_cpu[b]:
                        rule_correct[r] += 1

            # --- Per-timestep accuracy ---
            if Ls is not None:
                # Mixed-L mode: per-sample row boundaries
                is_error_cpu = is_error.cpu().numpy()
                eval_mask_cpu = eval_mask.cpu().numpy()
                lengths_cpu = lengths.cpu().numpy() if lengths is not None else None
                for b in range(B):
                    gidx = sample_offset + b
                    if gidx >= len(Ls):
                        continue
                    cw = int(Ls[gidx]) - 1       # cell_width = L - 1
                    s = cw + 1                     # stride = cell_width + 1 (cells + SEP)
                    actual_len = int(lengths_cpu[b]) if lengths_cpu is not None else T
                    num_rows = (actual_len + s - 1) // s
                    for t in range(num_rows):
                        rs = t * s
                        re = min(rs + cw, actual_len)
                        if re <= rs:
                            continue
                        row_eval = eval_mask_cpu[b, rs:re]
                        if not row_eval.any():
                            continue
                        row_err = is_error_cpu[b, rs:re].any()
                        if t not in row_correct:
                            row_correct[t] = 0
                            row_total[t] = 0
                        row_total[t] += 1
                        if not row_err:
                            row_correct[t] += 1

            elif stride is not None:
                # Fixed-L mode: vectorized (original path)
                num_rows = (T + stride - 1) // stride
                for t in range(num_rows):
                    rs = t * stride
                    re = min(rs + cell_width, T)
                    if re <= rs:
                        continue
                    row_eval = eval_mask[:, rs:re]
                    row_has = row_eval.any(dim=1)
                    row_err = is_error[:, rs:re].any(dim=1)
                    row_ok = row_has & ~row_err

                    if t not in row_correct:
                        row_correct[t] = 0
                        row_total[t] = 0
                    row_correct[t] += row_ok.sum().item()
                    row_total[t] += row_has.sum().item()

            sample_offset += B

    metrics: dict = {
        "loss": total_loss / total_mask if total_mask > 0 else 0,
        "cell_acc": total_correct / total_mask if total_mask > 0 else 0,
    }

    if total_samples > 0:
        metrics["zero_error_rate"] = zero_error_samples / total_samples

    if row_correct:
        metrics["per_timestep_acc"] = {
            t: row_correct[t] / row_total[t] if row_total[t] > 0 else 0
            for t in sorted(row_correct.keys())
        }

    if rule_correct:
        metrics["per_rule_acc"] = {
            r: rule_correct[r] / rule_total[r] if rule_total[r] > 0 else 0
            for r in sorted(rule_correct.keys())
        }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--cell_width", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_L", action="store_true",
                        help="Use mixed-L dataset loader (expects 4-tuple batches)")
    parser.add_argument("--ffn_dim_list", type=int, nargs='+', default=None,
                        help="Override per-layer MLP hidden dim (for old ckpts missing this in cfg).")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint_path, map_location=device)
    cfg = ckpt["model_config"]

    # Determine vocab size from checkpoint config
    is_mixed = cfg.get("mixed_L", False) or args.mixed_L
    vocab_size = 4 if is_mixed else 3

    # Load dataset
    if is_mixed:
        if ECADatasetMixedL is None:
            raise ImportError("data_generate_mixed_L.py not found.")
        test_ds = ECADatasetMixedL(args.test_path)
    else:
        test_ds = ECADataset(args.test_path)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    sample = test_ds[0]
    seq_len = sample[0].shape[0]

    model = VSimpleTransformer(
        vocab_size=vocab_size,
        hidden_size=cfg["hidden_size"],
        output_size=vocab_size,
        seq_len=seq_len,
        heads_list=cfg["heads_list"],
        use_mlp_list=cfg.get("use_mlp_list", None),
        ffn_dim_list=args.ffn_dim_list if args.ffn_dim_list is not None else cfg.get("ffn_dim_list", None),
        emb_dropout=cfg.get("emb_dropout", 0.0),
        dropout=cfg.get("dropout", 0.0),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    rules = test_ds.rules
    Ls = getattr(test_ds, 'Ls', None)
    metrics = evaluate(model, test_loader, criterion, device,
                       cell_width=args.cell_width, rules=rules, Ls=Ls)

    print("===== Evaluation Result =====")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Test data:  {args.test_path}")
    print(f"Loss:       {metrics['loss']:.6f}")
    print(f"Cell acc:   {metrics['cell_acc']:.6f}")

    if "zero_error_rate" in metrics:
        print(f"Zero-error: {metrics['zero_error_rate']:.6f}")

    if "per_rule_acc" in metrics:
        print("\nPer-rule zero-error rate:")
        for r, acc in sorted(metrics["per_rule_acc"].items()):
            print(f"  Rule {r:>3d}: {acc:.4f}")

    if "per_timestep_acc" in metrics:
        print("\nPer-row zero-error rate:")
        for t, acc in sorted(metrics["per_timestep_acc"].items()):
            print(f"  Row {t:>2d}: {acc:.4f}")


if __name__ == "__main__":
    main()