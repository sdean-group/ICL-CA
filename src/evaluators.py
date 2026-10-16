"""Evaluation functions for CA transformers (copied from the repo-root
evaluators.py, which produced the paper's TF/AR numbers).

The four loader-driven evaluators below are unchanged. The legacy
`evaluate()` at the end of the original file is replaced by the training-time
evaluator (same metric definitions; see the section at the bottom).

Reusable evaluation functions for CA transformers.

Four public entry points, all loader-driven, DDP-safe, integer-counter based,
and safe to call from either a standalone script or a training loop:

    evaluate_standard_accuracy(model, dataloader, device, ...)                   [deterministic]
        Teacher-forced cell_acc and seq_acc. Semantics match `eval.py::evaluate`.

    evaluate_autoregressive_cell_accuracy(model, dataloader, device, mode, ...)  [deterministic]
        Greedy-argmax AR rollout. Per-timestep + overall cell accuracy.

    evaluate_stochastic_teacher_forced(model, dataloader, device, mode, ...)     [stochastic]
        Teacher-forced KL & cell-accuracy against p_true (per-sample stochastic
        rule) and the causal add-one k-gram baseline. Convention is
        `KL(reference || predictor)`:
            KL(q_kgram || q_model)   -> kl_kgram_model
            KL(p_true  || q_model)   -> kl_true_model
            KL(p_true  || q_kgram)   -> kl_true_kgram
        plus model-argmax and k-gram-argmax cell accuracies vs the realized GT.

    evaluate_stochastic_autoregressive(model, dataloader, device, mode, ...)     [stochastic]
        Two independent sampled rollouts:
          Stream A (model): samples from q_model each cell; reports
            KL(q_kgram_A||q_model_A), KL(p_true_A||q_model_A), model_cell_acc.
          Stream B (kgram): samples from q_kgram each cell; reports
            KL(p_true_B||q_kgram_B) and kgram_cell_acc.
        Both streams share only the first M ground-truth states, then diverge —
        independent prefixes, independent counters. AR accuracy compares each
        stream's sampled token against the saved stochastic test trajectory.

All four:
  - Use `torch.inference_mode()`.
  - Preserve the model's train/eval state (temporarily eval, restore afterwards).
  - Accumulate integer / float64 counters and average once at the end
    (partial final batch OK; no batch-of-batch averaging).
  - All-reduce raw counters across ranks when torch.distributed is initialised.
  - Never touch W&B.

Stochastic conventions (per spec):
  - Use only the first V cell logits. Drop the SEP logit and softmax over the
    remaining V values (equivalent to conditioning on 'a cell comes next').
  - The causal add-one k-gram distribution for a target position j is
      q_kgram(v | c_j) = (N^{<j}[c_j, v] + 1) / (sum_u N^{<j}[c_j, u] + V).
    The current outcome must NOT be included when the current cell's metrics
    are computed. Counts are updated AFTER metric accumulation.
  - Teacher-forced k-gram counts use ground-truth outcomes and walk every
    valid causal transition (including t < M) so the baseline sees the same
    context the model sees. Metric accumulation itself only sums positions
    with mask > 0 (i.e. t >= M in the test split).
  - AR sampling is temperature 1 (no top-k/top-p). Structural tokens
    (1D SEP, 2D ROW_SEP / TIME_SEP) are injected at known positions and never
    sampled. Stream A's model counter and stream B's kgram counter each
    process only their own sampled outcomes after t >= M; both are pre-seeded
    with the same GT transitions from t < M context.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# =============================================================================
# Distributed helpers
# =============================================================================

def _autodetect_ddp() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


# =============================================================================
# Standard teacher-forced accuracy
# =============================================================================

def evaluate_standard_accuracy(model, dataloader, device, *,
                                v_states: Optional[int] = None,
                                use_ddp: Optional[bool] = None) -> dict:
    """Teacher-forced cell / sequence accuracy over the full dataloader.

    Cell accuracy: fraction of scored CELL-TARGET positions (mask > 0 AND
    y < v_states) whose argmax prediction matches the ground-truth target.
    Match to `eval.py::evaluate` cell_acc post-2026-09-16 SEP-mask fix.

    If v_states is not given, we infer it from `logits.shape[-1]` (the model's
    output vocab size) assuming the standard tokenization: 1D = V+1
    (state{0..V-1}, TIME_SEP=V), 2D VN = V+2 (states, TIME_SEP, ROW_SEP).
    Auto-detect via `max(y[mask>0])`: if max <= 1 -> V_states = 2 (1D V=2 or
    2D V=2 that only has states in y); we conservatively use output_size - 1
    (1D layout) unless caller passes v_states explicitly.

    Sequence accuracy: fraction of samples that have at least one scored
    position and zero prediction errors across all their scored positions.
    Matches `eval.py::evaluate` sequence_accuracy.

    DDP: raw integer counters are all-reduced across ranks before averaging.
    Preserves the model's original train/eval mode.

    Returns:
        {
            "cell_acc": float,
            "seq_acc": float,
            "n_correct_cells": int,   # after all-reduce
            "n_total_cells": int,
            "n_correct_seqs": int,
            "n_total_seqs": int,
        }
    """
    if use_ddp is None:
        use_ddp = _autodetect_ddp()

    was_training = model.training
    model.eval()

    n_correct_cells = torch.zeros((), dtype=torch.float64, device=device)
    n_total_cells = torch.zeros((), dtype=torch.float64, device=device)
    n_correct_seqs = torch.zeros((), dtype=torch.float64, device=device)
    n_total_seqs = torch.zeros((), dtype=torch.float64, device=device)

    try:
        with torch.inference_mode():
            for batch in dataloader:
                X, y, mask = batch
                X = X.to(device, non_blocking=True).long()
                y = y.to(device, non_blocking=True).long()
                mask = mask.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                    logits = model(X)
                logits = logits.float()

                preds = logits.argmax(dim=-1)
                eval_mask = mask > 0
                # Restrict to CELL-target positions (drop SEP-target rows the
                # data mask accidentally flags as scored) so SEP predictions
                # don't inflate cell_acc / seq_acc. See evaluate() for the
                # detailed diagnosis (2D VN was inflated by ~4%).
                if v_states is None:
                    # Conservative default: infer from output vocab. Callers
                    # should pass v_states explicitly when known (see
                    # run_ckpt_eval.py).
                    v_states_local = int(logits.shape[-1]) - 1
                else:
                    v_states_local = int(v_states)
                cell_mask = eval_mask & (y < v_states_local)
                correct = (preds == y) & cell_mask

                n_correct_cells += correct.sum().double()
                n_total_cells += cell_mask.sum().double()

                is_error = (preds != y) & cell_mask
                has_error = is_error.any(dim=1)
                has_eval = cell_mask.any(dim=1)
                n_correct_seqs += ((~has_error) & has_eval).sum().double()
                n_total_seqs += has_eval.sum().double()

        if use_ddp:
            _all_reduce_sum(n_correct_cells)
            _all_reduce_sum(n_total_cells)
            _all_reduce_sum(n_correct_seqs)
            _all_reduce_sum(n_total_seqs)
    finally:
        if was_training:
            model.train()

    nc = int(n_correct_cells.item())
    nt = int(n_total_cells.item())
    ns_c = int(n_correct_seqs.item())
    ns_t = int(n_total_seqs.item())

    return {
        "cell_acc": (nc / nt) if nt > 0 else 0.0,
        "seq_acc": (ns_c / ns_t) if ns_t > 0 else 0.0,
        "n_correct_cells": nc,
        "n_total_cells": nt,
        "n_correct_seqs": ns_c,
        "n_total_seqs": ns_t,
    }


# =============================================================================
# 1D AR rollout helpers
# =============================================================================

def _reconstruct_full_1d(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """(B, seq_len_inp) + (B, seq_len_inp) -> (B, seq_len_inp + 1) full seq."""
    return torch.cat([X[:, :1], y], dim=1)


def _extract_cell_grid_1d(full: torch.Tensor, T: int, L: int) -> torch.Tensor:
    """(B, T*(L+1)-1) -> (B, T, L) cell grid.

    Layout: row t at positions t*(L+1)..t*(L+1)+L-1; SEP at t*(L+1)+L for
    t < T-1; no trailing SEP.
    """
    B = full.shape[0]
    expected = T * (L + 1) - 1
    assert full.shape[1] == expected, (
        f"full length {full.shape[1]} != T*(L+1)-1 = {expected}")
    grid = torch.empty((B, T, L), dtype=full.dtype, device=full.device)
    for t in range(T):
        base = t * (L + 1)
        grid[:, t, :] = full[:, base:base + L]
    return grid


def _ar_generate_1d_batched(model, ctx_tokens, L, T, M, V, sep_token, device):
    """Batched greedy AR of rows M..T-1. See analysis/ar_cell_acc_1d.py."""
    B = ctx_tokens.shape[0]
    tokens = ctx_tokens.to(device)
    for t in range(M, T):
        for _ in range(L):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                logits = model(tokens)
            logits = logits.float()
            pred = logits[:, -1, :V].argmax(dim=-1)   # drop SEP logit
            tokens = torch.cat([tokens, pred.unsqueeze(1)], dim=1)
        if t < T - 1:
            sep_col = torch.full((B, 1), sep_token,
                                 dtype=torch.long, device=device)
            tokens = torch.cat([tokens, sep_col], dim=1)
    gen = torch.empty((B, T - M, L), dtype=torch.long, device=device)
    for i, t in enumerate(range(M, T)):
        base = t * (L + 1)
        gen[:, i, :] = tokens[:, base:base + L]
    return gen


# =============================================================================
# 2D VN AR rollout helpers
# =============================================================================

def _build_2d_types_and_steps(L1: int, L2: int, T: int, use_row_sep: bool):
    """Per-position ('cell'|'row_sep'|'time_sep', step_index)."""
    types, step_of = [], []
    for t in range(T):
        for r in range(L1):
            types.extend(["cell"] * L2)
            step_of.extend([t] * L2)
            if use_row_sep and r < L1 - 1:
                types.append("row_sep")
                step_of.append(t)
        if t < T - 1:
            types.append("time_sep")
            step_of.append(t)
    return types, step_of


def _ar_generate_2d_batched(model, context, seq_full, types,
                             ROW_SEP, TIME_SEP, device):
    """Batched greedy AR filling positions [ctx_len, seq_full)."""
    tokens = context.to(device)
    B = tokens.shape[0]
    while tokens.shape[1] < seq_full:
        pos = tokens.shape[1]
        t_type = types[pos]
        if t_type == "row_sep":
            nxt = torch.full((B, 1), ROW_SEP, dtype=torch.long, device=device)
        elif t_type == "time_sep":
            nxt = torch.full((B, 1), TIME_SEP, dtype=torch.long, device=device)
        else:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                logits = model(tokens)
            nxt = logits[:, -1].float().argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, nxt], dim=1)
    return tokens


# =============================================================================
# Autoregressive cell accuracy — unified public entry
# =============================================================================

def evaluate_autoregressive_cell_accuracy(model, dataloader, device, *,
                                          mode: str,
                                          num_context_states: int,
                                          first_predicted_timestamp: int,
                                          last_predicted_timestamp: int,
                                          # 1D-specific
                                          L: Optional[int] = None,
                                          V: Optional[int] = None,
                                          sep_token: Optional[int] = None,
                                          # 2D-specific
                                          L1: Optional[int] = None,
                                          L2: Optional[int] = None,
                                          use_row_sep: bool = True,
                                          row_sep_token: Optional[int] = None,
                                          time_sep_token: Optional[int] = None,
                                          verify_2d_layout: bool = True,
                                          use_ddp: Optional[bool] = None) -> dict:
    """Greedy autoregressive rollout cell accuracy over the full dataloader.

    Convention (1-indexed timestamps):
      * A trajectory has states 1..T. State 1 is the initial condition.
      * `num_context_states = M` means the model is fed the first M ground-truth
        states (rows/steps 1..M) and asked to AR-generate states M+1..T.
      * `first_predicted_timestamp` must equal M+1.
      * `last_predicted_timestamp` must equal T.
      Between-cell structural tokens (1D SEP, 2D ROW_SEP / TIME_SEP) are
      injected at their known positions; the model only produces cell tokens.

    Modes:
      - mode="1d": pass L, V, sep_token.
      - mode="2d": pass L1, L2, use_row_sep, row_sep_token, time_sep_token.

    Returns:
        {
          "cell_acc_by_timestamp": {t: float},              # t = M+1..T
          "cell_acc_overall": float,
          "total_correct_by_timestamp": {t: int},
          "total_cells_by_timestamp": {t: int},
          "total_correct_overall": int,
          "total_cells_overall": int,
        }
    """
    M = int(num_context_states)
    T = int(last_predicted_timestamp)
    assert first_predicted_timestamp == M + 1, (
        f"first_predicted_timestamp ({first_predicted_timestamp}) must equal "
        f"num_context_states + 1 ({M + 1}); partial-range scoring not supported")
    assert T > M, f"last_predicted_timestamp ({T}) must be > num_context_states ({M})"

    if mode == "1d":
        assert L is not None and V is not None and sep_token is not None, \
            "mode='1d' requires L, V, sep_token"
        return _ar_1d(model, dataloader, device, L=L, V=V, T=T, M=M,
                      sep_token=sep_token, use_ddp=use_ddp)
    if mode == "2d":
        assert L1 is not None and L2 is not None, \
            "mode='2d' requires L1, L2"
        assert row_sep_token is not None or not use_row_sep, \
            "mode='2d' with use_row_sep=True requires row_sep_token"
        assert time_sep_token is not None, \
            "mode='2d' requires time_sep_token"
        return _ar_2d(model, dataloader, device, L1=L1, L2=L2, T=T, M=M,
                      use_row_sep=use_row_sep,
                      row_sep_token=row_sep_token,
                      time_sep_token=time_sep_token,
                      verify_layout=verify_2d_layout,
                      use_ddp=use_ddp)
    raise ValueError(f"mode must be '1d' or '2d', got {mode!r}")


def _ar_1d(model, dataloader, device, *,
           L, V, T, M, sep_token, use_ddp) -> dict:
    if use_ddp is None:
        use_ddp = _autodetect_ddp()
    was_training = model.training
    model.eval()

    steps = T - M
    n_correct = torch.zeros(steps, dtype=torch.float64, device=device)
    n_total = torch.zeros(steps, dtype=torch.float64, device=device)

    try:
        with torch.inference_mode():
            for batch in dataloader:
                X, y = batch[0], batch[1]
                X = X.to(device, non_blocking=True).long()
                y = y.to(device, non_blocking=True).long()
                B = X.shape[0]

                full = _reconstruct_full_1d(X, y)          # (B, T*(L+1)-1)
                grid = _extract_cell_grid_1d(full, T, L)   # (B, T, L)
                ctx = full[:, :M * (L + 1)]                # rows 0..M-1 + SEPs

                gen = _ar_generate_1d_batched(
                    model, ctx, L, T, M, V, sep_token, device)  # (B, T-M, L)
                gt_gen = grid[:, M:]                       # (B, T-M, L)
                correct = (gen == gt_gen)                  # (B, T-M, L)
                n_correct += correct.sum(dim=(0, 2)).double()
                n_total += torch.full_like(n_correct, float(B * L))

        if use_ddp:
            _all_reduce_sum(n_correct)
            _all_reduce_sum(n_total)
    finally:
        if was_training:
            model.train()

    return _finalize_ar_metrics(n_correct.cpu().numpy(),
                                 n_total.cpu().numpy(),
                                 first_ts=M + 1)


def _ar_2d(model, dataloader, device, *,
           L1, L2, T, M, use_row_sep, row_sep_token, time_sep_token,
           verify_layout, use_ddp) -> dict:
    if use_ddp is None:
        use_ddp = _autodetect_ddp()
    was_training = model.training
    model.eval()

    types, step_of = _build_2d_types_and_steps(L1, L2, T, use_row_sep)
    seq_full = len(types)
    # First position whose step_of == M and type == 'cell' is where AR begins.
    ctx_len = next(i for i in range(seq_full)
                   if step_of[i] == M and types[i] == "cell")
    step_cells = {s: [i for i in range(seq_full)
                      if step_of[i] == s and types[i] == "cell"]
                  for s in range(M, T)}

    steps = T - M
    n_correct = torch.zeros(steps, dtype=torch.float64, device=device)
    n_total = torch.zeros(steps, dtype=torch.float64, device=device)

    layout_checked = not verify_layout

    try:
        with torch.inference_mode():
            for batch in dataloader:
                X, y = batch[0], batch[1]
                # Reconstruct full sequence (B, seq_full)
                full = torch.cat([X[:, :1], y], dim=1)
                assert full.shape[1] == seq_full, (
                    f"2D full length {full.shape[1]} != expected {seq_full}"
                    f" (L1={L1}, L2={L2}, T={T}, use_row_sep={use_row_sep})")
                X_dev = X.to(device, non_blocking=True).long()
                gt = full.to(device, non_blocking=True).long()

                if not layout_checked:
                    # One-time guard: SEP layout + token ids match the data.
                    for tag, tok in (("row_sep", row_sep_token),
                                     ("time_sep", time_sep_token)):
                        pos = [i for i, ty in enumerate(types) if ty == tag]
                        if pos and tok is not None:
                            bad = int((gt[:, pos] != tok).sum().item())
                            assert bad == 0, (
                                f"{tag}: expected token {tok} at {len(pos)} "
                                f"positions, {bad} mismatches in data — "
                                "SEP layout/id wrong")
                    cell_idx = [i for i, ty in enumerate(types)
                                if ty == "cell"]
                    max_cell = int(gt[:, cell_idx].max().item())
                    assert max_cell < min(
                        [tok for tok in (row_sep_token, time_sep_token)
                         if tok is not None] or [10**9]
                    ), "cell positions contain SEP tokens"
                    layout_checked = True

                ctx = X_dev[:, :ctx_len]
                gen = _ar_generate_2d_batched(
                    model, ctx, seq_full, types,
                    row_sep_token, time_sep_token, device)  # (B, seq_full)

                for i, s in enumerate(range(M, T)):
                    pos = step_cells[s]
                    eq = (gen[:, pos] == gt[:, pos])       # (B, n_cells)
                    n_correct[i] += eq.sum().double()
                    n_total[i] += float(eq.numel())

        if use_ddp:
            _all_reduce_sum(n_correct)
            _all_reduce_sum(n_total)
    finally:
        if was_training:
            model.train()

    return _finalize_ar_metrics(n_correct.cpu().numpy(),
                                 n_total.cpu().numpy(),
                                 first_ts=M + 1)


def _finalize_ar_metrics(n_correct: np.ndarray,
                         n_total: np.ndarray,
                         first_ts: int) -> dict:
    per_t_correct = {}
    per_t_total = {}
    per_t_acc = {}
    for i, nc in enumerate(n_correct):
        t = first_ts + i
        c = int(nc)
        n = int(n_total[i])
        per_t_correct[t] = c
        per_t_total[t] = n
        per_t_acc[t] = (c / n) if n > 0 else 0.0
    tot_c = int(n_correct.sum())
    tot_n = int(n_total.sum())
    return {
        "cell_acc_by_timestamp": per_t_acc,
        "cell_acc_overall": (tot_c / tot_n) if tot_n > 0 else 0.0,
        "total_correct_by_timestamp": per_t_correct,
        "total_cells_by_timestamp": per_t_total,
        "total_correct_overall": tot_c,
        "total_cells_overall": tot_n,
    }


# =============================================================================
# =============================================================================
# Stochastic evaluators (teacher-forced + autoregressive two-stream sampling)
# =============================================================================
# =============================================================================

# ---------- Numerics ----------

def _kl_pq(p: torch.Tensor, q: torch.Tensor,
           eps: float = 1e-12) -> torch.Tensor:
    """Forward KL(p || q) = sum_v p_v log(p_v / q_v).

    Convention throughout this module: FIRST arg is the reference distribution,
    SECOND is the predictor. Callers pick the direction by argument order:
        kl_kgram_model = _kl_pq(q_kgram, q_model)     # KL(kgram || model)
        kl_true_model  = _kl_pq(p_true,  q_model)     # KL(true  || model)
        kl_true_kgram  = _kl_pq(p_true,  q_kgram)     # KL(true  || kgram)

    Handles p=0 via 0*log0=0 convention. Requires q strictly positive at every
    coordinate p>0 puts mass on; callers must guarantee this (add-one k-gram,
    softmax, or clamp-eps p_true).
    """
    logp = torch.where(p > 0, torch.log(p.clamp_min(eps)),
                       torch.zeros_like(p))
    logq = torch.log(q.clamp_min(eps))
    return (p * (logp - logq)).sum(dim=-1)


def _self_test_kl_direction():
    """Numerical sanity check: KL is asymmetric, and this module's helper
    respects the (reference, predictor) argument order. Run manually via
      python -c "from evaluators import _self_test_kl_direction as t; t()"
    """
    import math
    p = torch.tensor([[0.7, 0.3]], dtype=torch.float64)
    q = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    kl_pq = float(_kl_pq(p, q).item())
    kl_qp = float(_kl_pq(q, p).item())
    expected_pq = 0.7 * math.log(0.7 / 0.5) + 0.3 * math.log(0.3 / 0.5)
    expected_qp = 0.5 * math.log(0.5 / 0.7) + 0.5 * math.log(0.5 / 0.3)
    assert abs(kl_pq - expected_pq) < 1e-10, (kl_pq, expected_pq)
    assert abs(kl_qp - expected_qp) < 1e-10, (kl_qp, expected_qp)
    assert abs(kl_pq - kl_qp) > 1e-6, "KL(p||q) must differ from KL(q||p) here"
    print(f"OK  KL(p||q)={kl_pq:.6f}  KL(q||p)={kl_qp:.6f}")


# ---------- Neighborhood config indexing (base-V) ----------

def _cfg_idx_1d(g: torch.Tensor, V: int) -> torch.Tensor:
    """1D K=3 cyclic neighborhood config per cell in base-V on the LAST axis.
    g: (..., L) int -> (..., L) int in [0, V**3). Matches kl_metrics._cfg_idx_lastdim."""
    L = torch.roll(g, shifts=1, dims=-1)
    C = g
    R = torch.roll(g, shifts=-1, dims=-1)
    return (L * V + C) * V + R


def _cfg_idx_2d_vn(g: torch.Tensor, V: int) -> torch.Tensor:
    """2D VN K=5 cyclic neighborhood config per cell in base-V (NWCES encoding).
    g: (..., L1, L2) int -> same-shape int in [0, V**5). Matches
    kl_metrics._cfg_idx_vn2d and data_generate_2d_stoch._cfg_idx."""
    N = torch.roll(g, shifts=1, dims=-2)
    W = torch.roll(g, shifts=1, dims=-1)
    C = g
    E = torch.roll(g, shifts=-1, dims=-1)
    S = torch.roll(g, shifts=-1, dims=-2)
    return ((((N * V + W) * V + C) * V + E) * V + S)


# ---------- Batch helpers ----------

def _slice_rules_batch(rules_np: np.ndarray, sample_offset: int,
                       B: int, device: torch.device) -> torch.Tensor:
    """Return (B, NC, V) float64 tensor of per-sample stochastic rule rows for
    the current batch, aligned with the shuffle=False dataloader order."""
    end = sample_offset + B
    slab = np.asarray(rules_np[sample_offset:end])
    return torch.as_tensor(slab, device=device, dtype=torch.float64)


def _model_forward_cell_logits(model, tokens: torch.Tensor,
                                V: int) -> torch.Tensor:
    """Run model on `tokens` and return the *last* position's cell logits
    (`[:, :V]`, SEP logit dropped). Returns float32 for numerical stability
    downstream; caller casts to double for KL.
    """
    with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                            enabled=(tokens.device.type == 'cuda')):
        logits = model(tokens)
    return logits[:, -1, :V].float()


# =============================================================================
# Stochastic — TEACHER-FORCED
# =============================================================================

def evaluate_stochastic_teacher_forced(model, dataloader, device, *,
                                       mode: str, T: int, M: int, V: int,
                                       K: Optional[int] = None,
                                       L: Optional[int] = None,
                                       L1: Optional[int] = None,
                                       L2: Optional[int] = None,
                                       rules_np: np.ndarray,
                                       use_ddp: Optional[bool] = None,
                                       eps: float = 1e-12) -> dict:
    """Teacher-forced stochastic eval.

    For every valid cell target at t >= 1, compute three KL directions and two
    argmax cell accuracies. Aggregation is restricted to positions where
    eval_mask > 0 (the dataloader's test mask, i.e. t >= M in the test split).
    The k-gram counter, however, walks EVERY valid causal transition regardless
    of the mask, so the baseline sees the same context the model sees.

    Returns:
        {
          "kl_kgram_model", "kl_true_model", "kl_true_kgram",
          "cell_acc", "kgram_cell_acc",
          plus sums / counts fields used to build them (all int/float64,
          post-all-reduce when DDP is on).
        }
    """
    if use_ddp is None:
        use_ddp = _autodetect_ddp()
    assert mode in ("1d", "2d")
    if mode == "1d":
        assert L is not None and K is not None
        assert K == 3, "1D stochastic evaluator assumes K=3"
        NC = V ** K
        n_cells = L
        # Positions of the T*L cell tokens in the full sequence
        cell_dst = ((torch.arange(T * L, device=device) // L) * (L + 1)
                    + (torch.arange(T * L, device=device) % L))
    else:
        assert L1 is not None and L2 is not None
        K = 5
        NC = V ** K
        n_cells = L1 * L2
        # 2D VN cell_dst
        row_len = L1 * L2 + (L1 - 1)   # includes row_seps
        cd = []
        pos = 0
        for _t in range(T):
            for r in range(L1):
                for _c in range(L2):
                    cd.append(pos); pos += 1
                if r < L1 - 1:
                    pos += 1     # ROW_SEP
            if _t < T - 1:
                pos += 1         # TIME_SEP
        cell_dst = torch.tensor(cd, dtype=torch.long, device=device)

    was_training = model.training
    model.eval()

    # Aggregators (float64 for KL, int for counts). Denominators use scored
    # cells (mask>0). Counts are updated on every valid causal transition.
    # Convention: reference || predictor (first arg is the reference).
    kl_km_sum = torch.zeros((), dtype=torch.float64, device=device)  # KL(q_kgram||q_model)
    kl_tm_sum = torch.zeros((), dtype=torch.float64, device=device)  # KL(p_true||q_model)
    kl_tk_sum = torch.zeros((), dtype=torch.float64, device=device)  # KL(p_true||q_kgram)
    n_scored = torch.zeros((), dtype=torch.float64, device=device)
    n_correct_model = torch.zeros((), dtype=torch.float64, device=device)
    n_correct_kgram = torch.zeros((), dtype=torch.float64, device=device)

    sample_offset = 0
    try:
        with torch.inference_mode():
            for batch in dataloader:
                X, y, mask = batch[0], batch[1], batch[2]
                X = X.to(device, non_blocking=True).long()
                y = y.to(device, non_blocking=True).long()
                mask = mask.to(device, non_blocking=True)
                B = X.shape[0]

                # Model forward on whole (teacher-forced) sequence
                with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                    logits = model(X)
                logits = logits.float()               # (B, seq_len, output_size)

                # Reconstruct full GT sequence + cell grid
                full = torch.cat([X[:, :1], y], dim=1)
                if mode == "1d":
                    states = full[:, cell_dst].view(B, T, L).long()
                    cfg_all = _cfg_idx_1d(states, V)          # (B, T, L)
                else:
                    states = full[:, cell_dst].view(B, T, L1, L2).long()
                    cfg_all = _cfg_idx_2d_vn(states, V)       # (B, T, L1, L2)

                # Per-sample rule tables aligned to shuffle=False order
                rules_b = _slice_rules_batch(rules_np, sample_offset, B, device)
                assert rules_b.shape == (B, NC, V), (
                    f"rules batch shape {rules_b.shape} != {(B, NC, V)}")

                counts = torch.zeros((B, NC, V), dtype=torch.float64,
                                     device=device)
                b_range = torch.arange(B, device=device)

                for t in range(1, T):
                    # Iterate cells in the same left-to-right, row-major order
                    # they appear in the serialized sequence.
                    if mode == "1d":
                        rng = [(c,) for c in range(L)]
                    else:
                        rng = [(r, c) for r in range(L1) for c in range(L2)]
                    for i, coord in enumerate(rng):
                        if mode == "1d":
                            c = coord[0]
                            cfg_c = cfg_all[:, t - 1, c]      # (B,)
                            outcome = states[:, t, c]         # (B,)
                            flat = t * L + c
                        else:
                            r, c = coord
                            cfg_c = cfg_all[:, t - 1, r, c]
                            outcome = states[:, t, r, c]
                            flat = (t * L1 + r) * L2 + c
                        # y-axis position corresponding to this cell target
                        j = int(cell_dst[flat].item()) - 1
                        w = mask[:, j].to(torch.float64)      # (B,) 0/1

                        # p_true from rule row, clamp for KL safety
                        p_true = rules_b[b_range, cfg_c]      # (B, V)

                        # q_model: softmax over first V logits at position j
                        q_model = torch.softmax(
                            logits[:, j, :V].double(), dim=-1)

                        # q_kgram from causal add-1 counts (BEFORE update)
                        N = counts[b_range, cfg_c]            # (B, V)
                        q_kgram = (N + 1.0) / (N.sum(dim=-1, keepdim=True) + V)

                        # KL directions: reference || predictor.
                        kl_km = _kl_pq(q_kgram, q_model, eps=eps)  # KL(kgram||model)
                        kl_tm = _kl_pq(p_true, q_model, eps=eps)   # KL(true||model)
                        kl_tk = _kl_pq(p_true, q_kgram, eps=eps)   # KL(true||kgram)

                        # Argmax accuracies against GT outcome
                        pred_m = q_model.argmax(dim=-1)
                        pred_k = q_kgram.argmax(dim=-1)
                        ok_m = (pred_m == outcome).to(torch.float64)
                        ok_k = (pred_k == outcome).to(torch.float64)

                        kl_km_sum += (kl_km * w).sum()
                        kl_tm_sum += (kl_tm * w).sum()
                        kl_tk_sum += (kl_tk * w).sum()
                        n_correct_model += (ok_m * w).sum()
                        n_correct_kgram += (ok_k * w).sum()
                        n_scored += w.sum()

                        # Update AFTER metrics (counts include this transition
                        # for the NEXT cell's k-gram baseline).
                        counts[b_range, cfg_c, outcome] += 1.0

                sample_offset += B

        if use_ddp:
            for t_ in (kl_km_sum, kl_tm_sum, kl_tk_sum,
                       n_scored, n_correct_model, n_correct_kgram):
                _all_reduce_sum(t_)
    finally:
        if was_training:
            model.train()

    n = float(n_scored.item())
    def _avg(t):
        return (float(t.item()) / n) if n > 0 else 0.0
    return {
        # Reference || predictor convention.
        "kl_kgram_model": _avg(kl_km_sum),
        "kl_true_model": _avg(kl_tm_sum),
        "kl_true_kgram": _avg(kl_tk_sum),
        "cell_acc": _avg(n_correct_model),
        "kgram_cell_acc": _avg(n_correct_kgram),
        "n_scored_cells": int(n_scored.item()),
        "sum_kl_kgram_model": float(kl_km_sum.item()),
        "sum_kl_true_model": float(kl_tm_sum.item()),
        "sum_kl_true_kgram": float(kl_tk_sum.item()),
        "n_correct_model": int(n_correct_model.item()),
        "n_correct_kgram": int(n_correct_kgram.item()),
    }


# =============================================================================
# Stochastic — AUTOREGRESSIVE (two independent streams)
# =============================================================================

def evaluate_stochastic_autoregressive(model, dataloader, device, *,
                                       mode: str, T: int, M: int, V: int,
                                       K: Optional[int] = None,
                                       # 1D
                                       L: Optional[int] = None,
                                       sep_token: Optional[int] = None,
                                       # 2D
                                       L1: Optional[int] = None,
                                       L2: Optional[int] = None,
                                       use_row_sep: bool = True,
                                       row_sep_token: Optional[int] = None,
                                       time_sep_token: Optional[int] = None,
                                       rules_np: np.ndarray,
                                       num_rollouts: int = 1,
                                       seed: int = 42,
                                       use_ddp: Optional[bool] = None,
                                       eps: float = 1e-12) -> dict:
    """Two-stream sampled autoregressive stochastic eval.

    Stream A (MODEL rollout): feeds model tokens one-at-a-time; at each cell
      forwards the model, softmaxes over V, samples one token from that
      distribution (temperature 1), appends it to the prefix, and updates
      stream A's own kgram counter with the SAMPLED outcome.

    Stream B (KGRAM rollout): from the same M-state context, generates only
      via the causal add-one kgram, samples from q_kgram at each cell,
      updates stream B's own counter with the SAMPLED outcome. Never sees
      the model.

    Both streams start with identical grids (rows/steps 0..M-1 = GT) and
    identical counters (all GT transitions at t < M added in). At t >= M
    they diverge — independent prefixes and counters.

    Accuracy: for each cell we compare the stream's sampled token with the
    saved stochastic test-trajectory token at the same (t, coord). This is a
    trajectory-agreement metric, not a probability-recovery one — even a
    perfect stochastic model will not approach 1.

    Returns:
        {
          # Stream A
          "kl_kgram_model", "kl_true_model", "model_cell_acc",
          # Stream B
          "kl_true_kgram", "kgram_cell_acc",
          # plus raw sums/counts and denominators, and metadata.
        }
    """
    if use_ddp is None:
        use_ddp = _autodetect_ddp()
    assert mode in ("1d", "2d")
    assert num_rollouts >= 1

    if mode == "1d":
        assert L is not None and sep_token is not None and K is not None
        assert K == 3
    else:
        assert L1 is not None and L2 is not None and time_sep_token is not None
        K = 5
        if use_row_sep:
            assert row_sep_token is not None
    NC = V ** K

    was_training = model.training
    model.eval()

    # Per-timestamp aggregators (float64). Indexed by (t - M) for t in M..T-1.
    # Overall values are derived by summing along axis 0 at the end so the
    # user sees `overall = total_sum / total_count`, never a mean-of-means.
    # Convention: reference || predictor (first arg = reference).
    steps = T - M
    kl_km_sum_t = torch.zeros(steps, dtype=torch.float64, device=device)  # KL(q_kgram_A||q_model_A)
    kl_tm_sum_t = torch.zeros(steps, dtype=torch.float64, device=device)  # KL(p_true_A||q_model_A)
    kl_tk_sum_t = torch.zeros(steps, dtype=torch.float64, device=device)  # KL(p_true_B||q_kgram_B)
    n_scored_A_t = torch.zeros(steps, dtype=torch.float64, device=device)
    n_scored_B_t = torch.zeros(steps, dtype=torch.float64, device=device)
    n_correct_A_t = torch.zeros(steps, dtype=torch.float64, device=device)
    n_correct_B_t = torch.zeros(steps, dtype=torch.float64, device=device)

    # 2D positional layout (built once)
    if mode == "2d":
        types, step_of = _build_2d_types_and_steps(L1, L2, T, use_row_sep)
        seq_full = len(types)
        # Position of the very first cell of step M — where AR starts.
        ctx_len_2d = next(i for i in range(seq_full)
                          if step_of[i] == M and types[i] == "cell")
    else:
        seq_full = T * (L + 1) - 1

    sample_offset = 0
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                X, y = batch[0], batch[1]
                X = X.to(device, non_blocking=True).long()
                y = y.to(device, non_blocking=True).long()
                B = X.shape[0]

                # Full GT sequence + cell grid (for saved-trajectory compare
                # and for seeding both streams' first-M states / counters).
                full = torch.cat([X[:, :1], y], dim=1)
                if mode == "1d":
                    # cell_dst 1D
                    cell_dst = ((torch.arange(T * L, device=device) // L)
                                * (L + 1)
                                + (torch.arange(T * L, device=device) % L))
                    gt_grid = full[:, cell_dst].view(B, T, L).long()
                else:
                    # cell_dst 2D VN
                    cd = []
                    pos = 0
                    for _t in range(T):
                        for r in range(L1):
                            for _c in range(L2):
                                cd.append(pos); pos += 1
                            if r < L1 - 1:
                                pos += 1
                        if _t < T - 1:
                            pos += 1
                    cell_dst = torch.tensor(cd, dtype=torch.long, device=device)
                    gt_grid = full[:, cell_dst].view(B, T, L1, L2).long()

                # Per-sample rule tables (aligned, shuffle=False)
                rules_b = _slice_rules_batch(rules_np, sample_offset, B, device)
                assert rules_b.shape == (B, NC, V), (
                    f"rules batch shape {rules_b.shape} != {(B, NC, V)}")
                b_range = torch.arange(B, device=device)

                for roll_idx in range(num_rollouts):
                    gen_A = torch.Generator(device=device.type)
                    gen_A.manual_seed(int(seed) * 1_000_003
                                      + batch_idx * 1009 + roll_idx * 17 + 1)
                    gen_B = torch.Generator(device=device.type)
                    gen_B.manual_seed(int(seed) * 1_000_003
                                      + batch_idx * 1009 + roll_idx * 17 + 2)

                    # -------- Init streams --------
                    # Both grids initialised with GT rows/steps 0..M-1.
                    if mode == "1d":
                        grid_A = gt_grid.clone()   # (B, T, L)
                        grid_B = gt_grid.clone()
                    else:
                        grid_A = gt_grid.clone()   # (B, T, L1, L2)
                        grid_B = gt_grid.clone()
                    # Zero out t >= M so anything written later is stream-only.
                    grid_A[:, M:] = 0
                    grid_B[:, M:] = 0

                    counts_A = torch.zeros((B, NC, V), dtype=torch.float64,
                                            device=device)
                    counts_B = torch.zeros((B, NC, V), dtype=torch.float64,
                                            device=device)

                    # Pre-seed counters with GT transitions from t in 1..M-1.
                    if mode == "1d":
                        cfg_ctx = _cfg_idx_1d(gt_grid, V)     # (B, T, L)
                        for t in range(1, M):
                            for c in range(L):
                                cfg = cfg_ctx[:, t - 1, c]
                                out_ = gt_grid[:, t, c]
                                counts_A[b_range, cfg, out_] += 1.0
                                counts_B[b_range, cfg, out_] += 1.0
                    else:
                        cfg_ctx = _cfg_idx_2d_vn(gt_grid, V)  # (B, T, L1, L2)
                        for t in range(1, M):
                            for r in range(L1):
                                for c in range(L2):
                                    cfg = cfg_ctx[:, t - 1, r, c]
                                    out_ = gt_grid[:, t, r, c]
                                    counts_A[b_range, cfg, out_] += 1.0
                                    counts_B[b_range, cfg, out_] += 1.0

                    # -------- Stream A: model prefix tokens --------
                    if mode == "1d":
                        # ctx = rows 0..M-1 + M-1 inter-row SEPs + 1 SEP after
                        # row M-1 = M*(L+1) tokens. Same as det AR.
                        tokens_A = full[:, :M * (L + 1)].clone().long()
                    else:
                        tokens_A = X[:, :ctx_len_2d].clone().long()

                    # -------- Rollout t = M..T-1 --------
                    for t in range(M, T):
                        # Config for cells at row/step t comes from the
                        # COMPLETE previous row/step of the CORRESPONDING stream.
                        if mode == "1d":
                            prev_A = grid_A[:, t - 1]         # (B, L)
                            prev_B = grid_B[:, t - 1]
                            cfg_A_row = _cfg_idx_1d(prev_A, V)  # (B, L)
                            cfg_B_row = _cfg_idx_1d(prev_B, V)
                            iter_coords = [(c,) for c in range(L)]
                        else:
                            prev_A = grid_A[:, t - 1]         # (B, L1, L2)
                            prev_B = grid_B[:, t - 1]
                            cfg_A_row = _cfg_idx_2d_vn(prev_A, V)  # (B, L1, L2)
                            cfg_B_row = _cfg_idx_2d_vn(prev_B, V)
                            iter_coords = [(r, c) for r in range(L1)
                                                    for c in range(L2)]

                        for coord_i, coord in enumerate(iter_coords):
                            if mode == "1d":
                                c = coord[0]
                                cfg_A = cfg_A_row[:, c]
                                cfg_B = cfg_B_row[:, c]
                                gt_out = gt_grid[:, t, c]
                            else:
                                r, c = coord
                                cfg_A = cfg_A_row[:, r, c]
                                cfg_B = cfg_B_row[:, r, c]
                                gt_out = gt_grid[:, t, r, c]

                            # p_true according to EACH stream's own cfg
                            p_true_A = rules_b[b_range, cfg_A]
                            p_true_B = rules_b[b_range, cfg_B]

                            # ---- Stream A: model forward, then sample ----
                            q_model_A = torch.softmax(
                                _model_forward_cell_logits(model, tokens_A, V)
                                .double(), dim=-1)              # (B, V)
                            N_A = counts_A[b_range, cfg_A]
                            q_kgram_A = (N_A + 1.0) / (
                                N_A.sum(dim=-1, keepdim=True) + V)

                            # KL(kgram_A || model_A), KL(true_A || model_A).
                            kl_km_sum_t[t - M] += _kl_pq(
                                q_kgram_A, q_model_A, eps).sum()
                            kl_tm_sum_t[t - M] += _kl_pq(
                                p_true_A, q_model_A, eps).sum()

                            samp_A = torch.multinomial(
                                q_model_A.float(), num_samples=1,
                                generator=gen_A).squeeze(-1)    # (B,)
                            n_correct_A_t[t - M] += (
                                samp_A == gt_out).to(torch.float64).sum()
                            n_scored_A_t[t - M] += float(B)

                            # Append sampled token to A's prefix
                            tokens_A = torch.cat(
                                [tokens_A, samp_A.unsqueeze(1)], dim=1)
                            # Store in A's grid
                            if mode == "1d":
                                grid_A[:, t, c] = samp_A
                            else:
                                grid_A[:, t, r, c] = samp_A
                            # Update A's kgram counter with A's own outcome
                            counts_A[b_range, cfg_A, samp_A] += 1.0

                            # ---- Stream B: sample from q_kgram ----
                            N_B = counts_B[b_range, cfg_B]
                            q_kgram_B = (N_B + 1.0) / (
                                N_B.sum(dim=-1, keepdim=True) + V)
                            # KL(true_B || kgram_B).
                            kl_tk_sum_t[t - M] += _kl_pq(
                                p_true_B, q_kgram_B, eps).sum()

                            samp_B = torch.multinomial(
                                q_kgram_B.float(), num_samples=1,
                                generator=gen_B).squeeze(-1)
                            n_correct_B_t[t - M] += (
                                samp_B == gt_out).to(torch.float64).sum()
                            n_scored_B_t[t - M] += float(B)

                            if mode == "1d":
                                grid_B[:, t, c] = samp_B
                            else:
                                grid_B[:, t, r, c] = samp_B
                            counts_B[b_range, cfg_B, samp_B] += 1.0

                            # ---- Structural token injection into A ----
                            # (B does not maintain a token stream.)
                            if mode == "2d" and coord_i < len(iter_coords) - 1:
                                r_here, c_here = coord
                                # After the last col of a row (that isn't the
                                # last row of the step), insert ROW_SEP.
                                if (use_row_sep and c_here == L2 - 1
                                        and r_here < L1 - 1):
                                    sep = torch.full((B, 1), row_sep_token,
                                                     dtype=torch.long,
                                                     device=device)
                                    tokens_A = torch.cat([tokens_A, sep], dim=1)

                        # End of this row/step
                        if mode == "1d":
                            if t < T - 1:
                                sep = torch.full((B, 1), sep_token,
                                                 dtype=torch.long,
                                                 device=device)
                                tokens_A = torch.cat([tokens_A, sep], dim=1)
                        else:
                            if t < T - 1:
                                sep = torch.full((B, 1), time_sep_token,
                                                 dtype=torch.long,
                                                 device=device)
                                tokens_A = torch.cat([tokens_A, sep], dim=1)

                sample_offset += B

        if use_ddp:
            for t_ in (kl_km_sum_t, kl_tm_sum_t, kl_tk_sum_t,
                       n_scored_A_t, n_scored_B_t,
                       n_correct_A_t, n_correct_B_t):
                _all_reduce_sum(t_)
    finally:
        if was_training:
            model.train()

    kl_km_np = kl_km_sum_t.cpu().numpy()
    kl_tm_np = kl_tm_sum_t.cpu().numpy()
    kl_tk_np = kl_tk_sum_t.cpu().numpy()
    nA_np = n_scored_A_t.cpu().numpy()
    nB_np = n_scored_B_t.cpu().numpy()
    cA_np = n_correct_A_t.cpu().numpy()
    cB_np = n_correct_B_t.cpu().numpy()

    # 1-indexed timestamp labels, matching the deterministic convention: the
    # first predicted state is t = M+1, the last is t = T.
    ts = list(range(M + 1, T + 1))

    def _safe(num, den):
        return float(num / den) if den > 0 else 0.0

    by_timestamp = {
        t: {
            "kl_kgram_model": _safe(kl_km_np[i], nA_np[i]),
            "kl_true_model": _safe(kl_tm_np[i], nA_np[i]),
            "kl_true_kgram": _safe(kl_tk_np[i], nB_np[i]),
            "model_cell_acc": _safe(cA_np[i], nA_np[i]),
            "kgram_cell_acc": _safe(cB_np[i], nB_np[i]),
        }
        for i, t in enumerate(ts)
    }
    sum_kl_km_all = float(kl_km_np.sum())
    sum_kl_tm_all = float(kl_tm_np.sum())
    sum_kl_tk_all = float(kl_tk_np.sum())
    nA_all = float(nA_np.sum())
    nB_all = float(nB_np.sum())
    cA_all = float(cA_np.sum())
    cB_all = float(cB_np.sum())
    return {
        # Overall (single-scalar) values — each computed once from raw totals,
        # never as an average of per-timestamp averages.
        # Convention: reference || predictor.
        "kl_kgram_model": _safe(sum_kl_km_all, nA_all),
        "kl_true_model": _safe(sum_kl_tm_all, nA_all),
        "kl_true_kgram": _safe(sum_kl_tk_all, nB_all),
        "model_cell_acc": _safe(cA_all, nA_all),
        "kgram_cell_acc": _safe(cB_all, nB_all),
        # Per-timestamp breakdown, keyed by 1-indexed timestep label.
        "by_timestamp": by_timestamp,
        "evaluated_timestamps": ts,
        # Raw sums & counts (for downstream inspection, DDP-safe post-reduce).
        "sum_kl_kgram_model": sum_kl_km_all,
        "sum_kl_true_model": sum_kl_tm_all,
        "sum_kl_true_kgram": sum_kl_tk_all,
        "n_scored_cells_A": int(nA_all),
        "n_scored_cells_B": int(nB_all),
        "n_correct_model": int(cA_all),
        "n_correct_kgram": int(cB_all),
        "num_rollouts": num_rollouts,
        "seed": seed,
    }


# =============================================================================
# =============================================================================
# Full teacher-forced evaluate() — drop-in replacement for eval.py::evaluate.
# Used by train.py's val loop. Returns loss + cell_acc + sequence_accuracy +
# per_rule_acc + per_timestep_acc + kl_true_model + kl_true_kgram +
# kl_kgram_model + kgram_cell_acc. Signature bit-identical to the old eval.py.
# =============================================================================
# =============================================================================


# =============================================================================
# Training-time teacher-forced evaluation (replaces the legacy evaluate() that
# depended on kl_metrics.py; verified equal to it on the saved val splits).
#
# Teacher-forced evaluation used during training (one function for 1D and 2D).
#
# Same metric definitions as the original evaluators.evaluate / kl_metrics.py:
#     loss                CE over masked target positions (cells and separators)
#     cell_acc            full-vocab argmax == target, over masked CELL targets
#     sequence_accuracy   fraction of sequences with every masked cell correct
#   stochastic only (KL in nats per scored cell; k-gram = causal add-one):
#     kl_true_model       KL(p_true  || q_model)      [needs rule tables]
#     kl_true_kgram       KL(p_true  || q_kgram)      [needs rule tables]
#     kl_kgram_model      KL(q_kgram || q_model)
#     kl_model_kgram      KL(q_model || q_kgram)      (legacy direction)
#     kgram_cell_acc      k-gram argmax == realized next state
# q_model is the softmax over the V cell logits. The k-gram counter walks every
# transition; only masked targets are averaged.
# =============================================================================

from .data import serialization as ser  # noqa: E402


def _config_index_torch(states: torch.Tensor, V: int, dimension: int) -> torch.Tensor:
    if dimension == 1:
        return (torch.roll(states, 1, -1) * V + states) * V + torch.roll(states, -1, -1)
    n, w = torch.roll(states, 1, -2), torch.roll(states, 1, -1)
    e, s = torch.roll(states, -1, -1), torch.roll(states, -1, -2)
    return (((n * V + w) * V + states) * V + e) * V + s


def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    logp = torch.where(p > 0, torch.log(p.clamp_min(eps)), torch.zeros_like(p))
    return (p * (logp - torch.log(q.clamp_min(eps)))).sum(-1)


@torch.inference_mode()
def evaluate(model, loader, spec, criterion, device, rules=None) -> dict:
    """rules: per-sample rule ids (deterministic, unused) or (N, V**K, V) tables
    aligned with the loader's (unshuffled) order."""
    was_training = model.training
    model.eval()
    V, T, C = spec.V, spec.T, spec.cells_per_frame
    stoch = spec.stochastic
    with_true = stoch and rules is not None and np.ndim(rules) == 3
    cell_pos = torch.as_tensor(ser.cell_positions(spec), device=device)
    amp = device.type == "cuda"

    tot = {k: torch.zeros((), dtype=torch.float64, device=device) for k in
           ("loss", "loss_n", "correct", "cells", "seq_ok", "seq_n",
            "kl_tm", "kl_tk", "kl_km", "kl_mk", "kg_ok", "kl_n")}
    offset = 0
    try:
        for X, y, mask in loader:
            X, y, mask = X.to(device), y.to(device), mask.to(device)
            B = X.shape[0]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(X)
            out = out.float()
            loss_raw = criterion(out.reshape(-1, out.shape[-1]), y.reshape(-1)).view(B, -1)
            m = mask > 0
            cell_m = m & (y < V)
            wrong = (out.argmax(-1) != y) & cell_m
            tot["loss"] += (loss_raw * m).sum()
            tot["loss_n"] += m.sum()
            tot["correct"] += (cell_m & ~wrong).sum()
            tot["cells"] += cell_m.sum()
            has = cell_m.any(1)
            tot["seq_ok"] += (has & ~wrong.any(1)).sum()
            tot["seq_n"] += has.sum()

            if stoch:
                full = torch.cat([X[:, :1], y], 1)
                states = full[:, cell_pos].view((B, T) + tuple(spec.grid_shape))
                cfg = _config_index_torch(states, V, spec.dimension).reshape(B, T, C)
                cells = states.reshape(B, T, C)
                q_all = torch.softmax(out[..., :V].double(), -1)
                counts = torch.zeros((B, spec.num_configs, V), dtype=torch.float64, device=device)
                b = torch.arange(B, device=device)
                if with_true:
                    table = torch.as_tensor(np.asarray(rules[offset:offset + B]),
                                            dtype=torch.float64, device=device)
                for t in range(1, T):
                    for c in range(C):
                        k = cfg[:, t - 1, c]
                        outcome = cells[:, t, c]
                        j = int(cell_pos[t * C + c]) - 1
                        w = mask[:, j].double()
                        q_model = q_all[:, j]
                        N = counts[b, k]
                        q_kgram = (N + 1.0) / (N.sum(-1, keepdim=True) + V)
                        tot["kl_km"] += (_kl(q_kgram, q_model) * w).sum()
                        tot["kl_mk"] += (_kl(q_model, q_kgram) * w).sum()
                        tot["kg_ok"] += ((q_kgram.argmax(-1) == outcome).double() * w).sum()
                        if with_true:
                            p_true = table[b, k]
                            tot["kl_tm"] += (_kl(p_true, q_model) * w).sum()
                            tot["kl_tk"] += (_kl(p_true, q_kgram) * w).sum()
                        tot["kl_n"] += w.sum()
                        counts[b, k, outcome] += 1.0
            offset += B
    finally:
        if was_training:
            model.train()

    v = {k: float(x) for k, x in tot.items()}
    metrics = {"loss": v["loss"] / max(v["loss_n"], 1),
               "cell_acc": v["correct"] / max(v["cells"], 1)}
    if v["seq_n"]:
        metrics["sequence_accuracy"] = v["seq_ok"] / v["seq_n"]
    if stoch and v["kl_n"]:
        n = v["kl_n"]
        metrics.update(kl_kgram_model=v["kl_km"] / n, kl_model_kgram=v["kl_mk"] / n,
                       kgram_cell_acc=v["kg_ok"] / n)
        if with_true:
            metrics.update(kl_true_model=v["kl_tm"] / n, kl_true_kgram=v["kl_tk"] / n)
    return metrics
