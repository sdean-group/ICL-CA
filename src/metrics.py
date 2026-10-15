"""Metrics on the scored frames M..T-1 (same definitions as evaluators.py).

Deterministic:
    tf/cell_acc, tf/seq_acc      teacher-forced argmax vs ground truth
    ar/cell_acc, ar/seq_acc      greedy rollout from the first M frames
    ar/cell_acc_by_step          per generated frame t = M..T-1

Stochastic (KL is always KL(reference || predictor), in nats per cell):
    tf/kl_kgram_model   KL(q_kgram || q_model)
    tf/kl_true_model    KL(p_true  || q_model)     (needs the rule table)
    tf/kl_true_kgram    KL(p_true  || q_kgram)     (baseline floor)
    tf/cell_acc, tf/kgram_cell_acc   argmax vs the realized next state
    ar/...              the same, along the model's own sampled rollout
                        ("stream A"); ar/kl_true_kgram and ar/kgram_cell_acc
                        come from an independent k-gram rollout ("stream B").

q_kgram is the causal add-one estimate from all transitions seen so far in the
sequence (in serialization order):
    q_kgram(v | c) = (N[c, v] + 1) / (sum_u N[c, u] + V)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .data.simulation import config_index


def _flat(spec, arr: np.ndarray) -> np.ndarray:
    """(B, T, *grid, [V]) -> (B, T, C, [V])."""
    B, T = arr.shape[:2]
    return arr.reshape((B, T, spec.cells_per_frame) + arr.shape[2 + len(spec.grid_shape):])


def kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL(p || q) over the last axis with 0 log 0 = 0."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    logp = np.where(p > 0, np.log(np.clip(p, eps, None)), 0.0)
    return (p * (logp - np.log(np.clip(q, eps, None)))).sum(-1)


def kgram_probs(spec, traj: np.ndarray) -> np.ndarray:
    """Causal add-one k-gram distribution for every cell of frames 1..T-1.

    traj (B, T, *grid) -> (B, T, *grid, V); frame 0 is NaN."""
    B, T = traj.shape[:2]
    V, C = spec.V, spec.cells_per_frame
    cfg = _flat(spec, config_index(traj, V, spec.dimension))
    out = _flat(spec, traj)
    counts = np.zeros((B, spec.num_configs, V))
    q = np.full((B, T, C, V), np.nan)
    b = np.arange(B)
    for t in range(1, T):
        for c in range(C):
            k = cfg[:, t - 1, c]
            N = counts[b, k]
            q[:, t, c] = (N + 1.0) / (N.sum(-1, keepdims=True) + V)
            counts[b, k, out[:, t, c]] += 1.0
    return q.reshape(traj.shape + (V,))


def kgram_rollout(spec, context: np.ndarray, horizon: int,
                  rng: np.random.Generator) -> tuple:
    """Stream B: roll out by sampling from the causal k-gram itself.

    context (B, m, *grid) -> (states (B, m+horizon, *grid), q (same + V))."""
    B, m = context.shape[:2]
    V, C = spec.V, spec.cells_per_frame
    n = m + horizon
    states = np.zeros((B, n, C), dtype=np.int64)
    states[:, :m] = _flat(spec, context)
    q_all = np.full((B, n, C, V), np.nan)
    counts = np.zeros((B, spec.num_configs, V))
    b = np.arange(B)
    grid = (B,) + tuple(spec.grid_shape)

    def cfg_of(frame):
        return config_index(frame.reshape(grid), V, spec.dimension).reshape(B, C)

    for t in range(1, n):
        k_row = cfg_of(states[:, t - 1])
        for c in range(C):
            k = k_row[:, c]
            if t >= m:
                N = counts[b, k]
                q = (N + 1.0) / (N.sum(-1, keepdims=True) + V)
                q_all[:, t, c] = q
                u = rng.random(B)[:, None]
                states[:, t, c] = (u >= np.cumsum(q, -1)[:, :-1]).sum(-1)
            counts[b, k, states[:, t, c]] += 1.0
    shape = (B, n) + tuple(spec.grid_shape)
    return states.reshape(shape), q_all.reshape(shape + (V,))


def _true_probs(spec, traj: np.ndarray, tables: np.ndarray) -> np.ndarray:
    """p_true for every cell of frames 1..T-1 given each trajectory's own history."""
    B, T = traj.shape[:2]
    cfg = _flat(spec, config_index(traj, spec.V, spec.dimension))   # (B, T, C)
    p = np.full((B, T, spec.cells_per_frame, spec.V), np.nan)
    p[:, 1:] = tables[np.arange(B)[:, None, None], cfg[:, :-1]]
    return p.reshape(traj.shape + (spec.V,))


def deterministic_metrics(spec, traj, tf_states=None, ar_states=None) -> dict:
    M = spec.M
    out = {}
    gt = _flat(spec, traj)[:, M:]
    if tf_states is not None:
        ok = _flat(spec, np.asarray(tf_states).reshape(traj.shape))[:, M:] == gt
        out["tf/cell_acc"] = float(ok.mean())
        out["tf/seq_acc"] = float(ok.all(axis=(1, 2)).mean())
    if ar_states is not None:
        ok = _flat(spec, np.asarray(ar_states).reshape(traj.shape))[:, M:] == gt
        out["ar/cell_acc"] = float(ok.mean())
        out["ar/seq_acc"] = float(ok.all(axis=(1, 2)).mean())
        out["ar/cell_acc_by_step"] = {int(M + i): float(a)
                                      for i, a in enumerate(ok.mean(axis=(0, 2)))}
    out["n_trajectories"] = int(traj.shape[0])
    return out


def stochastic_metrics(spec, traj, tf_probs=None, ar_states=None, ar_probs=None,
                       true_tables: Optional[np.ndarray] = None, seed: int = 0) -> dict:
    M, V = spec.M, spec.V
    B = traj.shape[0]
    out = {}

    def scored(x):   # (B, T, *grid, V?) -> frames M.. flattened over cells
        return _flat(spec, np.asarray(x).reshape(traj.shape + np.asarray(x).shape[traj.ndim:]))[:, M:]

    gt = scored(traj)
    if tf_probs is not None:
        q_model = scored(tf_probs)
        q_kgram = scored(kgram_probs(spec, traj))
        out["tf/kl_kgram_model"] = float(kl(q_kgram, q_model).mean())
        if true_tables is not None:
            p_true = scored(_true_probs(spec, traj, true_tables))
            out["tf/kl_true_model"] = float(kl(p_true, q_model).mean())
            out["tf/kl_true_kgram"] = float(kl(p_true, q_kgram).mean())
        out["tf/cell_acc"] = float((q_model.argmax(-1) == gt).mean())
        out["tf/kgram_cell_acc"] = float((q_kgram.argmax(-1) == gt).mean())

    if ar_states is not None:
        ar_states = np.asarray(ar_states).reshape(traj.shape)
        q_model = scored(ar_probs)
        q_kgram = scored(kgram_probs(spec, ar_states))   # counts follow the model's own samples
        out["ar/kl_kgram_model"] = float(kl(q_kgram, q_model).mean())
        out["ar/cell_acc"] = float((scored(ar_states) == gt).mean())
        b_states, b_q = kgram_rollout(spec, traj[:, :M], spec.T - M,
                                      np.random.default_rng(seed + 1))
        out["ar/kgram_cell_acc"] = float((scored(b_states) == gt).mean())
        if true_tables is not None:
            out["ar/kl_true_model"] = float(
                kl(scored(_true_probs(spec, ar_states, true_tables)), q_model).mean())
            out["ar/kl_true_kgram"] = float(
                kl(scored(_true_probs(spec, b_states, true_tables)), scored(b_q)).mean())
    out["n_trajectories"] = int(B)
    return out
