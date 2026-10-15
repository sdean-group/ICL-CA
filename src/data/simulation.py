"""Cellular-automaton simulation on periodic grids (1D ring / 2D torus).

Same update semantics as the training data generators (data_generate_1d*.py,
data_generate_2d_*.py): uniform random initial frame, periodic boundaries, the
config indexing documented in `rules.py`, and inverse-CDF categorical sampling
for stochastic rules. (Random streams use numpy's Generator, so seeds do not
reproduce specific training/test samples.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def config_index(grid: np.ndarray, V: int, dimension: int) -> np.ndarray:
    """Neighborhood-config index per cell. grid: (..., L) or (..., L1, L2)."""
    g = np.asarray(grid).astype(np.int64)
    if dimension == 1:
        left = np.roll(g, 1, axis=-1)
        right = np.roll(g, -1, axis=-1)
        return (left * V + g) * V + right
    n = np.roll(g, 1, axis=-2)
    w = np.roll(g, 1, axis=-1)
    e = np.roll(g, -1, axis=-1)
    s = np.roll(g, -1, axis=-2)
    return (((n * V + w) * V + g) * V + e) * V + s


def step(frame: np.ndarray, rule, V: int, dimension: int,
         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply one update. rule: (V**K,) int table or (V**K, V) probability table."""
    idx = config_index(frame, V, dimension)
    rule = np.asarray(rule)
    if rule.ndim == 1:
        return rule[idx].astype(np.int8)
    if rng is None:
        raise ValueError("stochastic step needs an rng")
    cdf = np.cumsum(rule, axis=-1)
    cdf[..., -1] = 1.0
    u = rng.random(size=idx.shape)
    return (u[..., None] >= cdf[idx]).sum(axis=-1).astype(np.int8)


def simulate(initial: np.ndarray, rule, num_frames: int, V: int, dimension: int,
             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Roll `initial` (one frame) forward to `num_frames` frames total."""
    initial = np.asarray(initial, dtype=np.int8)
    out = np.empty((num_frames,) + initial.shape, dtype=np.int8)
    out[0] = initial
    for t in range(1, num_frames):
        out[t] = step(out[t - 1], rule, V, dimension, rng)
    return out


def covers_all_configs(trajectory: np.ndarray, M: int, V: int, K: int,
                       dimension: int) -> bool:
    """Test-set filter: every one of the V**K neighborhood configs appears in
    frames 0..M-2, so each rule-table entry is observable from the context."""
    idx = config_index(trajectory[: M - 1], V, dimension)
    return np.unique(idx).size == V ** K


def simulate_batch(rules: np.ndarray, num_frames: int, grid_shape, V: int,
                   dimension: int, rng: np.random.Generator) -> np.ndarray:
    """B trajectories, each under its own rule, from uniform random initial frames.

    rules: (B, V**K) int tables (deterministic) or (B, V**K, V) probabilities.
    Returns (B, num_frames, *grid_shape) int8.
    """
    rules = np.asarray(rules)
    B = rules.shape[0]
    grid_shape = tuple(grid_shape)
    states = np.empty((B, num_frames) + grid_shape, dtype=np.int8)
    states[:, 0] = rng.integers(0, V, size=(B,) + grid_shape, dtype=np.int8)
    bidx = np.arange(B).reshape((B,) + (1,) * len(grid_shape))
    stochastic = rules.ndim == 3
    if stochastic:
        cdf = np.cumsum(rules, axis=-1)
        cdf[..., -1] = 1.0
    for t in range(1, num_frames):
        idx = config_index(states[:, t - 1], V, dimension)
        if stochastic:
            u = rng.random(size=idx.shape)
            states[:, t] = (u[..., None] >= cdf[bidx, idx]).sum(axis=-1)
        else:
            states[:, t] = rules[bidx, idx]
    return states


def coverage_batch(states: np.ndarray, M: int, V: int, K: int, dimension: int) -> np.ndarray:
    """(B,) bool: does frames 0..M-2 of each trajectory contain all V**K configs?"""
    B = states.shape[0]
    flat = config_index(states[:, : M - 1], V, dimension).reshape(B, -1)
    present = np.zeros((B, V ** K), dtype=bool)
    present[np.arange(B)[:, None], flat] = True
    return present.all(axis=1)
