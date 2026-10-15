"""Trajectory <-> token sequence.

Layout (matches the generators' build_token_template):
    for t in 0..T-1:
        for r in rows:  cells of row r, then ROW_SEP if 2D and r < L1-1
        TIME_SEP if t < T-1
The model input is tokens[:-1]; the logit at position j predicts token j+1.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np

CELL, ROW_SEP, TIME_SEP = 0, 1, 2   # position kinds


@lru_cache(maxsize=None)
def _layout(dimension: int, grid_shape: Tuple[int, ...], T: int):
    """Returns (kind, frame) arrays over the full T-frame token sequence."""
    kind, frame = [], []
    rows, cols = (1, grid_shape[0]) if dimension == 1 else grid_shape
    for t in range(T):
        for r in range(rows):
            kind += [CELL] * cols
            frame += [t] * cols
            if dimension == 2 and r < rows - 1:
                kind.append(ROW_SEP)
                frame.append(t)
        if t < T - 1:
            kind.append(TIME_SEP)
            frame.append(t)
    return np.array(kind, dtype=np.int8), np.array(frame, dtype=np.int64)


def layout(spec, T: int = None):
    return _layout(spec.dimension, tuple(spec.grid_shape), T or spec.T)


def full_length(spec, T: int = None) -> int:
    return len(layout(spec, T)[0])


def cell_positions(spec, T: int = None) -> np.ndarray:
    """Positions of the cell tokens, in frame-major / row-major order."""
    kind, _ = layout(spec, T)
    return np.nonzero(kind == CELL)[0]


def encode(spec, frames: np.ndarray, trailing_sep: bool = False) -> np.ndarray:
    """(B, n_frames, *grid) or (n_frames, *grid) -> token ids (..., n_tokens).

    With trailing_sep=True a TIME_SEP is appended after the last frame, i.e.
    the sequence ends right before the next frame's first cell (the prompt
    format for generation)."""
    frames = np.asarray(frames)
    squeeze = frames.ndim == 1 + len(spec.grid_shape)
    if squeeze:
        frames = frames[None]
    B, n = frames.shape[:2]
    kind, _ = layout(spec, n)
    out = np.empty((B, len(kind)), dtype=np.int64)
    out[:, kind == CELL] = frames.reshape(B, -1)
    out[:, kind == ROW_SEP] = spec.row_sep if spec.row_sep is not None else -1
    out[:, kind == TIME_SEP] = spec.time_sep
    if trailing_sep:
        out = np.concatenate([out, np.full((B, 1), spec.time_sep, dtype=np.int64)], axis=1)
    return out[0] if squeeze else out


def decode(spec, tokens: np.ndarray, n_frames: int) -> np.ndarray:
    """Inverse of `encode` for a full n_frames sequence."""
    tokens = np.asarray(tokens)
    kind, _ = layout(spec, n_frames)
    cells = tokens[..., : len(kind)][..., kind == CELL]
    return cells.reshape(tokens.shape[:-1] + (n_frames,) + tuple(spec.grid_shape))
