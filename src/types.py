"""Plain dataclasses shared across the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to build inputs for, and interpret outputs of, one model.

    Shapes: a 1D trajectory is (T, L); a 2D trajectory is (T, L1, L2).
    """
    key: str
    repo_id: str
    dimension: int                 # 1 or 2
    stochastic: bool
    V: int                         # number of cell states
    K: int                         # neighborhood size (3 elementary, 5 von Neumann)
    grid_shape: Tuple[int, ...]    # (L,) or (L1, L2)
    T: int                         # frames per training sequence
    M: int                         # context frames given to the model
    neighborhood: str              # "elementary" | "von_neumann"
    architecture: str              # "shared" | "dedicated"
    d_model: int
    heads: Tuple[int, ...]
    mlp: Tuple[int, ...]
    time_sep: int
    row_sep: Optional[int]
    rule_pool: str                 # held-out rule pool name, or "dirichlet"
    train_data: str = ""

    @property
    def vocab_size(self) -> int:
        return self.V + (2 if self.dimension == 2 else 1)

    @property
    def num_configs(self) -> int:
        return self.V ** self.K

    @property
    def cells_per_frame(self) -> int:
        return int(np.prod(self.grid_shape))

    @property
    def seq_len(self) -> int:
        """Model input length for a full T-frame sequence (tokens[:-1])."""
        from .data.serialization import full_length
        return full_length(self) - 1

    def describe(self) -> str:
        task = (f"{self.dimension}D {'stoch' if self.stochastic else 'det'} "
                f"V={self.V} K={self.K}")
        grid = "x".join(str(s) for s in self.grid_shape)
        return (f"{self.key}: {task}, grid {grid}, T={self.T}, M={self.M}, "
                f"{self.architecture} (d_model={self.d_model}, heads={list(self.heads)})")


@dataclass
class Example:
    """One ground-truth trajectory.

    trajectory : (T, *grid_shape) int array, all frames.
    context    : first M frames (what the model is shown).
    rule       : int (deterministic) or (V**K, V) float array (stochastic).
    """
    spec: ModelSpec
    trajectory: np.ndarray
    rule: Any
    seed: Optional[int] = None
    rule_split: str = "unknown"    # "heldout_test" | "heldout_val" | "train" | "random" | ...

    @property
    def context(self) -> np.ndarray:
        return self.trajectory[: self.spec.M]

    @property
    def future(self) -> np.ndarray:
        return self.trajectory[self.spec.M:]


@dataclass
class PredictionResult:
    """Output of `predict` on a batch of B trajectories (B=1 for a single Example).

    tf_probs   : (B, T, *grid, V) teacher-forced next-state distribution for
                 every cell of frames 1..T-1 (frame 0 is NaN: never predicted).
    tf_states  : argmax of tf_probs (frame 0 copied from the input).
    ar_states  : (B, T, *grid) autoregressive rollout; frames < M are the context.
    ar_probs   : (B, T, *grid, V) predictive distribution at each AR step
                 (NaN for context frames).
    metrics    : dict of scalar metrics on the scored frames M..T-1.
    """
    spec: ModelSpec
    tf_logits: Optional[np.ndarray] = None
    tf_probs: Optional[np.ndarray] = None
    tf_states: Optional[np.ndarray] = None
    ar_states: Optional[np.ndarray] = None
    ar_probs: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)
    batched: bool = False

    def __repr__(self) -> str:
        parts = [f"PredictionResult({self.spec.key}"]
        for name in ("tf_probs", "ar_states"):
            arr = getattr(self, name)
            if arr is not None:
                parts.append(f"{name}={tuple(arr.shape)}")
        return ", ".join(parts) + ")"


@dataclass
class RolloutResult:
    """Output of `rollout` from a user-supplied context.

    states : (n_context + horizon, *grid) context followed by generated frames.
    probs  : (n_context + horizon, *grid, V) predictive distributions
             (NaN for context frames).
    reference : optional (same shape as states) ground truth, if a rule was given.
    """
    spec: ModelSpec
    states: np.ndarray
    probs: np.ndarray
    n_context: int
    reference: Optional[np.ndarray] = None
    metrics: dict = field(default_factory=dict)
