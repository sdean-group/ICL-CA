"""Plots for examples, predictions and rollouts.

One color convention everywhere: each state v has a color, and a probability
vector is drawn as the probability-weighted mix of those colors. A one-hot
(deterministic) prediction is therefore exactly the state color, and for V=2
a probability field reads as grayscale P(state=1).

1D trajectories are space-time diagrams (time runs downward); 2D trajectories
are rows of frames. A dashed line / gap marks the end of the M-frame context;
red boxes mark wrong cells in deterministic predictions.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .types import Example, ModelSpec, PredictionResult, RolloutResult

_STATE_COLORS = np.array([
    [0.96, 0.96, 0.95],   # 0  near-white
    [0.13, 0.13, 0.15],   # 1  near-black
    [0.85, 0.42, 0.10],   # 2  orange
    [0.17, 0.48, 0.71],   # 3  blue
])
_MISSING = np.array([0.85, 0.85, 0.85])
_ERROR = "#d62728"


def to_rgb(x: np.ndarray, V: int) -> np.ndarray:
    """States (...) int or probabilities (..., V) -> RGB (..., 3)."""
    colors = _STATE_COLORS[:V]
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        return colors[x]
    rgb = np.nan_to_num(x) @ colors
    rgb[np.isnan(x).any(-1)] = _MISSING
    return np.clip(rgb, 0, 1)


def _legend(ax_or_fig, V: int, probabilistic: bool, extra_error: bool) -> None:
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_STATE_COLORS[v], edgecolor="0.5", label=f"state {v}")
               for v in range(V)]
    if extra_error:
        handles.append(Patch(facecolor="none", edgecolor=_ERROR, label="wrong cell"))
    title = "colors mix by probability" if probabilistic else None
    ax_or_fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                     frameon=False, title=title, bbox_to_anchor=(0.5, 0.005))


def _mark_errors_1d(ax, pred, truth, start: int) -> None:
    from matplotlib.patches import Rectangle
    for t, c in zip(*np.nonzero(pred[start:] != truth[start:])):
        ax.add_patch(Rectangle((c - 0.5, t + start - 0.5), 1, 1, fill=False,
                               edgecolor=_ERROR, linewidth=1.4))


def _panel_1d(ax, rgb, title, M, n_frames):
    ax.imshow(rgb, interpolation="nearest", aspect="auto")
    ax.axhline(M - 0.5, color="#1f77b4", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks(range(0, n_frames, max(1, n_frames // 5)))
    ax.set_ylabel("time step" if ax.get_subplotspec().colspan.start == 0 else "")


def _panels(spec: ModelSpec, rows: List[Tuple[str, np.ndarray, Optional[np.ndarray]]],
            M: int, n_frames: int, max_frames: int, suptitle: str):
    """rows: (title, states-or-probs, truth for error marking or None)."""
    import matplotlib.pyplot as plt
    det_errors = any(r[2] is not None for r in rows)
    probabilistic = any(not np.issubdtype(np.asarray(r[1]).dtype, np.integer) for r in rows)
    if spec.dimension == 1:
        L = spec.grid_shape[0]
        fig, axes = plt.subplots(1, len(rows), figsize=(2.2 + 0.11 * L * len(rows), 0.35 * n_frames + 1.6),
                                 squeeze=False)
        for ax, (title, data, truth) in zip(axes[0], rows):
            _panel_1d(ax, to_rgb(data, spec.V), title, M, n_frames)
            if truth is not None:
                _mark_errors_1d(ax, data, truth, M)
    else:
        frames = list(range(max(M - 1, 0), min(n_frames, M - 1 + max_frames)))
        fig, axes = plt.subplots(len(rows), len(frames),
                                 figsize=(1.35 * len(frames) + 1.4, 1.45 * len(rows) + 0.9),
                                 squeeze=False)
        for i, (title, data, truth) in enumerate(rows):
            for j, t in enumerate(frames):
                ax = axes[i, j]
                ax.imshow(to_rgb(data[t], spec.V), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if truth is not None and t >= M:
                    _mark_errors_1d(ax, data[t], truth[t], 0)
                if t < M:
                    for s in ax.spines.values():
                        s.set_edgecolor("#1f77b4"); s.set_linewidth(2)
                if i == 0:
                    ax.set_title(f"t={t}" + (" (ctx)" if t < M else ""), fontsize=9)
                if j == 0:
                    ax.set_ylabel(title, fontsize=9)
    _legend(fig, spec.V, probabilistic, det_errors)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.1 if probabilistic else 0.07, 1, 1))
    return fig


def _rule_label(example: Example) -> str:
    if example.spec.stochastic:
        return f"random stochastic rule ({example.rule_split})"
    s = f"rule {example.rule} ({example.rule_split})"
    if example.spec.dimension == 1 and example.spec.V == 2:
        from .data.rules import wolfram_class
        cls = wolfram_class(example.rule)
        s += f", Wolfram class {cls}" if cls else ""
    return s


def plot_example(example: Example, max_frames: int = 6):
    spec = example.spec
    return _panels(spec, [("ground truth", example.trajectory.astype(np.int64), None)],
                   spec.M, spec.T, max_frames + 1,
                   f"{spec.key}: {_rule_label(example)}")


def plot_prediction(example: Example, result: PredictionResult, max_frames: int = 6):
    """Ground truth vs teacher-forced vs autoregressive for one Example."""
    if result.batched:
        raise ValueError("plot_prediction expects the result of predict() on a single Example")
    spec = example.spec
    gt = example.trajectory.astype(np.int64)
    rows = [("ground truth", gt, None)]
    if spec.stochastic:
        if example.rule is not None:
            from .metrics import _true_probs
            p_true = _true_probs(spec, gt[None], np.asarray(example.rule)[None])[0]
            p_true[0] = np.eye(spec.V)[gt[0]]
            rows.append(("true rule P(next)", p_true, None))
        if result.tf_probs is not None:
            tf = result.tf_probs.copy()
            tf[0] = np.eye(spec.V)[gt[0]]
            rows.append(("teacher-forced P(next)", tf, None))
        if result.ar_states is not None:
            rows.append(("AR sample", result.ar_states.astype(np.int64), None))
            ar = result.ar_probs.copy()
            ar[:spec.M] = np.eye(spec.V)[gt[:spec.M]]
            rows.append(("AR P(next)", ar, None))
    else:
        if result.tf_states is not None:
            rows.append(("teacher-forced", result.tf_states.astype(np.int64), gt))
        if result.ar_states is not None:
            rows.append(("autoregressive", result.ar_states.astype(np.int64), gt))
    m = result.metrics
    keys = (["tf/cell_acc", "ar/cell_acc"] if not spec.stochastic
            else ["tf/kl_true_model", "tf/kl_kgram_model", "ar/kl_true_model"])
    stats = "  ".join(f"{k}={m[k]:.3f}" for k in keys if k in m)
    return _panels(spec, rows, spec.M, spec.T, max_frames,
                   f"{spec.key}: {_rule_label(example)}\n{stats}")


def plot_rollout(result: RolloutResult, max_frames: int = 6):
    """Context + generated frames (and the reference, if one was given)."""
    spec = result.spec
    if result.states.ndim != 1 + len(spec.grid_shape):
        raise ValueError("plot_rollout expects a single (unbatched) rollout")
    n = result.states.shape[0]
    rows = []
    if result.reference is not None:
        rows.append(("reference", np.asarray(result.reference).astype(np.int64), None))
    truth = (np.asarray(result.reference).astype(np.int64)
             if result.reference is not None and not spec.stochastic else None)
    rows.append(("model rollout", result.states.astype(np.int64), truth))
    if spec.stochastic:
        p = result.probs.copy()
        p[:result.n_context] = np.eye(spec.V)[result.states[:result.n_context]]
        rows.append(("model P(next)", p, None))
    stats = "  ".join(f"{k}={v:.3f}" for k, v in result.metrics.items())
    title = f"{spec.key}: rollout from a {result.n_context}-frame context"
    if result.reference is None:
        title += "  (no reference given)"
    return _panels(spec, rows, result.n_context, n, max_frames,
                   title + (f"\n{stats}" if stats else ""))


def plot_ar_accuracy(result: PredictionResult):
    """Per-step AR cell accuracy (deterministic models)."""
    import matplotlib.pyplot as plt
    by_step = result.metrics.get("ar/cell_acc_by_step")
    if not by_step:
        raise ValueError("no ar/cell_acc_by_step in metrics (deterministic models, mode 'ar'/'both')")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ts, acc = zip(*sorted(by_step.items()))
    ax.plot(ts, acc, marker="o", color="#1f77b4")
    ax.set_xlabel("time step"); ax.set_ylabel("AR cell accuracy")
    ax.set_ylim(min(0.5, min(acc) - 0.05), 1.01)
    ax.set_xticks(ts)
    ax.grid(alpha=0.3)
    ax.set_title(f"{result.spec.key} (n={result.metrics['n_trajectories']})", fontsize=10)
    fig.tight_layout()
    return fig
