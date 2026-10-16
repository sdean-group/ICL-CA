"""Teacher-forced prediction and autoregressive rollout.

Conventions (identical to the paper evaluators):
  * A cell's predictive distribution is the softmax over the first V logits
    (separator logits dropped, i.e. conditioned on "a cell comes next").
  * Separators are never generated; they are inserted at known positions.
  * Deterministic models roll out greedily (argmax); stochastic models sample
    at temperature 1.

AR decoding uses a KV cache (`_CachedDecoder`), which is mathematically the
same as re-running the full causal forward pass at every step but O(n) instead
of O(n^2) forward passes. The two give the same results.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from .data import serialization as ser
from .data.simulation import config_index
from .metrics import deterministic_metrics, stochastic_metrics
from .pretrained import ModelBundle
from .types import Example, PredictionResult, RolloutResult


# ---------------------------------------------------------------------------
# KV-cached incremental decoding
# ---------------------------------------------------------------------------

class _CachedDecoder:
    """Incremental forward for CATransformer (absolute positional embeddings,
    standard multi-head attention)."""

    def __init__(self, model):
        self.model = model
        self.k: List[Optional[torch.Tensor]] = [None] * len(model.transformer_layers)
        self.v: List[Optional[torch.Tensor]] = [None] * len(model.transformer_layers)
        self.pos = 0

    @torch.inference_mode()
    def feed(self, tokens: torch.Tensor) -> torch.Tensor:
        """Append tokens (B, n); return logits (B, n, vocab) for those positions."""
        m = self.model
        B, n = tokens.shape
        if self.pos + n > m.seq_len:
            raise ValueError(f"sequence would exceed the model's maximum length {m.seq_len}")
        pos_ids = torch.arange(self.pos, self.pos + n, device=tokens.device)
        x = m.embedding(tokens) + m.positional_embedding(pos_ids)[None]
        total = self.pos + n
        mask = None
        if n > 1:   # causal mask among the new tokens; all cached keys are visible
            q_pos = pos_ids[:, None]
            k_pos = torch.arange(total, device=tokens.device)[None, :]
            mask = k_pos <= q_pos
        for i, layer in enumerate(m.transformer_layers):
            attn = layer.self_attn
            H = attn.num_heads
            hd = x.shape[-1] // H
            h = layer.norm1(x)
            q, k, v = F.linear(h, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
            q, k, v = (t.view(B, n, H, hd).transpose(1, 2) for t in (q, k, v))
            if self.k[i] is not None:
                k = torch.cat([self.k[i], k], dim=2)
                v = torch.cat([self.v[i], v], dim=2)
            self.k[i], self.v[i] = k, v
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            o = o.transpose(1, 2).reshape(B, n, H * hd)
            x = x + attn.out_proj(o)
            if layer.use_mlp:
                y = layer.activation(layer.linear1(layer.norm2(x)))
                if layer.mlp_depth > 2:
                    for mid in layer.mid_layers:
                        y = layer.activation(mid(y))
                x = x + layer.linear2(y)
        self.pos = total
        return m.fc(x)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def _as_trajectories(bundle: ModelBundle, data) -> tuple:
    """Example | list[Example] | array -> (trajectories (B,T,*grid), rules or None, batched)."""
    spec = bundle.spec
    if isinstance(data, Example):
        return data.trajectory[None], [data.rule], False
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], Example):
        return np.stack([e.trajectory for e in data]), [e.rule for e in data], True
    arr = np.asarray(data)
    if arr.ndim == 1 + len(spec.grid_shape):
        return arr[None], None, False
    return arr, None, True


def _check_frames(spec, frames: np.ndarray, what: str) -> None:
    if tuple(frames.shape[2:]) != tuple(spec.grid_shape):
        raise ValueError(f"{what}: frame shape {tuple(frames.shape[2:])} does not match the "
                         f"model's grid {tuple(spec.grid_shape)}")
    if frames.min() < 0 or frames.max() >= spec.V:
        raise ValueError(f"{what}: cell values must be in [0, {spec.V})")


def _cell_probs(logits: torch.Tensor, V: int) -> torch.Tensor:
    return torch.softmax(logits[..., :V].float(), dim=-1)


# ---------------------------------------------------------------------------
# Teacher forcing
# ---------------------------------------------------------------------------

@torch.inference_mode()
def teacher_forced(bundle: ModelBundle, trajectories: np.ndarray,
                   batch_size: int = 64):
    """One causal forward pass per trajectory on the ground-truth sequence.

    Returns (logits, probs), each (B, T, *grid, V); frame 0 is NaN (not predicted).
    """
    spec, model, dev = bundle.spec, bundle.model, bundle.device
    B, T = trajectories.shape[:2]
    cell_pos = ser.cell_positions(spec, T)
    tokens = torch.as_tensor(ser.encode(spec, trajectories), device=dev)
    n_cells = T * spec.cells_per_frame
    logits_out = np.full((B, n_cells, spec.V), np.nan, dtype=np.float32)
    probs_out = np.full((B, n_cells, spec.V), np.nan, dtype=np.float32)
    targets = torch.as_tensor(cell_pos[spec.cells_per_frame:] - 1, device=dev)  # frames 1..T-1
    for s in range(0, B, batch_size):
        lg = model(tokens[s:s + batch_size, :-1])[:, targets, :spec.V].float()
        logits_out[s:s + batch_size, spec.cells_per_frame:] = lg.cpu().numpy()
        probs_out[s:s + batch_size, spec.cells_per_frame:] = torch.softmax(lg, -1).cpu().numpy()
    shape = (B, T) + tuple(spec.grid_shape) + (spec.V,)
    logits, probs = logits_out.reshape(shape), probs_out.reshape(shape)
    return logits, probs


# ---------------------------------------------------------------------------
# Autoregressive rollout
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _rollout_batch(bundle: ModelBundle, context: np.ndarray, horizon: int,
                   sample: bool, generator: Optional[torch.Generator]):
    """context (B, m, *grid) -> states (B, m+horizon, *grid), probs (same + V)."""
    spec, dev = bundle.spec, bundle.device
    B, m = context.shape[:2]
    n_frames = m + horizon
    kind, _ = ser.layout(spec, n_frames)
    prompt = torch.as_tensor(ser.encode(spec, context, trailing_sep=True), device=dev)

    dec = _CachedDecoder(bundle.model)
    logits = dec.feed(prompt)[:, -1]
    gen_cells, gen_probs = [], []
    for pos in range(prompt.shape[1], len(kind)):
        if kind[pos] == ser.CELL:
            p = _cell_probs(logits, spec.V)
            if sample:   # sampled on CPU so a seed gives the same rollout on any device
                nxt = torch.multinomial(p.cpu(), 1, generator=generator)[:, 0].to(dev)
            else:
                nxt = p.argmax(-1)
            gen_cells.append(nxt)
            gen_probs.append(p)
        else:
            nxt = torch.full((B,), spec.row_sep if kind[pos] == ser.ROW_SEP else spec.time_sep,
                             device=dev, dtype=torch.long)
        if pos < len(kind) - 1:
            logits = dec.feed(nxt[:, None])[:, -1]

    grid = tuple(spec.grid_shape)
    states = np.concatenate(
        [context.astype(np.int64),
         torch.stack(gen_cells, 1).cpu().numpy().reshape((B, horizon) + grid)], axis=1)
    probs = np.full((B, n_frames) + grid + (spec.V,), np.nan, dtype=np.float32)
    probs[:, m:] = torch.stack(gen_probs, 1).cpu().numpy().reshape((B, horizon) + grid + (spec.V,))
    return states, probs


def _ar(bundle, context, horizon, sample, seed, batch_size):
    gen = None
    if sample:
        gen = torch.Generator().manual_seed(int(seed))
    outs = [_rollout_batch(bundle, context[s:s + batch_size], horizon, sample, gen)
            for s in range(0, context.shape[0], batch_size)]
    return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])


def rollout(bundle: ModelBundle, context: np.ndarray, horizon: Optional[int] = None,
            sample: Optional[bool] = None, seed: int = 0, reference: Optional[np.ndarray] = None,
            batch_size: int = 64) -> RolloutResult:
    """Generate future frames from a context the user provides.

    context : (m, *grid) int array (or (B, m, *grid)). The released models were
              trained with m = spec.M context frames; other m work but are
              off-distribution.
    horizon : frames to generate; default T - m (the most the positional
              table allows is T - m).
    sample  : default greedy for deterministic models, temperature-1 sampling
              for stochastic ones.
    reference : optional ground truth of shape (m + horizon, *grid), e.g. from
              `continue_trajectory`. Without it there is nothing to compare to.
    """
    spec = bundle.spec
    ctx = np.asarray(context)
    batched = ctx.ndim == 2 + len(spec.grid_shape)
    if not batched:
        ctx = ctx[None]
    _check_frames(spec, ctx, "context")
    m = ctx.shape[1]
    if not 1 <= m < spec.T:
        raise ValueError(f"context must have 1..{spec.T - 1} frames, got {m}")
    if horizon is None:
        horizon = spec.T - m
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if m + horizon > spec.T:
        raise ValueError(f"context ({m}) + horizon ({horizon}) exceeds the model's "
                         f"trained length T={spec.T}")
    if sample is None:
        sample = spec.stochastic
    states, probs = _ar(bundle, ctx, horizon, sample, seed, batch_size)
    result = RolloutResult(spec, states if batched else states[0],
                           probs if batched else probs[0], m)
    if reference is not None:
        ref = np.asarray(reference)
        result.reference = ref
        if ref.shape != result.states.shape:
            raise ValueError(f"reference shape {ref.shape} != rollout shape {result.states.shape}")
        t_axis = 1 if batched else 0
        gen = np.take(result.states, range(m, m + horizon), axis=t_axis)
        ref_f = np.take(ref, range(m, m + horizon), axis=t_axis)
        result.metrics["cell_acc_vs_reference"] = float((gen == ref_f).mean())
    return result


# ---------------------------------------------------------------------------
# predict = teacher forcing + rollout + metrics on ground-truth trajectories
# ---------------------------------------------------------------------------

def predict(bundle: ModelBundle, data: Union[Example, Sequence[Example], np.ndarray],
            mode: str = "both", seed: int = 0, batch_size: int = 64) -> PredictionResult:
    """Run the model on full ground-truth trajectories and score frames M..T-1.

    data : an Example, a list of Examples, or trajectories (B, T, *grid).
    mode : "tf" (teacher-forced), "ar" (autoregressive from the first M frames),
           or "both".
    Rule tables (from Examples) enable the KL-to-true-rule metrics for
    stochastic models.
    """
    if mode not in ("tf", "ar", "both"):
        raise ValueError("mode must be 'tf', 'ar' or 'both'")
    spec = bundle.spec
    traj, rules, batched = _as_trajectories(bundle, data)
    traj = traj.astype(np.int64)
    _check_frames(spec, traj, "trajectory")
    if traj.shape[1] != spec.T:
        raise ValueError(f"predict needs full trajectories of T={spec.T} frames; "
                         f"use rollout() for a bare context")

    res = PredictionResult(spec, batched=batched)
    if mode in ("tf", "both"):
        res.tf_logits, res.tf_probs = teacher_forced(bundle, traj, batch_size)
        res.tf_states = np.nan_to_num(res.tf_probs, nan=0.0).argmax(-1)
        res.tf_states[:, 0] = traj[:, 0]
    if mode in ("ar", "both"):
        res.ar_states, res.ar_probs = _ar(bundle, traj[:, :spec.M], spec.T - spec.M,
                                          spec.stochastic, seed, batch_size)

    if spec.stochastic:
        true_tables = None
        if rules is not None and all(r is not None for r in rules):
            true_tables = np.stack([np.asarray(r, dtype=np.float64) for r in rules])
        res.metrics = stochastic_metrics(spec, traj, res.tf_probs, res.ar_states,
                                         res.ar_probs, true_tables)
    else:
        res.metrics = deterministic_metrics(spec, traj, res.tf_states, res.ar_states)

    if not batched:
        for name in ("tf_logits", "tf_probs", "tf_states", "ar_states", "ar_probs"):
            arr = getattr(res, name)
            if arr is not None:
                setattr(res, name, arr[0])
    return res


def evaluate(bundle: ModelBundle, n: int = 128, seed: int = 0, mode: str = "both",
             batch_size: int = 64) -> dict:
    """Quick estimate of the paper metrics on n fresh held-out examples.

    For the full numbers (100k trajectories by default) use
    `python -m src.evaluate --model KEY`."""
    from .data import generate_examples
    examples = generate_examples(bundle.spec, n, seed=seed)
    return predict(bundle, examples, mode=mode, seed=seed, batch_size=batch_size).metrics
