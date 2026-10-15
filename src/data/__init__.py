"""Example generation: sample a rule, simulate a trajectory, package it."""

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np

from ..types import Example, ModelSpec
from . import rules as _rules
from .simulation import config_index, covers_all_configs, simulate


def _resolve_rule(spec: ModelSpec, rule, rng: np.random.Generator, split: str):
    """-> (rule_for_simulation, rule_as_returned, split_label, was_sampled)."""
    if spec.stochastic:
        if rule is None:
            r = _rules.random_stochastic_rule(spec.V, spec.K, rng)
            return r, r, "random", True
        r = _rules.validate_stochastic_rule(rule, spec.V, spec.K)
        return r, r, "user", False

    if rule is None:
        sampler = _rules.RuleSampler(spec.rule_pool, split,
                                     _rules.uses_k_balanced(spec.dimension, split))
        rule = int(sampler(1, rng)[0])
        sampled = True
    else:
        sampled = False
    if np.ndim(rule) == 0:
        rule_int = int(rule)
    else:
        table = np.asarray(rule)
        if table.shape != (spec.num_configs,) or table.min() < 0 or table.max() >= spec.V:
            raise ValueError(f"deterministic rule table must have shape "
                             f"({spec.num_configs},) with values in [0, {spec.V})")
        rule_int = _rules.table_to_rule(table, spec.V)
    table = _rules.rule_table(rule_int, spec.V, spec.K)
    return table, rule_int, _rules.rule_split(spec.rule_pool, rule_int), sampled


def generate_example(spec: ModelSpec, rule=None, seed: Optional[int] = 0,
                     split: str = "test", require_coverage: bool = True,
                     max_tries: int = 200) -> Example:
    """Simulate one ground-truth trajectory of T frames.

    rule:
        deterministic — None (sample from the held-out `split` pool, i.e. rules
            the model never saw in training), an int rule number (Wolfram
            number for 1D V=2), or a (V**K,) output table.
        stochastic    — None (uniform random transition table, the test
            distribution) or a (V**K, V) array of next-state probabilities.
    require_coverage:
        Like the test set, resample the initial frame until every neighborhood
        config appears in the first M-1 frames, so the whole rule is
        identifiable from the context.
    """
    rng = np.random.default_rng(seed)
    sim_rule, rule_out, label, sampled = _resolve_rule(spec, rule, rng, split)
    shape = tuple(spec.grid_shape)
    for attempt in range(max_tries):
        init = rng.integers(0, spec.V, size=shape).astype(np.int8)
        traj = simulate(init, sim_rule, spec.T, spec.V, spec.dimension, rng)
        if not require_coverage or covers_all_configs(traj, spec.M, spec.V, spec.K, spec.dimension):
            return Example(spec, traj, rule_out, seed, label)
        if sampled and attempt % 20 == 19:   # this rule rarely covers; draw another
            sim_rule, rule_out, label, _ = _resolve_rule(spec, None, rng, split)
    warnings.warn(f"no initial state covered all {spec.num_configs} configs in the "
                  f"first {spec.M - 1} frames after {max_tries} tries; returning an "
                  f"uncovered example (the rule is only partly identifiable).")
    return Example(spec, traj, rule_out, seed, label)


def generate_examples(spec: ModelSpec, n: int, seed: int = 0, **kwargs) -> List[Example]:
    """n independent examples (each with its own rule unless `rule` is given)."""
    seeds = np.random.SeedSequence(seed).generate_state(n)
    return [generate_example(spec, seed=int(s), **kwargs) for s in seeds]


def continue_trajectory(spec: ModelSpec, context: np.ndarray, rule, num_frames: int,
                        seed: Optional[int] = 0) -> np.ndarray:
    """Ground-truth continuation of a user context under a known rule.

    Returns (len(context) + num_frames, *grid). For a deterministic rule the
    context itself should be consistent with the rule (frame t+1 = f(frame t));
    a stochastic rule yields one random realization per seed.
    """
    context = np.asarray(context, dtype=np.int8)
    rng = np.random.default_rng(seed)
    sim_rule, *_ = _resolve_rule(spec, rule, rng, "test")
    tail = simulate(context[-1], sim_rule, num_frames + 1, spec.V, spec.dimension, rng)
    return np.concatenate([context, tail[1:]], axis=0)


def check_context(spec: ModelSpec, context: np.ndarray, rule) -> dict:
    """Does a user context fit `rule`, and does it show the whole rule?

    follows_rule  -- every transition in the context is possible under the rule
                     (deterministic: frame t+1 == f(frame t); stochastic: every
                     observed next state has nonzero probability).
    configs_seen  -- how many of the V**K neighborhood configs occur in frames
                     0..m-2, i.e. how many rule entries the model can observe.
                     The test set requires all of them.
    """
    ctx = np.asarray(context, dtype=np.int64)
    if ctx.ndim != 1 + len(spec.grid_shape) or tuple(ctx.shape[1:]) != tuple(spec.grid_shape):
        raise ValueError(f"context must have shape (m, {', '.join(map(str, spec.grid_shape))}), "
                         f"got {ctx.shape}")
    if ctx.min() < 0 or ctx.max() >= spec.V:
        raise ValueError(f"context values must be in [0, {spec.V})")
    if len(ctx) < 2:
        raise ValueError("context needs at least 2 frames")
    sim_rule, *_ = _resolve_rule(spec, rule, np.random.default_rng(0), "test")
    idx = config_index(ctx[:-1], spec.V, spec.dimension)
    nxt = ctx[1:]
    if spec.stochastic:
        bad = sim_rule[idx, nxt] <= 0
    else:
        bad = sim_rule[idx] != nxt
    return {"follows_rule": not bool(bad.any()), "violations": int(bad.sum()),
            "configs_seen": int(np.unique(idx).size), "num_configs": spec.num_configs}
