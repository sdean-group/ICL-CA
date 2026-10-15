"""On-the-fly datasets: no data files, everything is simulated when needed.

Mirrors the dataset generators' splits (data_generate_*.py `split_specs`):

    deterministic                 rules from        coverage filter   scored targets
      train                       train pool        no                all tokens
      val_in_nofilter             train pool        no                all tokens
      val_in_filter               valin pool        yes               frames >= M
      val_out                     valout pool       yes               frames >= M
      test                        test pool         yes               frames >= M
    stochastic (fresh Dirichlet(1,...,1) table per sequence)
      train                                         no                all tokens
      val, test                                     yes               frames >= M

2D valout/test rules are drawn fold-k balanced, as in data_generate_2d_byk.py.
Eval splits are generated once, deterministically from a seed, and held in
memory; the train split is an endless stream that never repeats a sample.
(V=2 stochastic training also skipped one measure-zero OOD rule; not replicated.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from . import rules as _rules
from . import serialization as ser
from .simulation import coverage_batch, simulate_batch

DET_SPLITS = {   # name -> (rule pool split, coverage filter, seed offset)
    "train": ("train", False, 0),
    "train_eval": ("train", False, 7),
    "val_in_nofilter": ("train", False, 50),
    "val_in_filter": ("valin", True, 60),
    "val_out": ("valout", True, 100),
    "test": ("test", True, 150),
}
STOCH_SPLITS = {
    "train": (None, False, 0),
    "train_eval": (None, False, 7),
    "val": (None, True, 50),
    "test": (None, True, 100),
}


def split_table(spec) -> dict:
    return STOCH_SPLITS if spec.stochastic else DET_SPLITS


def default_val_splits(spec) -> list:
    return ["val"] if spec.stochastic else ["val_in_nofilter", "val_in_filter", "val_out"]


def default_select_split(spec) -> str:
    return "val" if spec.stochastic else "val_out"


class _RuleSource:
    """Draws (simulation tables, stored rules) for one split."""

    def __init__(self, spec, split: str):
        if split not in split_table(spec):
            raise ValueError(f"unknown split '{split}' for {'stochastic' if spec.stochastic else 'deterministic'} "
                             f"tasks; choose from {sorted(split_table(spec))}")
        self.spec = spec
        pool_split, self.filtered, self.seed_offset = split_table(spec)[split]
        self.sampler = (None if spec.stochastic else
                        _rules.RuleSampler(spec.rule_pool, pool_split,
                                           _rules.uses_k_balanced(spec.dimension, pool_split)))

    def __call__(self, n: int, rng: np.random.Generator):
        s = self.spec
        if s.stochastic:
            z = rng.standard_exponential(size=(n, s.num_configs, s.V))
            probs = z / z.sum(-1, keepdims=True)
            return probs, probs.astype(np.float32)
        ids = self.sampler(n, rng)
        return _rules.rule_tables(ids, s.V, s.K), ids


def simulate_split(spec, split: str, n: int, rng: np.random.Generator, batch: int = 4096):
    """n trajectories of `split` -> (states (n, T, *grid) int8, rules)."""
    source = _RuleSource(spec, split)
    states_out, rules_out, have = [], [], 0
    while have < n:
        tables, stored = source(batch, rng)
        states = simulate_batch(tables, spec.T, spec.grid_shape, spec.V, spec.dimension, rng)
        if source.filtered:
            keep = coverage_batch(states, spec.M, spec.V, spec.K, spec.dimension)
            states, stored = states[keep], stored[keep]
        take = min(len(states), n - have)
        states_out.append(states[:take])
        rules_out.append(stored[:take])
        have += take
    return np.concatenate(states_out), np.concatenate(rules_out)


def target_mask(spec, filtered: bool) -> np.ndarray:
    """(seq_len,) loss/eval mask over targets: all ones, or targets in frames >= M."""
    _, frame = ser.layout(spec)
    tgt_frame = frame[1:]
    return (tgt_frame >= spec.M).astype(np.float32) if filtered else np.ones(len(tgt_frame), np.float32)


class EvalSplit(Dataset):
    """A fixed, in-memory eval split. Items are (X, y, mask) like the original
    on-disk datasets; `.rules` holds per-sample rule ids / tables, `.states`
    the trajectories."""

    def __init__(self, spec, split: str, n: int, seed: int = 42):
        source = _RuleSource(spec, split)
        rng = np.random.default_rng([int(seed), source.seed_offset])
        self.spec, self.split = spec, split
        self.states, self.rules = simulate_split(spec, split, n, rng)
        self.tokens = torch.from_numpy(ser.encode(spec, self.states).astype(np.int8))
        self.mask = torch.from_numpy(target_mask(spec, source.filtered))

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, i):
        t = self.tokens[i].long()
        return t[:-1], t[1:], self.mask


class OnTheFlyTrainDataset(IterableDataset):
    """Endless-stream train split. Each (seed, epoch, rank, worker) gets its own
    RNG, so samples never repeat across epochs, ranks or DataLoader workers.
    Yields (X, y); the training loop treats a 2-tuple as an all-ones mask,
    exactly like the original on-disk train shards."""

    def __init__(self, spec, samples_per_epoch: int, seed: int = 42, rank: int = 0,
                 world_size: int = 1, sim_batch: int = 256):
        super().__init__()
        self.spec = spec
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed, self.rank, self.world_size = int(seed), int(rank), int(world_size)
        self.sim_batch = int(sim_batch)
        self.epoch = 0
        self.source = _RuleSource(spec, "train")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def total_samples(self) -> int:
        return self.samples_per_epoch

    def rank_total_samples(self) -> int:
        base, rem = divmod(self.samples_per_epoch, self.world_size)
        return base + (1 if self.rank < rem else 0)

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        base, rem = divmod(self.rank_total_samples(), nw)
        remaining = base + (1 if wid < rem else 0)
        rng = np.random.default_rng([self.seed, 1_000 + self.epoch, self.rank, wid])
        s = self.spec
        while remaining > 0:
            B = min(self.sim_batch, remaining)
            tables, _ = self.source(B, rng)
            states = simulate_batch(tables, s.T, s.grid_shape, s.V, s.dimension, rng)
            tokens = torch.from_numpy(ser.encode(s, states))
            for i in range(B):
                yield tokens[i, :-1], tokens[i, 1:]
            remaining -= B
