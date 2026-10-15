"""Rule encodings, held-out rule pools, and stochastic rule sampling.

Deterministic rule <-> table (matches the data generators):
    table[i] = output state for neighborhood config i
    rule_int = sum_i table[i] * V**i            (little-endian, base V)
For 1D V=2 this is exactly the Wolfram rule number.

Neighborhood config index (big-endian, base V):
    1D elementary : idx = (left*V + center)*V + right
    2D von Neumann: idx = ((((N*V + W)*V + C)*V + E)*V + S)

Stochastic rule: a (V**K, V) array whose row i is the next-state distribution
for config i. Test-time rules are drawn uniformly from the product of simplices
(Dirichlet(1,...,1) per row).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np

RuleLike = Union[int, np.integer, np.ndarray]


def rule_table(rule_int: int, V: int, K: int) -> np.ndarray:
    """Integer rule -> (V**K,) int8 output table."""
    n = V ** K
    r = int(rule_int)
    if r < 0 or r >= V ** n:
        raise ValueError(f"rule {r} out of range for V={V}, K={K} (need < {V}**{n})")
    t = np.empty(n, dtype=np.int8)
    for i in range(n):
        t[i] = r % V
        r //= V
    return t


def table_to_rule(table: np.ndarray, V: int) -> int:
    table = np.asarray(table, dtype=np.int64).ravel()
    return int(sum(int(v) * V ** i for i, v in enumerate(table)))


def random_stochastic_rule(V: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform draw from the product of simplices: (V**K, V) float64."""
    z = rng.standard_exponential(size=(V ** K, V))
    return z / z.sum(axis=-1, keepdims=True)


def validate_stochastic_rule(rule: np.ndarray, V: int, K: int) -> np.ndarray:
    rule = np.asarray(rule, dtype=np.float64)
    if rule.shape != (V ** K, V):
        raise ValueError(f"stochastic rule must have shape {(V ** K, V)}, got {rule.shape}")
    if (rule < 0).any() or not np.allclose(rule.sum(-1), 1.0, atol=1e-6):
        raise ValueError("stochastic rule rows must be non-negative and sum to 1")
    return rule


# ---------------------------------------------------------------------------
# Rule pools (the exact train/val/test rule splits used for the paper models)
# ---------------------------------------------------------------------------

POOL_SPLITS = ("train", "valin", "valout", "test")


@lru_cache(maxsize=None)
def _pools() -> dict:
    path = Path(__file__).resolve().parent.parent / "assets" / "rule_pools.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def rule_pool(pool: str, split: str) -> np.ndarray:
    """Rule integers of one split. pool in {1d_v2k3, 1d_v3k3, 2d_v2k5};
    split in {train, valin, valout, test}. valin is a subset of train (used for
    in-distribution validation); valout and test are disjoint from train."""
    key = f"{pool}_{split}"
    pools = _pools()
    if key not in pools:
        raise KeyError(f"no rule pool '{key}'; available: {sorted(pools)}")
    return pools[key]


def heldout_rules(pool: str, split: str = "test") -> np.ndarray:
    """Rules never seen in training (split = "test" or "valout")."""
    if split not in ("test", "valout"):
        raise ValueError("held-out splits are 'test' and 'valout'")
    return rule_pool(pool, split)


@lru_cache(maxsize=None)
def _pool_sets(pool: str) -> dict:
    return {s: set(rule_pool(pool, s).tolist()) for s in ("train", "valout", "test")}


def rule_split(pool: str, rule_int: int) -> str:
    """Which split a deterministic rule belonged to for this model family."""
    sets = _pool_sets(pool)
    if rule_int in sets["test"]:
        return "heldout_test"
    if rule_int in sets["valout"]:
        return "heldout_val"
    return "train" if rule_int in sets["train"] else "unused"


def rule_tables(rules: np.ndarray, V: int, K: int) -> np.ndarray:
    """Vectorized rule_table: (B,) rule ints -> (B, V**K) int8 tables."""
    rules = np.asarray(rules, dtype=np.int64)
    powers = V ** np.arange(V ** K, dtype=np.int64)
    return ((rules[:, None] // powers[None, :]) % V).astype(np.int8)


def fold_k(rules: np.ndarray, num_configs: int = 32) -> np.ndarray:
    """2D V=2 complement-equivalence class index: min(popcount, 32 - popcount)."""
    rules = np.asarray(rules, dtype=np.int64)
    pop = np.zeros(rules.shape, dtype=np.int64)
    for i in range(num_configs):
        pop += (rules >> i) & 1
    return np.minimum(pop, num_configs - pop)


class RuleSampler:
    """Samples rules from one pool split the way the dataset generators did:
    uniformly over the pool, or (2D valout/test) uniformly over fold-k classes
    first and then uniformly within the class ("k_balanced")."""

    def __init__(self, pool: str, split: str, k_balanced: bool = False):
        self.rules = rule_pool(pool, split)
        self.k_balanced = k_balanced
        if k_balanced:
            k = fold_k(self.rules)
            self.ks = np.unique(k)
            self.buckets = [self.rules[k == kk] for kk in self.ks]

    def __call__(self, n: int, rng) -> np.ndarray:
        if not self.k_balanced:
            return self.rules[rng.integers(len(self.rules), size=n)]
        which = rng.integers(len(self.buckets), size=n)
        out = np.empty(n, dtype=np.int64)
        for i, bucket in enumerate(self.buckets):
            m = which == i
            if m.any():
                out[m] = bucket[rng.integers(len(bucket), size=int(m.sum()))]
        return out


def uses_k_balanced(dimension: int, split: str) -> bool:
    """The 2D generator drew valout/test rules k-balanced; everything else uniform."""
    return dimension == 2 and split in ("valout", "test")


# ---------------------------------------------------------------------------
# Wolfram classes (1D, V=2) — handy labels for elementary CA examples
# ---------------------------------------------------------------------------

_WOLFRAM_CLASSES = {
    "I": [0, 8, 32, 40, 128, 136, 160, 168],
    "II": [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26, 27,
           28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57, 58, 62,
           72, 73, 74, 76, 77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142,
           152, 154, 156, 162, 164, 170, 172, 178, 184, 200, 204, 232],
    "III": [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150],
    "IV": [41, 54, 106, 110],
}


def _eca_orbit(r: int) -> set:
    """Reflection x complement orbit of an elementary CA rule."""
    t = rule_table(r, 2, 3)
    refl = np.array([t[((i & 1) << 2) | (i & 2) | (i >> 2)] for i in range(8)])
    comp = np.array([1 - t[7 - i] for i in range(8)])
    both = np.array([1 - refl[7 - i] for i in range(8)])
    return {r, table_to_rule(refl, 2), table_to_rule(comp, 2), table_to_rule(both, 2)}


@lru_cache(maxsize=None)
def _wolfram_lookup() -> dict:
    out = {}
    for cls, canon in _WOLFRAM_CLASSES.items():
        for c in canon:
            for r in _eca_orbit(c):
                out[r] = cls
    return out


def wolfram_class(rule_int: int) -> Optional[str]:
    """Wolfram behavioural class (I-IV) of an elementary CA rule, else None."""
    return _wolfram_lookup().get(int(rule_int))
