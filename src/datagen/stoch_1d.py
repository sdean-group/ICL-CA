"""1D V-class STOCHASTIC-rule CA data generator.

For each sequence, sample ONE stochastic rule table of shape (V**K, V) = (NUM_CONFIGS, V)
uniformly from the product of (V-1)-simplices:

    z_{c,v} ~ Exp(1),   p_{c,v} = z_{c,v} / sum_v' z_{c,v'}

(Equivalent to Dirichlet(1, ..., 1) per row.) The rule is fixed for the entire
trajectory. At every transition, the next cell state is drawn from

    x_i^{(t+1)} ~ Categorical(p_{cfg_idx, 0}, ..., p_{cfg_idx, V-1}).

Neighborhood configuration index is base-V:

    idx = (L*V + C)*V + R   in {0, ..., V**K - 1}

Train, val, and test rules are all sampled i.i.d. from the same uniform-simplex
distribution with separate fixed seeds. Since the rule space is continuous, there
is no notion of held-out rule pools.

Splits (masking follows the byk convention: train scores every position,
evals only score step >= M):

  train : uniform simplex  (no coverage filter, mask = all ones)
  val   : uniform simplex  (coverage-filtered on states[:, :M-1], mask = step>=M)
  test  : uniform simplex  (coverage-filtered on states[:, :M-1], mask = step>=M)

Default V=2, L=16, T=20, M=10 -> AR sequence length 338.

Usage (parallel):
    python -m src.datagen.stoch_1d --prepare \
        --V 2 --num_train 50000000 --num_eval 500000 --seed 42
    python -m src.datagen.stoch_1d --worker_id $W --num_workers 32 \
        --V 2 --output_dir <DIR from step 1>

Smoke test (single process):
    python -m src.datagen.stoch_1d --V 3 --num_train 100000 --num_eval 10000 \
        --num_eval_test 10000 --seed 42

Self-check tests (V in {2, 3, 4}):
    python -m src.datagen.stoch_1d --test
"""

import os
import json
import argparse

import numpy as np
from numpy.lib.format import open_memmap


# ============================================================
# Structural constants — 1D, K=3, offsets (-1,0,1), big-endian (L,C,R).
# V (alphabet size) is a runtime parameter; NUM_CONFIGS = V**K, TIME_SEP = V,
# VOCAB = V + 1 are derived per invocation.
# ============================================================
K = 3


def num_configs(V):
    return V ** K


def vocab_size(V):
    return V + 1


def time_sep_id(V):
    return V


# ============================================================
# Stochastic rule sampling
# ============================================================

def _dk_rule_table(V=2, p1=0.68, p2=0.88):
    """Domany-Kinzel rule table (V=2, K=3). Left+right neighbors decide next
    state via Bernoulli(q):
      (L=0, R=0) -> q=0
      (L=0, R=1) or (L=1, R=0) -> q=p1
      (L=1, R=1) -> q=p2
    Returns (num_configs, V) = (8, 2) float32."""
    assert V == 2, "DK rule is defined for V=2 only"
    tbl = np.zeros((num_configs(V), V), dtype=np.float32)
    for cfg in range(num_configs(V)):
        L = (cfg >> 2) & 1
        R = cfg & 1
        if L == 0 and R == 0:
            q = 0.0
        elif L == 1 and R == 1:
            q = float(p2)
        else:
            q = float(p1)
        tbl[cfg, 1] = q
        tbl[cfg, 0] = 1.0 - q
    return tbl


# Defensive DK-exclusion: Dirichlet(1,1) samples match the fixed DK table with
# probability 0, so this filter is a no-op in practice — but keeps the user
# safe from an astronomical fluke matching the OOD test distribution.
_DK_EXCLUDE_TABLE = _dk_rule_table(V=2, p1=0.68, p2=0.88)
_DK_EXCLUDE_ATOL = 1e-4


def _matches_dk(probs, atol=_DK_EXCLUDE_ATOL):
    """probs (B, 8, 2) -> bool array (B,) True where sample matches DK table."""
    if probs.shape[-2:] != _DK_EXCLUDE_TABLE.shape:
        return np.zeros(probs.shape[0], dtype=bool)
    return np.max(np.abs(probs - _DK_EXCLUDE_TABLE), axis=(-1, -2)) < atol


def sample_rules(B, V, rng):
    """Sample B rule tables of shape (B, V**K, V) uniformly from the product of
    (V-1)-simplices (Dirichlet(1, ..., 1) per row) via normalized Exp(1).

    For V=2 K=3, any (astronomically rare) sample that lands within
    `_DK_EXCLUDE_ATOL` of the DK(p1=0.68, p2=0.88) rule is rejected and
    redrawn — DK is the OOD test rule, and train/val must not see it."""
    z = rng.standard_exponential(size=(B, num_configs(V), V))
    z /= z.sum(axis=-1, keepdims=True)
    if V == 2:
        bad = _matches_dk(z)
        while bad.any():
            n_bad = int(bad.sum())
            zr = rng.standard_exponential(size=(n_bad, num_configs(V), V))
            zr /= zr.sum(axis=-1, keepdims=True)
            z[bad] = zr
            bad = _matches_dk(z)
    return z.astype(np.float32)


# ============================================================
# Vectorized 1D stochastic simulation (base-V ring)
# ============================================================

def _cfg_idx(g, V):
    """Neighborhood-config index per cell in base-V.
    g: (..., L). idx = (L*V + C)*V + R,  values in {0, ..., V**K - 1}."""
    Ln = np.roll(g, 1, axis=-1)
    C = g
    Rn = np.roll(g, -1, axis=-1)
    return (Ln * V + C) * V + Rn


def _categorical_from_cdf(cdf, rng, size):
    """Draw one categorical sample per cell via inverse-CDF over the last axis.
    cdf: (..., V) cumulative sums with cdf[..., -1] == 1.0 (snapped).
    size: shape of the uniform draws (must match cdf.shape[:-1]).
    Returns int8 array of the same shape, values in {0, ..., V-1}.
    Uses the identity  x = #{v : cdf[v] <= u}  which is safe because
    u ~ U[0, 1) < cdf[..., -1] = 1, so the count never overflows V-1."""
    u = rng.random_sample(size=size)
    return (u[..., None] >= cdf).sum(axis=-1).astype(np.int8)


def simulate_batch_stoch(rule_probs, num_steps, L, V, rng):
    """B trajectories, each under ITS OWN stochastic rule.
        rule_probs: (B, V**K, V) float, rows summing to 1
        returns states (B, num_steps, L) int8 in {0, ..., V-1}.
    The same rule_probs[b] is indexed at every t in this loop, so the rule stays
    fixed throughout the trajectory."""
    B = rule_probs.shape[0]
    assert rule_probs.shape == (B, num_configs(V), V), (
        f"rule_probs must be (B, V**K, V)=({B},{num_configs(V)},{V}); "
        f"got {rule_probs.shape}")
    states = np.empty((B, num_steps, L), dtype=np.int8)
    states[:, 0] = rng.randint(0, V, size=(B, L)).astype(np.int8)
    # Per-rule per-config CDF, snapped so the last entry is exactly 1.
    cdf = rule_probs.cumsum(axis=-1)                          # (B, V**K, V)
    cdf[..., -1] = 1.0
    bidx = np.arange(B)[:, None]
    for t in range(1, num_steps):
        idx = _cfg_idx(states[:, t - 1].astype(np.int64), V)  # (B, L)
        p_cdf = cdf[bidx, idx]                                # (B, L, V)
        states[:, t] = _categorical_from_cdf(p_cdf, rng, size=(B, L))
    return states


def coverage_batch(states, M, V):
    """Per-trajectory: do the first M-1 rows cover all V**K configs?"""
    B = states.shape[0]
    NC = num_configs(V)
    ctx = states[:, :M - 1].astype(np.int64)
    flat = _cfg_idx(ctx, V).reshape(B, -1)
    pres = np.zeros((B, NC), dtype=bool)
    pres[np.arange(B)[:, None], flat] = True
    return pres.all(axis=1)


# ============================================================
# Tokenization (1D: T rows of L cells, TIME_SEP between rows; no row-sep)
# ============================================================

def compute_seq_len(L, T):
    # (input length) = L*T + (T-1) - 1 = tokens_per_ar_seq
    return L * T + (T - 1) - 1


def build_token_template(L, T):
    cell_dst, timesep_dst, step_tpl = [], [], []
    pos = 0
    for t in range(T):
        for _c in range(L):
            cell_dst.append(pos); step_tpl.append(t); pos += 1
        if t < T - 1:
            timesep_dst.append(pos); step_tpl.append(t); pos += 1
    return (np.array(cell_dst, dtype=np.int64),
            np.array(timesep_dst, dtype=np.int64),
            np.array(step_tpl, dtype=np.int64))


def tokenize_batch(states, tpl, M, require_all_patterns, V):
    """states (B,T,L) int8 -> inp (B,seq) int8, tgt (B,seq) int8, mask_row (seq,) uint8.
    byk masking: require_all_patterns -> score only rows step>=M; else score all.
    Uses TIME_SEP = V as the inter-row separator token."""
    cell_dst, timesep_dst, step_tpl = tpl
    B = states.shape[0]
    total = len(step_tpl)
    out = np.empty((B, total), dtype=np.int8)
    out[:, cell_dst] = states.reshape(B, -1)
    if len(timesep_dst):
        out[:, timesep_dst] = time_sep_id(V)
    inp = out[:, :-1]
    tgt = out[:, 1:]
    target_steps = step_tpl[1:]
    if require_all_patterns:
        mask_row = (target_steps >= M).astype(np.uint8)
    else:
        mask_row = np.ones(total - 1, dtype=np.uint8)
    return inp, tgt, mask_row


# ============================================================
# Logger
# ============================================================

class Logger:
    def __init__(self, path=None, mode="w"):
        self.fh = open(path, mode) if path else None

    def log(self, msg=""):
        print(msg, flush=True)
        if self.fh:
            self.fh.write(str(msg) + "\n"); self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()


# ============================================================
# Phase — batched sample generation with memmap writing
# ============================================================

def generate_split(name, num_samples, out_dir, L, T, M, V, seed,
                   require_all_patterns, worker_tag, batch_size, tpl, logger):
    NC = num_configs(V)
    rng = np.random.RandomState(seed)
    seq_len = len(tpl[2]) - 1
    base = os.path.join(out_dir, f"{name}{worker_tag}")
    mm_inp = open_memmap(base + "_inputs.npy", mode="w+",
                         dtype=np.int8, shape=(num_samples, seq_len))
    mm_tgt = open_memmap(base + "_targets.npy", mode="w+",
                         dtype=np.int8, shape=(num_samples, seq_len))
    mm_mask = open_memmap(base + "_masks.npy", mode="w+",
                          dtype=np.uint8, shape=(num_samples, seq_len))
    mm_rules = open_memmap(base + "_rules.npy", mode="w+",
                           dtype=np.float32, shape=(num_samples, NC, V))

    collected = n_sim = 0
    log_every = 2_000_000 if num_samples >= 5_000_000 else 100_000
    next_log = log_every

    while collected < num_samples:
        B = batch_size
        probs = sample_rules(B, V, rng)                        # (B, V**K, V)
        states = simulate_batch_stoch(probs, T, L, V, rng)     # (B, T, L)
        n_sim += B
        if require_all_patterns:
            cov = coverage_batch(states, M, V)
            states = states[cov]
            probs = probs[cov]

        remaining = num_samples - collected
        if len(states) > remaining:
            states = states[:remaining]
            probs = probs[:remaining]
        nb = len(states)
        if nb == 0:
            continue

        inp, tgt, mask_row = tokenize_batch(states, tpl, M, require_all_patterns, V)
        s = slice(collected, collected + nb)
        mm_inp[s] = inp
        mm_tgt[s] = tgt
        mm_mask[s] = mask_row
        mm_rules[s] = probs
        collected += nb

        if collected >= next_log:
            logger.log(f"    [{name}{worker_tag}] {collected}/{num_samples}  "
                       f"accept={collected/max(n_sim,1):.3f}")
            next_log += log_every

    for m in (mm_inp, mm_tgt, mm_mask, mm_rules):
        m.flush()
    del mm_inp, mm_tgt, mm_mask, mm_rules

    logger.log(f"  [{name}{worker_tag}] DONE {collected} -> {name}{worker_tag}_*.npy  "
               f"seq_len={seq_len}  raq={require_all_patterns}  "
               f"accept_rate={collected/max(n_sim,1):.3f}")


# ============================================================
# Orchestration
# ============================================================

def count_str(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def resolve_output_dir(args):
    if args.output_dir:
        return args.output_dir
    return os.path.join(
        args.output_root,
        f"V{args.V}_stoch_L{args.L}_T{args.T}_M{args.M}"
        f"_n{count_str(args.num_train)}_val{count_str(args.num_eval)}"
        f"_test{count_str(args.num_eval_test)}_seed{args.seed}")


def split_specs(args):
    """List of (name, num_samples, seed, require_all_patterns)."""
    return [
        ("train", args.num_train,     args.seed,        False),
        ("val",   args.num_eval,      args.seed + 50,   True),
        ("test",  args.num_eval_test, args.seed + 100,  True),
    ]


def worker_share(total, w, n):
    base, rem = divmod(total, n)
    return base + (1 if w < rem else 0)


def do_prepare(args, logger_path_override=None):
    V = args.V
    NC = num_configs(V)
    print(f"=== PREPARE: stochastic-rule 1D CA (V={V}; per-seq Dirichlet(1,...,1)) ===")
    out_dir = resolve_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(logger_path_override or os.path.join(out_dir, "stats.txt"))
    logger.log("=== CONFIG ===")
    logger.log(f"seed={args.seed}  L={args.L}  T={args.T}  M={args.M}  V={V}  K={K}  offsets=(-1,0,1)")
    logger.log(f"seq_len={compute_seq_len(args.L, args.T)}  "
               f"vocab={vocab_size(V)}  time_sep_id={time_sep_id(V)}")
    logger.log(f"samples: train={args.num_train}, val={args.num_eval}, test={args.num_eval_test}; "
               f"batch_size={args.batch_size}")
    logger.log(f"rule_distribution: uniform simplex per row (Dirichlet(1,...,1) with {V} entries); "
               f"rule shape=({NC},{V}); one rule per sequence, fixed for T steps.")
    logger.log(f"filtering: train unfiltered (mask all-1); "
               f"val+test coverage-filtered on states[:, :M-1] over all {NC} configs (mask step>=M).")
    logger.log(f"output_dir={out_dir}\n")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"V": V, "K": K, "L": args.L, "T": args.T, "M": args.M,
                   "num_train": args.num_train, "num_val": args.num_eval,
                   "num_test": args.num_eval_test, "seed": args.seed,
                   "vocab": vocab_size(V), "time_sep_id": time_sep_id(V),
                   "seq_len": compute_seq_len(args.L, args.T),
                   "rule_shape": [NC, V],
                   "rule_distribution": f"uniform_simplex_dirichlet_1_x{V}",
                   "one_rule_per_sequence": True}, f, indent=2)
    logger.close()
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--V", type=int, default=2, help="alphabet size (>=2)")
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--M", type=int, default=10)
    p.add_argument("--num_train", type=int, default=50_000_000)
    p.add_argument("--num_eval", type=int, default=500_000, help="val split size")
    p.add_argument("--num_eval_test", type=int, default=None,
                   help="test split size (default: same as --num_eval)")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", type=str, default="data")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--prepare", action="store_true",
                   help="write meta/config only, print OUTPUT_DIR")
    p.add_argument("--worker_id", type=int, default=None,
                   help="this worker's index (0..num_workers-1)")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--test", action="store_true", help="run self-check tests and exit")
    args = p.parse_args()

    if args.test:
        _run_tests()
        return

    if args.V < 2:
        raise SystemExit("--V must be >= 2")
    if args.num_eval_test is None:
        args.num_eval_test = args.num_eval

    tpl = build_token_template(args.L, args.T)

    if args.prepare:
        out_dir = do_prepare(args)
        print(f"OUTPUT_DIR={out_dir}")
        return

    if args.worker_id is not None:
        if not args.output_dir:
            raise SystemExit("worker mode requires --output_dir (from `--prepare`)")
        W, N = args.worker_id, args.num_workers
        logger = Logger(os.path.join(args.output_dir, f"stats_w{W}.txt"))
        logger.log(f"=== WORKER {W}/{N}  V={args.V}  dir={args.output_dir} ===")
        for name, n_total, sd, raq in split_specs(args):
            n = worker_share(n_total, W, N)
            if n == 0:
                continue
            logger.log(f"\n-- {name} worker {W}: {n}/{n_total} samples --")
            generate_split(name, n, args.output_dir, args.L, args.T, args.M, args.V,
                           sd + W * 100003, raq, f"_w{W}", args.batch_size, tpl, logger)
        logger.log(f"\n=== WORKER {W} DONE ===")
        logger.close()
        return

    # single-process smoke
    out_dir = do_prepare(args)
    logger = Logger(os.path.join(out_dir, "stats.txt"), mode="a")
    logger.log("\n=== SAMPLE GENERATION (single process) ===")
    for name, n_total, sd, raq in split_specs(args):
        logger.log(f"\n-- {name}: {n_total} samples --")
        generate_split(name, n_total, out_dir, args.L, args.T, args.M, args.V,
                       sd, raq, "", args.batch_size, tpl, logger)
    logger.log(f"\n=== DONE === output_dir={out_dir}")
    logger.close()


# ============================================================
# Self-check tests (invoked with `--test` or directly).
# All V-parameterized tests are run for V in {2, 3, 4}.
# ============================================================

def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _test_rule_sampling(V):
    """Every sampled rule row is nonneg and sums to 1."""
    rng = np.random.RandomState(0)
    r = sample_rules(5000, V, rng)
    NC = num_configs(V)
    _assert(r.shape == (5000, NC, V),
            f"[V={V}] rule shape wrong: {r.shape} vs expected (5000,{NC},{V})")
    _assert(np.all(r >= 0.0), f"[V={V}] rule contains negative probabilities")
    _assert(np.all(r <= 1.0), f"[V={V}] rule contains probabilities > 1")
    sums = r.sum(axis=-1)                                     # (B, NC)
    _assert(np.allclose(sums, 1.0, atol=1e-5),
            f"[V={V}] rule rows do not sum to 1: max|sum-1|={np.max(np.abs(sums-1)):.2e}")


def _test_rules_are_independent(V):
    """Different sequences receive independently sampled rules."""
    rng = np.random.RandomState(1)
    r = sample_rules(1000, V, rng)                            # (1000, NC, V)
    flat = r.reshape(1000, -1)
    uniq = np.unique(flat, axis=0)
    _assert(uniq.shape[0] == 1000,
            f"[V={V}] only {uniq.shape[0]}/1000 unique rules")
    # Uniform simplex has row-mean = 1/V; check with a loose tolerance.
    mean = r.mean(axis=0)                                     # (NC, V)
    _assert(np.all(np.abs(mean - 1.0 / V) < 0.05),
            f"[V={V}] row-mean drift too large: max dev "
            f"{np.max(np.abs(mean-1.0/V)):.3f} (target 1/V={1.0/V:.3f})")


def _test_states_in_range(V):
    """Every generated state is in {0, ..., V-1}."""
    rng = np.random.RandomState(4)
    probs = sample_rules(64, V, rng)
    states = simulate_batch_stoch(probs, 20, 32, V, rng)
    _assert(states.min() >= 0 and states.max() <= V - 1,
            f"[V={V}] states out of range: [{states.min()}, {states.max()}]")


def _test_rule_fixed_throughout_sequence(V):
    """One-hot categorical rule -> exact deterministic transitions.
    Constructs, per sequence, a rule where every config row is a one-hot vector
    (arbitrary deterministic next-state table). The stochastic simulator must
    reproduce the corresponding deterministic base-V CA byte-for-byte."""
    rng = np.random.RandomState(2)
    B, T, L = 32, 20, 16
    NC = num_configs(V)
    # Deterministic per-config next-state table in {0, ..., V-1}.
    table = rng.randint(0, V, size=(B, NC)).astype(np.int8)   # (B, NC)
    probs = np.zeros((B, NC, V), dtype=np.float32)
    probs[np.arange(B)[:, None], np.arange(NC)[None, :], table] = 1.0
    # Rows are one-hot -> nonneg and sum to 1 (sanity).
    _assert(np.allclose(probs.sum(-1), 1.0),
            f"[V={V}] one-hot rule rows do not sum to 1")

    sim_rng = np.random.RandomState(42)
    states = simulate_batch_stoch(probs, T, L, V, sim_rng)    # (B, T, L)

    ref = np.empty_like(states)
    ref[:, 0] = states[:, 0]
    bidx = np.arange(B)[:, None]
    for t in range(1, T):
        idx = _cfg_idx(ref[:, t - 1].astype(np.int64), V)
        ref[:, t] = table[bidx, idx]
    _assert(np.array_equal(states, ref),
            f"[V={V}] one-hot rule stochastic sim did not match deterministic reference")


def _test_categorical_frequencies(V):
    """Empirical marginal frequencies of sampled states match specified probs.
    Build a rule whose rows are ALL equal to a single categorical p; then every
    transition samples from Categorical(p) regardless of prior state, so the
    empirical distribution over states[:, 1:] should match p."""
    rng = np.random.RandomState(5)
    NC = num_configs(V)
    # Fixed target distribution p (skewed but nondegenerate).
    p = rng.standard_exponential(size=V)
    p = (p / p.sum()).astype(np.float32)                      # (V,)
    B, T, L = 256, 10, 16
    probs = np.broadcast_to(p, (B, NC, V)).astype(np.float32).copy()
    states = simulate_batch_stoch(probs, T, L, V, rng)        # (B, T, L)
    # Aggregate over transitions (skip the uniform-random initial row).
    counts = np.bincount(states[:, 1:].ravel(), minlength=V).astype(np.float64)
    freq = counts / counts.sum()
    tol = 0.02
    _assert(np.max(np.abs(freq - p)) < tol,
            f"[V={V}] empirical frequency {freq} deviates from target {p} by "
            f">{tol}: max|freq-p|={np.max(np.abs(freq-p)):.3f}")


def _test_reproducibility(V):
    """Same seed => byte-identical rules and trajectories."""
    r1 = sample_rules(64, V, np.random.RandomState(7))
    r2 = sample_rules(64, V, np.random.RandomState(7))
    _assert(np.array_equal(r1, r2),
            f"[V={V}] sample_rules not reproducible for same seed")
    s1 = simulate_batch_stoch(r1, 20, 16, V, np.random.RandomState(11))
    s2 = simulate_batch_stoch(r2, 20, 16, V, np.random.RandomState(11))
    _assert(np.array_equal(s1, s2),
            f"[V={V}] simulate_batch_stoch not reproducible for same probs+seed")


def _test_coverage_filter(V):
    """Filtered eval sequences satisfy the V**K-config coverage requirement on
    states[:, :M-1]. Grid sizes scale with V so coupon-collector expectations
    give a nonzero acceptance rate."""
    NC = num_configs(V)
    # Coupon-collector expectation is NC * H_{NC}; pick (M-1)*L generously above it.
    sizes = {2: (16, 10, 20), 3: (32, 15, 20), 4: (48, 20, 25)}
    L, M, T = sizes[V]
    rng = np.random.RandomState(3)
    probs = sample_rules(2048, V, rng)
    states = simulate_batch_stoch(probs, T, L, V, rng)
    cov = coverage_batch(states, M, V)
    kept = states[cov]
    _assert(kept.shape[0] > 0,
            f"[V={V}] no eval samples passed coverage on (L={L},M={M})")
    ctx = kept[:, :M - 1].astype(np.int64)
    idxs = _cfg_idx(ctx, V).reshape(kept.shape[0], -1)
    pres = np.zeros((kept.shape[0], NC), dtype=bool)
    pres[np.arange(kept.shape[0])[:, None], idxs] = True
    missing = (~pres.all(axis=1)).sum()
    _assert(missing == 0,
            f"[V={V}] {missing}/{kept.shape[0]} filtered samples miss >=1 of {NC} configs")
    # Also confirm coverage_batch flags a genuinely-missing state.
    bad = kept[:1].copy()
    bad[:, :M - 1] = 0                                        # only config 0 present
    _assert(not coverage_batch(bad, M, V)[0],
            f"[V={V}] coverage_batch failed to reject all-zero context")
    print(f"    [V={V}] {kept.shape[0]}/2048 kept, all cover {NC} configs on states[:, :M-1]")


def _test_end_to_end_smoke(V):
    """Tiny smoke: shapes, ranges, masks, rules; base-V TIME_SEP token."""
    import tempfile
    L, T, M = 16, 20, 10
    NC = num_configs(V)
    SEP = time_sep_id(V)
    with tempfile.TemporaryDirectory() as tmp:
        tpl = build_token_template(L, T)
        seq_len = compute_seq_len(L, T)
        _assert(seq_len == L * T + (T - 1) - 1, "seq_len formula wrong")
        _assert(seq_len == 338, f"expected seq_len 338 at L=16,T=20; got {seq_len}")
        logger = Logger(None)
        generate_split("train", 200, tmp, L, T, M, V, seed=0,
                       require_all_patterns=False, worker_tag="",
                       batch_size=128, tpl=tpl, logger=logger)
        # Small V >= 3 makes 27+ configs harder to cover in a tiny batch;
        # only run the val (filtered) leg for V=2 where sizes give a decent
        # accept rate. Coverage correctness itself is tested separately.
        if V == 2:
            generate_split("val", 200, tmp, L, T, M, V, seed=1,
                           require_all_patterns=True, worker_tag="",
                           batch_size=128, tpl=tpl, logger=logger)
            splits = (("train", False), ("val", True))
        else:
            splits = (("train", False),)
        for split, req in splits:
            inp = np.load(os.path.join(tmp, f"{split}_inputs.npy"))
            tgt = np.load(os.path.join(tmp, f"{split}_targets.npy"))
            msk = np.load(os.path.join(tmp, f"{split}_masks.npy"))
            rls = np.load(os.path.join(tmp, f"{split}_rules.npy"))
            _assert(inp.shape == (200, seq_len), f"[V={V}] {split} inp shape {inp.shape}")
            _assert(tgt.shape == (200, seq_len), f"[V={V}] {split} tgt shape {tgt.shape}")
            _assert(msk.shape == (200, seq_len), f"[V={V}] {split} msk shape {msk.shape}")
            _assert(rls.shape == (200, NC, V), f"[V={V}] {split} rules shape {rls.shape}")
            _assert(inp.min() >= 0 and inp.max() <= SEP,
                    f"[V={V}] {split} inp out of range: [{inp.min()},{inp.max()}]")
            _assert(tgt.min() >= 0 and tgt.max() <= SEP,
                    f"[V={V}] {split} tgt out of range: [{tgt.min()},{tgt.max()}]")
            _assert(np.all(np.isin(msk, [0, 1])), f"[V={V}] {split} mask not binary")
            _assert(np.allclose(rls.sum(-1), 1.0, atol=1e-5),
                    f"[V={V}] {split} rules rows do not sum to 1")
            if req:
                _, _, step_tpl = tpl
                target_steps = step_tpl[1:]
                expected = (target_steps >= M).astype(np.uint8)
                _assert(np.array_equal(msk[0], expected),
                        f"[V={V}] {split} mask does not match step>=M")
                _assert(msk.sum() > 0, f"[V={V}] {split} mask all zeros")
            else:
                _assert(msk.sum() == 200 * seq_len,
                        f"[V={V}] {split} train mask should be all-ones")


def _run_tests():
    print("=== SELF-CHECK TESTS ===")
    for V in (2, 3, 4):
        print(f"\n-- V = {V} (NUM_CONFIGS = V**K = {num_configs(V)}) --")
        print(f"[1] rule sampling nonnegative + rows sum to 1  (V={V})")
        _test_rule_sampling(V)
        print(f"[2] rules independent across sequences         (V={V})")
        _test_rules_are_independent(V)
        print(f"[3] generated states in {{0, ..., V-1}}         (V={V})")
        _test_states_in_range(V)
        print(f"[4] one-hot rule -> exact deterministic sim    (V={V})")
        _test_rule_fixed_throughout_sequence(V)
        print(f"[5] empirical categorical freq matches target  (V={V})")
        _test_categorical_frequencies(V)
        print(f"[6] reproducibility for same seed              (V={V})")
        _test_reproducibility(V)
        print(f"[7] coverage over all V**K configs on ctx      (V={V})")
        _test_coverage_filter(V)
        print(f"[8] end-to-end smoke: shapes/ranges/masks      (V={V})")
        _test_end_to_end_smoke(V)
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
