"""2D Von Neumann (K=5) V-class STOCHASTIC-rule CA data generator.

For each sequence, sample ONE stochastic rule table of shape (V**K, V) = (32, V)
uniformly from the product of (V-1)-simplices:

    z_{c,v} ~ Exp(1),   p_{c,v} = z_{c,v} / sum_v' z_{c,v'}

(Equivalent to Dirichlet(1, ..., 1) per row.) The rule is fixed for the entire
trajectory. At every VN neighborhood transition, the next cell state is drawn from

    x_{r,c}^{(t+1)} ~ Categorical(p_{cfg_idx, 0}, ..., p_{cfg_idx, V-1}).

VN neighborhood index (K=5, big-endian N W C E S), same as data_generate_2d_byk.py:

    idx = ((((N*V + W)*V + C)*V + E)*V + S)   in {0, ..., V**K - 1}

Splits (masking follows the byk convention: train scores every position,
evals only score step >= M):

  train : uniform simplex   (no coverage filter, mask = all ones)
  val   : uniform simplex   (coverage-filtered on states[:, :M-1], mask = step>=M)
  test  : uniform simplex   (coverage-filtered on states[:, :M-1], mask = step>=M)

Default V=2, L1=L2=6, T=20, M=10 -> AR sequence length 838
(seq_len = T*(L1*L2 + L1-1) + (T-1) - 1 = 20*(36+5) + 19 - 1).

Usage (parallel):
    python -m src.datagen.stoch_2d --prepare \\
        --V 2 --L1 6 --L2 6 --T 20 --M 10 \\
        --num_train 50000000 --num_eval 500000 --seed 42
    python -m src.datagen.stoch_2d --worker_id $W --num_workers 32 \\
        --output_dir <DIR from step 1>

Smoke test (single process):
    python -m src.datagen.stoch_2d --V 2 --num_train 5000 --num_eval 500 \\
        --num_eval_test 500 --seed 42

Self-check tests (V in {2, 3}):
    python -m src.datagen.stoch_2d --test
"""

import os
import json
import argparse

import numpy as np
from numpy.lib.format import open_memmap


# ============================================================
# Structural constants — 2D VN, K=5, offsets (N W C E S), big-endian.
# V (alphabet size) is a runtime parameter; NUM_CONFIGS = V**K, TIME_SEP = V,
# ROW_SEP = V+1, VOCAB = V + 2 are derived per invocation.
# ============================================================
K = 5


def num_configs(V):
    return V ** K


def vocab_size(V):
    return V + 2


def time_sep_id(V):
    return V


def row_sep_id(V):
    return V + 1


# ============================================================
# Stochastic rule sampling
# ============================================================

def sample_rules(B, V, rng):
    """Sample B rule tables of shape (B, V**K, V) uniformly from the product of
    (V-1)-simplices (Dirichlet(1, ..., 1) per row) via normalized Exp(1)."""
    z = rng.standard_exponential(size=(B, num_configs(V), V))
    z /= z.sum(axis=-1, keepdims=True)
    return z.astype(np.float32)


# ============================================================
# Vectorized 2D VN stochastic simulation (base-V, big-endian N W C E S)
# ============================================================

def _cfg_idx(g, V):
    """VN neighborhood-config index per cell in base-V.
    g: (..., L1, L2). idx = ((((N*V + W)*V + C)*V + E)*V + S)."""
    N = np.roll(g, 1, axis=-2)
    W = np.roll(g, 1, axis=-1)
    C = g
    E = np.roll(g, -1, axis=-1)
    S = np.roll(g, -1, axis=-2)
    return ((((N * V + W) * V + C) * V + E) * V + S)


def _categorical_from_cdf(cdf, rng, size):
    """Draw one categorical sample per cell via inverse-CDF over the last axis.
    Same identity as 1d_stoch: x = #{v : cdf[v] <= u} with u ~ U[0, 1)."""
    u = rng.random_sample(size=size)
    return (u[..., None] >= cdf).sum(axis=-1).astype(np.int8)


def simulate_batch_stoch(rule_probs, num_steps, L1, L2, V, rng):
    """B trajectories, each under its own stochastic rule.
        rule_probs: (B, V**K, V) float, rows summing to 1
        returns states (B, num_steps, L1, L2) int8 in {0, ..., V-1}.
    The same rule_probs[b] is indexed at every t in this loop, so the rule stays
    fixed throughout the trajectory."""
    B = rule_probs.shape[0]
    NC = num_configs(V)
    assert rule_probs.shape == (B, NC, V), (
        f"rule_probs must be (B, V**K, V)=({B},{NC},{V}); got {rule_probs.shape}")
    states = np.empty((B, num_steps, L1, L2), dtype=np.int8)
    states[:, 0] = rng.randint(0, V, size=(B, L1, L2)).astype(np.int8)
    # Per-rule per-config CDF, snapped so the last entry is exactly 1.
    cdf = rule_probs.cumsum(axis=-1)                              # (B, V**K, V)
    cdf[..., -1] = 1.0
    bidx = np.arange(B)[:, None, None]
    for t in range(1, num_steps):
        idx = _cfg_idx(states[:, t - 1].astype(np.int64), V)      # (B, L1, L2)
        p_cdf = cdf[bidx, idx]                                    # (B, L1, L2, V)
        states[:, t] = _categorical_from_cdf(p_cdf, rng, size=(B, L1, L2))
    return states


def coverage_batch(states, M, V):
    """Per-trajectory: do the first M-1 grids cover all V**K configs?
    states: (B, >=M-1, L1, L2). Returns (B,) bool."""
    B = states.shape[0]
    NC = num_configs(V)
    ctx = states[:, :M - 1].astype(np.int64)                      # (B,M-1,L1,L2)
    flat = _cfg_idx(ctx, V).reshape(B, -1)                        # (B, (M-1)*L1*L2)
    pres = np.zeros((B, NC), dtype=bool)
    pres[np.arange(B)[:, None], flat] = True
    return pres.all(axis=1)


# ============================================================
# Tokenization (2D: T grids of L1 x L2 cells, ROW_SEP between rows, TIME_SEP between grids)
# ============================================================

def compute_seq_len(L1, L2, T):
    # tokens per grid = L1*L2 + (L1-1), plus T-1 TIME_SEPs; then drop last cell for AR.
    return T * (L1 * L2 + (L1 - 1)) + (T - 1) - 1


def build_token_template(L1, L2, T):
    cell_dst, rowsep_dst, timesep_dst, step_tpl = [], [], [], []
    pos = 0
    for t in range(T):
        for r in range(L1):
            for _c in range(L2):
                cell_dst.append(pos); step_tpl.append(t); pos += 1
            if r < L1 - 1:
                rowsep_dst.append(pos); step_tpl.append(t); pos += 1
        if t < T - 1:
            timesep_dst.append(pos); step_tpl.append(t); pos += 1
    return (np.array(cell_dst, dtype=np.int64),
            np.array(rowsep_dst, dtype=np.int64),
            np.array(timesep_dst, dtype=np.int64),
            np.array(step_tpl, dtype=np.int64))


def tokenize_batch(states, tpl, M, require_all_patterns, V):
    """states (B,T,L1,L2) int8 -> inp (B,seq) int8, tgt (B,seq) int8, mask_row (seq,) uint8.
    byk masking: require_all_patterns -> score only rows step>=M; else score all."""
    cell_dst, rowsep_dst, timesep_dst, step_tpl = tpl
    B = states.shape[0]
    total = len(step_tpl)
    out = np.empty((B, total), dtype=np.int8)
    out[:, cell_dst] = states.reshape(B, -1)
    if len(rowsep_dst):
        out[:, rowsep_dst] = row_sep_id(V)
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

def generate_split(name, num_samples, out_dir, L1, L2, T, M, V, seed,
                   require_all_patterns, worker_tag, batch_size, tpl, logger):
    NC = num_configs(V)
    rng = np.random.RandomState(seed)
    seq_len = len(tpl[3]) - 1
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
        probs = sample_rules(B, V, rng)                              # (B, V**K, V)
        states = simulate_batch_stoch(probs, T, L1, L2, V, rng)      # (B, T, L1, L2)
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
        f"V{args.V}_VN_stoch_L{args.L1}x{args.L2}_T{args.T}_M{args.M}"
        f"_n{count_str(args.num_train)}_val{count_str(args.num_eval)}"
        f"_test{count_str(args.num_eval_test)}_seed{args.seed}")


def split_specs(args):
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
    print(f"=== PREPARE: 2D VN stochastic-rule CA (V={V}, K={K}) ===")
    out_dir = resolve_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(logger_path_override or os.path.join(out_dir, "stats.txt"))
    logger.log("=== CONFIG ===")
    logger.log(f"seed={args.seed}  L1={args.L1} L2={args.L2}  T={args.T}  M={args.M}  "
               f"V={V}  K={K}  neighborhood=VN(N,W,C,E,S)")
    logger.log(f"seq_len={compute_seq_len(args.L1, args.L2, args.T)}  "
               f"vocab={vocab_size(V)}  time_sep_id={time_sep_id(V)}  row_sep_id={row_sep_id(V)}")
    logger.log(f"samples: train={args.num_train}, val={args.num_eval}, test={args.num_eval_test}; "
               f"batch_size={args.batch_size}")
    logger.log(f"rule_distribution: uniform simplex per row (Dirichlet(1,...,1) with {V} entries); "
               f"rule shape=({NC},{V}); one rule per sequence, fixed for T steps.")
    logger.log(f"filtering: train unfiltered (mask all-1); "
               f"val+test coverage-filtered on states[:, :M-1] over all {NC} configs (mask step>=M).")
    logger.log(f"output_dir={out_dir}\n")

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"V": V, "K": K, "L1": args.L1, "L2": args.L2,
                   "T": args.T, "M": args.M,
                   "num_train": args.num_train, "num_val": args.num_eval,
                   "num_test": args.num_eval_test, "seed": args.seed,
                   "vocab": vocab_size(V),
                   "time_sep_id": time_sep_id(V), "row_sep_id": row_sep_id(V),
                   "seq_len": compute_seq_len(args.L1, args.L2, args.T),
                   "rule_shape": [NC, V],
                   "rule_distribution": f"uniform_simplex_dirichlet_1_x{V}",
                   "one_rule_per_sequence": True,
                   "neighborhood": "vn_k5_NWCES"}, f, indent=2)
    logger.close()
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--V", type=int, default=2, help="alphabet size (>=2)")
    p.add_argument("--L1", type=int, default=6)
    p.add_argument("--L2", type=int, default=6)
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

    tpl = build_token_template(args.L1, args.L2, args.T)

    if args.prepare:
        out_dir = do_prepare(args)
        print(f"OUTPUT_DIR={out_dir}")
        return

    if args.worker_id is not None:
        if not args.output_dir:
            raise SystemExit("worker mode requires --output_dir (from `--prepare`)")
        W, N = args.worker_id, args.num_workers
        logger = Logger(os.path.join(args.output_dir, f"stats_w{W}.txt"))
        logger.log(f"=== WORKER {W}/{N}  V={args.V}  L1={args.L1}x{args.L2}  "
                   f"dir={args.output_dir} ===")
        for name, n_total, sd, raq in split_specs(args):
            n = worker_share(n_total, W, N)
            if n == 0:
                continue
            logger.log(f"\n-- {name} worker {W}: {n}/{n_total} samples --")
            generate_split(name, n, args.output_dir, args.L1, args.L2,
                           args.T, args.M, args.V,
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
        generate_split(name, n_total, out_dir, args.L1, args.L2, args.T, args.M, args.V,
                       sd, raq, "", args.batch_size, tpl, logger)
    logger.log(f"\n=== DONE === output_dir={out_dir}")
    logger.close()


# ============================================================
# Self-check tests
# ============================================================

def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _test_rule_sampling(V):
    rng = np.random.RandomState(0)
    r = sample_rules(2000, V, rng)
    NC = num_configs(V)
    _assert(r.shape == (2000, NC, V), f"[V={V}] rule shape {r.shape}")
    _assert(np.all(r >= 0.0), f"[V={V}] negative probs")
    sums = r.sum(axis=-1)
    _assert(np.allclose(sums, 1.0, atol=1e-5),
            f"[V={V}] rule rows don't sum to 1: max|sum-1|={np.max(np.abs(sums-1)):.2e}")


def _test_one_hot_rule_matches_deterministic(V):
    """One-hot categorical rule -> exact deterministic transitions."""
    rng = np.random.RandomState(2)
    B, T, L1, L2 = 16, 12, 6, 6
    NC = num_configs(V)
    table = rng.randint(0, V, size=(B, NC)).astype(np.int8)
    probs = np.zeros((B, NC, V), dtype=np.float32)
    probs[np.arange(B)[:, None], np.arange(NC)[None, :], table] = 1.0

    sim_rng = np.random.RandomState(42)
    states = simulate_batch_stoch(probs, T, L1, L2, V, sim_rng)
    ref = np.empty_like(states)
    ref[:, 0] = states[:, 0]
    bidx = np.arange(B)[:, None, None]
    for t in range(1, T):
        idx = _cfg_idx(ref[:, t - 1].astype(np.int64), V)
        ref[:, t] = table[bidx, idx]
    _assert(np.array_equal(states, ref),
            f"[V={V}] one-hot rule stoch sim did not match deterministic reference")


def _test_states_in_range(V):
    rng = np.random.RandomState(4)
    probs = sample_rules(32, V, rng)
    states = simulate_batch_stoch(probs, 12, 6, 6, V, rng)
    _assert(states.min() >= 0 and states.max() <= V - 1,
            f"[V={V}] states range [{states.min()},{states.max()}]")


def _test_categorical_frequencies(V):
    """All rows equal to same p -> empirical freq matches p."""
    rng = np.random.RandomState(5)
    NC = num_configs(V)
    p = rng.standard_exponential(size=V)
    p = (p / p.sum()).astype(np.float32)
    B, T, L1, L2 = 128, 10, 6, 6
    probs = np.broadcast_to(p, (B, NC, V)).astype(np.float32).copy()
    states = simulate_batch_stoch(probs, T, L1, L2, V, rng)
    counts = np.bincount(states[:, 1:].ravel(), minlength=V).astype(np.float64)
    freq = counts / counts.sum()
    _assert(np.max(np.abs(freq - p)) < 0.02,
            f"[V={V}] freq {freq} vs target {p} exceeds 0.02")


def _test_coverage_filter(V):
    """Filtered eval sequences cover all NC configs on states[:, :M-1].
    Grid sizes scale with V so coupon-collector expectation ((M-1)*L1*L2) sits
    above ~2 * NC * H_NC, giving a nonzero acceptance rate."""
    NC = num_configs(V)
    sizes = {2: (6, 6, 10, 20), 3: (16, 16, 15, 20)}  # (L1, L2, M, T)
    L1, L2, M, T = sizes[V]
    rng = np.random.RandomState(3)
    probs = sample_rules(2048, V, rng)
    states = simulate_batch_stoch(probs, T, L1, L2, V, rng)
    cov = coverage_batch(states, M, V)
    kept = states[cov]
    _assert(kept.shape[0] > 0,
            f"[V={V}] no eval samples passed coverage on L1={L1},L2={L2},M={M}")
    ctx = kept[:, :M - 1].astype(np.int64)
    idxs = _cfg_idx(ctx, V).reshape(kept.shape[0], -1)
    pres = np.zeros((kept.shape[0], NC), dtype=bool)
    pres[np.arange(kept.shape[0])[:, None], idxs] = True
    missing = (~pres.all(axis=1)).sum()
    _assert(missing == 0,
            f"[V={V}] {missing}/{kept.shape[0]} filtered samples miss configs")
    print(f"    [V={V}] {kept.shape[0]}/2048 kept, all cover {NC} configs")


def _test_reproducibility(V):
    r1 = sample_rules(32, V, np.random.RandomState(7))
    r2 = sample_rules(32, V, np.random.RandomState(7))
    _assert(np.array_equal(r1, r2), f"[V={V}] rule sampling not reproducible")
    s1 = simulate_batch_stoch(r1, 12, 6, 6, V, np.random.RandomState(11))
    s2 = simulate_batch_stoch(r2, 12, 6, 6, V, np.random.RandomState(11))
    _assert(np.array_equal(s1, s2), f"[V={V}] sim not reproducible")


def _test_seq_len_formula():
    _assert(compute_seq_len(6, 6, 20) == 838,
            f"seq_len(6,6,20)={compute_seq_len(6,6,20)} != 838")
    _assert(compute_seq_len(6, 6, 12) == 502,
            f"seq_len(6,6,12)={compute_seq_len(6,6,12)} != 502")


def _test_end_to_end_smoke(V):
    """Tiny smoke: shapes, ranges, masks, rules; VN tokens."""
    import tempfile
    L1, L2, T, M = 6, 6, 20, 10
    NC = num_configs(V)
    SEP, RSEP = time_sep_id(V), row_sep_id(V)
    with tempfile.TemporaryDirectory() as tmp:
        tpl = build_token_template(L1, L2, T)
        seq_len = compute_seq_len(L1, L2, T)
        logger = Logger(None)
        generate_split("train", 100, tmp, L1, L2, T, M, V, seed=0,
                       require_all_patterns=False, worker_tag="",
                       batch_size=128, tpl=tpl, logger=logger)
        # V=2 val-filter smoke works on L=6x6 in reasonable time
        if V == 2:
            generate_split("val", 50, tmp, L1, L2, T, M, V, seed=1,
                           require_all_patterns=True, worker_tag="",
                           batch_size=512, tpl=tpl, logger=logger)
            splits = (("train", 100, False), ("val", 50, True))
        else:
            splits = (("train", 100, False),)
        for split, n, req in splits:
            inp = np.load(os.path.join(tmp, f"{split}_inputs.npy"))
            tgt = np.load(os.path.join(tmp, f"{split}_targets.npy"))
            msk = np.load(os.path.join(tmp, f"{split}_masks.npy"))
            rls = np.load(os.path.join(tmp, f"{split}_rules.npy"))
            _assert(inp.shape == (n, seq_len), f"[V={V}] {split} inp shape {inp.shape}")
            _assert(rls.shape == (n, NC, V), f"[V={V}] {split} rules shape {rls.shape}")
            _assert(inp.min() >= 0 and inp.max() <= RSEP,
                    f"[V={V}] {split} inp range [{inp.min()},{inp.max()}]")
            _assert(np.all(np.isin(msk, [0, 1])), f"[V={V}] {split} mask not binary")
            _assert(np.allclose(rls.sum(-1), 1.0, atol=1e-5),
                    f"[V={V}] {split} rules rows do not sum to 1")
            if req:
                _, _, _, step_tpl = tpl
                target_steps = step_tpl[1:]
                expected = (target_steps >= M).astype(np.uint8)
                _assert(np.array_equal(msk[0], expected),
                        f"[V={V}] {split} mask does not match step>=M")
            else:
                _assert(msk.sum() == n * seq_len,
                        f"[V={V}] {split} train mask should be all-ones")


def _run_tests():
    print("=== SELF-CHECK TESTS (2D VN K=5) ===")
    _test_seq_len_formula()
    print("[0] seq_len formula OK")
    for V in (2, 3):
        print(f"\n-- V = {V} (NUM_CONFIGS = V**K = {num_configs(V)}) --")
        print(f"[1] rule sampling nonneg + rows sum to 1     (V={V})")
        _test_rule_sampling(V)
        print(f"[2] one-hot rule -> exact deterministic sim  (V={V})")
        _test_one_hot_rule_matches_deterministic(V)
        print(f"[3] generated states in {{0, ..., V-1}}       (V={V})")
        _test_states_in_range(V)
        print(f"[4] categorical freq matches target          (V={V})")
        _test_categorical_frequencies(V)
        print(f"[5] reproducibility for same seed            (V={V})")
        _test_reproducibility(V)
        print(f"[6] coverage over all V**K configs on ctx    (V={V})")
        _test_coverage_filter(V)
        print(f"[7] end-to-end smoke: shapes/ranges/masks    (V={V})")
        _test_end_to_end_smoke(V)
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
