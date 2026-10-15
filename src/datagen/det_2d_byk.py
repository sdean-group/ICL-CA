"""2D CA data generator — fold-k stratified, complement-only equivalence, held-out-rule OOD.

NEW design (see project plan). Differs from the old data_generate_2d.py
(lambda-bins + D4), which is kept only as reference:

  * Equivalence  : ONLY complement. canonical = min(rule, complement(rule)).
  * Stratify     : by fold-k = min(popcount, 32 - popcount), i.e. 0..16.
  * Quotas       : counted in CANONICAL rules. Complement is added at the end of
                   selection, so each canonical -> 2 rules (rule count = canon x2).
  * Five splits  :
        train             train_rules   require_all_patterns=False  mask=all-1   pick=uniform
        val_in_nofilter   train_rules   require_all_patterns=False  mask=all-1   pick=uniform
        val_in_filter     valin_rules   require_all_patterns=True   mask=step>=M pick=uniform
        val_out           valout_rules  require_all_patterns=True   mask=step>=M pick=k-balanced
        test              test_rules    require_all_patterns=True   mask=step>=M pick=k-balanced
    valin_rules = the coverable subset of train_rules (PRE-SCREENED): a train rule
    that can never cover all 32 configs has no valid filtered sample, so excluding
    it is correct, not a bias. This also kills the val_in_filter slowdown.
  * Pools disjoint at canonical level (train vs valout vs test). Order valout->test->train.
  * Scarce k     : measured (6x6,M=8,filter=(30,1)) — only k>=4 supplies 400+400.
                   k=3 ~562 passable total, k<=2 nearly none. val_out/test each get
                   min(VT_QUOTA, #passers//2) per k so the two stay symmetric.

Performance:
  * B (batched): one rule -> simulate `batch_size` trajectories at once, batched
    coverage + batched tokenization (no python per-sample loop). ~10-30x/core.
  * A (parallel): --prepare once (selection + screen -> rules.npz), then run many
    --worker_id W --num_workers N array tasks; each writes its own shard files.
  * int8 tokens / uint8 masks (loader casts to long/float32).

Two-step parallel run:
    # 1) prepare (fast, one process): selection + screen, prints OUTPUT_DIR
    python -m src.datagen.det_2d_byk --prepare --L1 6 --L2 6 --T 12 --M 8 \
        --num_train 50000000 --num_eval 500000 --train_quota 11765 --vt_quota 400 --seed 42
    # 2) SLURM array of workers (see menghan_scripts/gen_2d_byk.slurm)
    python -m src.datagen.det_2d_byk --worker_id $TASK --num_workers 64 \
        --output_dir <DIR from step 1> <same grid/sample args>

Single-process (smoke test): omit --prepare/--worker_id; does everything in one go.
"""

import os
import json
import argparse
import random
from math import comb
from itertools import combinations
from collections import Counter, defaultdict

import numpy as np
from numpy.lib.format import open_memmap


# ============================================================
# Constants — Von Neumann neighborhood, V=2, K=5
# Offsets N,W,C,E,S, big-endian: idx = N*16 + W*8 + C*4 + E*2 + S
# ============================================================
V = 2
K = 5
NUM_CONFIGS = V ** K          # 32
NUM_RULES = V ** NUM_CONFIGS  # 2^32

TOK_0, TOK_1, TIME_SEP, ROW_SEP = 0, 1, 2, 3


# ============================================================
# Vectorized simulation (works for 2D grids and any leading batch/time axes)
# ============================================================

def _cfg_idx(g):
    """Neighborhood-config index per cell. g: (..., L1, L2). Returns same leading shape."""
    N = np.roll(g, 1, axis=-2)
    W = np.roll(g, 1, axis=-1)
    C = g
    E = np.roll(g, -1, axis=-1)
    S = np.roll(g, -1, axis=-2)
    return ((((N * 2 + W) * 2 + C) * 2 + E) * 2 + S)


def rule_table(rule_int):
    return np.array([(rule_int >> i) & 1 for i in range(NUM_CONFIGS)], dtype=np.int8)


def configs_in_grid(grid):
    return set(np.unique(_cfg_idx(grid)).tolist())


def simulate_batch(B, table, num_steps, L1, L2, rng):
    """Simulate B trajectories of `num_steps` grids under one rule. Returns (B,num_steps,L1,L2) int8."""
    states = np.empty((B, num_steps, L1, L2), dtype=np.int8)
    states[:, 0] = rng.randint(0, V, size=(B, L1, L2)).astype(np.int8)
    for t in range(1, num_steps):
        idx = _cfg_idx(states[:, t - 1].astype(np.int64))
        states[:, t] = table[idx]
    return states


def simulate_batch_multi(tables, num_steps, L1, L2, rng):
    """Simulate B trajectories, EACH UNDER ITS OWN RULE. tables (B,32) int8.
    Returns (B,num_steps,L1,L2) int8. Preserves rule diversity within a batch."""
    B = tables.shape[0]
    states = np.empty((B, num_steps, L1, L2), dtype=np.int8)
    states[:, 0] = rng.randint(0, V, size=(B, L1, L2)).astype(np.int8)
    bidx = np.arange(B)[:, None, None]
    for t in range(1, num_steps):
        idx = _cfg_idx(states[:, t - 1].astype(np.int64))   # (B,L1,L2)
        states[:, t] = tables[bidx, idx]
    return states


def coverage_batch(states, M):
    """Per-trajectory: do the first M-1 grids cover all 32 configs? states (B,>=M-1,L1,L2)."""
    B = states.shape[0]
    ctx = states[:, :M - 1].astype(np.int64)        # (B, M-1, L1, L2)
    flat = _cfg_idx(ctx).reshape(B, -1)             # (B, (M-1)*L1*L2)
    pres = np.zeros((B, NUM_CONFIGS), dtype=bool)
    pres[np.arange(B)[:, None], flat] = True
    return pres.all(axis=1)


# --- coverage pre-screen (selection phase), batched ---
def rule_can_cover(rule_int, L1, L2, M, num_tries=30, min_successes=1, rng=None):
    """True iff >= min_successes of num_tries random ICs cover all 32 configs in context."""
    if rng is None:
        rng = np.random.RandomState(rule_int % (2 ** 31))
    table = rule_table(rule_int)
    states = simulate_batch(num_tries, table, M - 1, L1, L2, rng)
    return int(coverage_batch(states, M).sum()) >= min_successes


# ============================================================
# Complement-only equivalence + fold-k
# ============================================================

def popcount(rule_int):
    return bin(rule_int).count("1")


def fold_k(rule_int):
    p = popcount(rule_int)
    return min(p, NUM_CONFIGS - p)


def complement(rule_int):
    """comp[i] = 1 - rule[31 - i] (flip neighborhood + flip output)."""
    comp = 0
    for i in range(NUM_CONFIGS):
        bit = (rule_int >> (NUM_CONFIGS - 1 - i)) & 1
        comp |= (1 - bit) << i
    return comp


def canonical_rule(rule_int):
    return min(rule_int, complement(rule_int))


def expand_pool(canon_list):
    rules = set()
    for c in canon_list:
        rules.add(c)
        rules.add(complement(c))
    return sorted(rules)


# ============================================================
# Tokenization
# ============================================================

def compute_seq_len(L1, L2, T, use_row_sep=True):
    tokens_per_step = L1 * L2 + (L1 - 1 if use_row_sep else 0)
    return T * tokens_per_step + (T - 1) - 1


def build_token_template(L1, L2, T, use_row_sep):
    """Precompute output positions for cells / row-seps / time-seps + per-position step index."""
    cell_dst, rowsep_dst, timesep_dst, step_tpl = [], [], [], []
    pos = 0
    for t in range(T):
        for r in range(L1):
            for _c in range(L2):
                cell_dst.append(pos); step_tpl.append(t); pos += 1
            if use_row_sep and r < L1 - 1:
                rowsep_dst.append(pos); step_tpl.append(t); pos += 1
        if t < T - 1:
            timesep_dst.append(pos); step_tpl.append(t); pos += 1
    return (np.array(cell_dst, dtype=np.int64),
            np.array(rowsep_dst, dtype=np.int64),
            np.array(timesep_dst, dtype=np.int64),
            np.array(step_tpl, dtype=np.int64))


def tokenize_batch(states, tpl, M, require_all_patterns):
    """states (B,T,L1,L2) int8 -> inp (B,seq) int8, tgt (B,seq) int8, mask_row (seq,) uint8."""
    cell_dst, rowsep_dst, timesep_dst, step_tpl = tpl
    B = states.shape[0]
    total = len(step_tpl)
    out = np.empty((B, total), dtype=np.int8)
    out[:, cell_dst] = states.reshape(B, -1)
    if len(rowsep_dst):
        out[:, rowsep_dst] = ROW_SEP
    if len(timesep_dst):
        out[:, timesep_dst] = TIME_SEP
    inp = out[:, :-1]
    tgt = out[:, 1:]
    target_steps = step_tpl[1:]
    if require_all_patterns:
        mask_row = (target_steps >= M).astype(np.uint8)
    else:
        mask_row = np.ones(total - 1, dtype=np.uint8)
    return inp, tgt, mask_row


# ============================================================
# Logger (dual-write stdout + file)
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
# Phase 2 — rule selection (per fold-k, canonical, order valout->test->train)
# ============================================================

def _rule_from_positions(positions):
    r = 0
    for p in positions:
        r |= (1 << p)
    return r


def _enumerate_canonicals(j):
    seen = set()
    n = j if j < NUM_CONFIGS // 2 else 16
    for pos in combinations(range(NUM_CONFIGS), n):
        seen.add(canonical_rule(_rule_from_positions(pos)))
    return seen


def _avail_canon(j):
    return comb(NUM_CONFIGS, j) if j < NUM_CONFIGS // 2 else comb(NUM_CONFIGS, 16) // 2


def select_rules(L1, L2, M, train_quota, vt_quota, enum_threshold,
                 num_tries, min_successes, seed):
    pyr = random.Random(seed)
    filter_rng = np.random.RandomState(seed + 1)
    train_by_k, valout_by_k, test_by_k, stats = {}, {}, {}, []

    for j in range(0, NUM_CONFIGS // 2 + 1):
        avail = _avail_canon(j)
        n_tested = 0

        if j <= enum_threshold:
            all_canon = sorted(_enumerate_canonicals(j))
            passers = [c for c in all_canon
                       if rule_can_cover(c, L1, L2, M, num_tries, min_successes, filter_rng)]
            n_tested = len(all_canon)
            pyr.shuffle(passers)
            each = min(vt_quota, len(passers) // 2)
            valout, test = passers[:each], passers[each:2 * each]
            used = set(valout) | set(test)
            train_pool = [c for c in all_canon if c not in used]
            pyr.shuffle(train_pool)
            train = train_pool[:train_quota]
        else:
            seen, passers = set(), []
            target_pass = 2 * vt_quota
            max_attempts = target_pass * 300 + 2000
            for _ in range(max_attempts):
                if len(passers) >= target_pass:
                    break
                canon = canonical_rule(_rule_from_positions(pyr.sample(range(NUM_CONFIGS), j)))
                if canon in seen:
                    continue
                seen.add(canon); n_tested += 1
                if rule_can_cover(canon, L1, L2, M, num_tries, min_successes, filter_rng):
                    passers.append(canon)
            each = min(vt_quota, len(passers) // 2)
            valout, test = passers[:each], passers[each:2 * each]
            train_seen = set(valout) | set(test) | set(passers)
            train = []
            t_max = train_quota * 50 + 5000
            for _ in range(t_max):
                if len(train) >= train_quota:
                    break
                canon = canonical_rule(_rule_from_positions(pyr.sample(range(NUM_CONFIGS), j)))
                if canon in train_seen:
                    continue
                train_seen.add(canon); train.append(canon)

        train_by_k[j], valout_by_k[j], test_by_k[j] = train, valout, test
        stats.append({"k": j, "avail": avail, "tested": n_tested,
                      "train_canon": len(train), "valout_canon": len(valout),
                      "test_canon": len(test)})
    return train_by_k, valout_by_k, test_by_k, stats


# ============================================================
# Pre-screen train pool -> coverable subset (for val_in_filter)
# ============================================================

def prescreen_coverable(rules, L1, L2, M, target, scan_cap, screen_batch, seed,
                        table_cache, logger):
    rng = np.random.RandomState(seed)
    order = np.array(rules, dtype=np.int64)
    rng.shuffle(order)
    sim_rng = np.random.RandomState(seed + 1)
    coverable = []
    scanned = 0
    for r in order:
        if len(coverable) >= target or scanned >= scan_cap:
            break
        scanned += 1
        r = int(r)
        tbl = table_cache.get(r)
        if tbl is None:
            tbl = rule_table(r); table_cache[r] = tbl
        states = simulate_batch(screen_batch, tbl, M - 1, L1, L2, sim_rng)
        if coverage_batch(states, M).any():
            coverable.append(r)
    logger.log(f"  prescreen val_in pool: scanned {scanned}/{len(rules)} train rules, "
               f"coverable {len(coverable)} (target {target}, screen_batch {screen_batch})")
    return coverable


# ============================================================
# Phase 3 — batched sample generation with sharded writing
# ============================================================

def _sample_rules(B, rng, rule_pick, rules_arr, bucket_arr, nonempty_arr):
    """Sample B rules (with replacement). uniform over pool, or uniform-over-fold-k."""
    if rule_pick == "k_balanced":
        ks = nonempty_arr[rng.randint(len(nonempty_arr), size=B)]
        out = np.empty(B, dtype=np.int64)
        for k in nonempty_arr:
            m = ks == k
            c = int(m.sum())
            if c:
                bk = bucket_arr[int(k)]
                out[m] = bk[rng.randint(len(bk), size=c)]
        return out
    return rules_arr[rng.randint(len(rules_arr), size=B)]


def generate_split(name, rules, num_samples, out_dir, L1, L2, T, M, seed,
                   require_all_patterns, rule_pick, worker_tag,
                   batch_size, tpl, logger, table_cache):
    """Write one split as 4 memmap .npy files: {name}{worker_tag}_{inputs,targets,masks,rules}.npy.
    .npy (raw) is memmap-able for training; .npz (zip) is not."""
    rng = np.random.RandomState(seed)
    rules_arr = np.array(rules, dtype=np.int64)
    bucket_arr, nonempty_arr = None, None
    if rule_pick == "k_balanced":
        by_k = defaultdict(list)
        for r in rules:
            by_k[fold_k(int(r))].append(int(r))
        bucket_arr = {k: np.array(v, dtype=np.int64) for k, v in by_k.items()}
        nonempty_arr = np.array(sorted(by_k), dtype=np.int64)

    seq_len = len(tpl[3]) - 1
    base = os.path.join(out_dir, f"{name}{worker_tag}")
    mm_inp = open_memmap(base + "_inputs.npy", mode="w+", dtype=np.int8, shape=(num_samples, seq_len))
    mm_tgt = open_memmap(base + "_targets.npy", mode="w+", dtype=np.int8, shape=(num_samples, seq_len))
    mm_mask = open_memmap(base + "_masks.npy", mode="w+", dtype=np.uint8, shape=(num_samples, seq_len))
    mm_rules = open_memmap(base + "_rules.npy", mode="w+", dtype=np.int64, shape=(num_samples,))

    collected = n_sim = 0
    rule_counter, k_counter = Counter(), Counter()

    def get_tables(batch_rules):
        out = np.empty((len(batch_rules), NUM_CONFIGS), dtype=np.int8)
        for i, r in enumerate(batch_rules):
            r = int(r)
            t = table_cache.get(r)
            if t is None:
                t = rule_table(r); table_cache[r] = t
            out[i] = t
        return out

    log_every = 2_000_000 if num_samples >= 5_000_000 else 100_000
    next_log = log_every

    while collected < num_samples:
        B = batch_size
        batch_rules = _sample_rules(B, rng, rule_pick, rules_arr, bucket_arr, nonempty_arr)
        tables = get_tables(batch_rules)
        states = simulate_batch_multi(tables, T, L1, L2, rng)
        n_sim += B
        if require_all_patterns:
            cov = coverage_batch(states, M)
            states = states[cov]
            brules = batch_rules[cov]
        else:
            brules = batch_rules

        remaining = num_samples - collected
        if len(states) > remaining:
            states = states[:remaining]
            brules = brules[:remaining]
        nb = len(states)
        if nb == 0:
            continue
        inp, tgt, mask_row = tokenize_batch(states, tpl, M, require_all_patterns)
        s = slice(collected, collected + nb)
        mm_inp[s] = inp
        mm_tgt[s] = tgt
        mm_mask[s] = mask_row          # (seq_len,) broadcasts over nb rows
        mm_rules[s] = brules

        # stats (vectorized over the batch's unique rules)
        u, c = np.unique(brules, return_counts=True)
        for rr, cc in zip(u.tolist(), c.tolist()):
            rule_counter[rr] += cc
            k_counter[fold_k(rr)] += cc
        collected += nb

        if collected >= next_log:
            logger.log(f"    [{name}{worker_tag}] {collected}/{num_samples}  "
                       f"accept={collected/max(n_sim,1):.2f}  uniq_rules={len(rule_counter)}")
            next_log += log_every

    mm_inp.flush(); mm_tgt.flush(); mm_mask.flush(); mm_rules.flush()
    del mm_inp, mm_tgt, mm_mask, mm_rules

    logger.log(f"  [{name}{worker_tag}] DONE {collected} -> {name}{worker_tag}_*.npy  "
               f"seq_len={seq_len}  raq={require_all_patterns}  pick={rule_pick}  "
               f"accept_rate={collected/max(n_sim,1):.3f}  uniq_rules={len(rule_counter)}/{len(rules)}")
    k_line = "  ".join(f"k{k}:{100*k_counter[k]/collected:.1f}%" for k in sorted(k_counter))
    logger.log(f"    k distribution: {k_line}")


# ============================================================
# Orchestration
# ============================================================

def count_str(n):
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def resolve_output_dir(args, train_canon, valout_canon, test_canon):
    if args.output_dir:
        return args.output_dir
    voto = len(valout_canon) + len(test_canon)
    return os.path.join(
        args.output_root,
        f"VN_L{args.L1}x{args.L2}_T{args.T}_M{args.M}_Kv{V}"
        f"_train{count_str(len(train_canon))}_voto{count_str(voto)}"
        f"_n{count_str(args.num_train)}_test{count_str(args.num_eval)}_seed{args.seed}")


def do_prepare(args, logger_path_override=None):
    """Selection + expand + pre-screen -> write rules.npz + stats header. Returns dir + pools."""
    print(f"=== PREPARE: selection (by fold-k, complement-only) ===")
    train_by_k, valout_by_k, test_by_k, stats = select_rules(
        args.L1, args.L2, args.M, args.train_quota, args.vt_quota,
        args.enum_threshold, args.num_tries, args.min_successes, args.seed)
    train_canon = [c for j in train_by_k for c in train_by_k[j]]
    valout_canon = [c for j in valout_by_k for c in valout_by_k[j]]
    test_canon = [c for j in test_by_k for c in test_by_k[j]]

    out_dir = resolve_output_dir(args, train_canon, valout_canon, test_canon)
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(logger_path_override or os.path.join(out_dir, "stats.txt"))

    logger.log("=== CONFIG ===")
    logger.log(f"seed={args.seed}  grid={args.L1}x{args.L2}  T={args.T}  M={args.M}  V={V}  K={K}")
    logger.log(f"enum/sample boundary fold-k<= {args.enum_threshold}; filter=({args.num_tries},{args.min_successes})")
    logger.log(f"use_row_sep={not args.no_row_sep}  seq_len={compute_seq_len(args.L1,args.L2,args.T,not args.no_row_sep)}")
    logger.log(f"samples: train={args.num_train}, each eval={args.num_eval}; batch_size={args.batch_size}")
    logger.log(f"quotas (canonical): train={args.train_quota}/k, val_out=test={args.vt_quota}/k each")
    logger.log(f"complement: quotas are CANONICAL; each -> 2 rules (x2)\noutput_dir={out_dir}\n")

    logger.log("=== RULE STATS (per fold-k; canonical counts, rule = canon x2) ===")
    logger.log(f"{'k':>3} {'avail':>13} {'tested':>8} {'tr_can':>8} {'vo_can':>7} {'te_can':>7} "
               f"{'tr_rule':>9} {'vo_rule':>8} {'te_rule':>8}")
    tot = {"a": 0, "tr": 0, "vo": 0, "te": 0}
    for s in stats:
        logger.log(f"{s['k']:>3} {s['avail']:>13,} {s['tested']:>8} {s['train_canon']:>8} "
                   f"{s['valout_canon']:>7} {s['test_canon']:>7} {s['train_canon']*2:>9} "
                   f"{s['valout_canon']*2:>8} {s['test_canon']*2:>8}")
        tot["a"] += s["avail"]; tot["tr"] += s["train_canon"]
        tot["vo"] += s["valout_canon"]; tot["te"] += s["test_canon"]
    logger.log(f"{'SUM':>3} {tot['a']:>13,} {'-':>8} {tot['tr']:>8} {tot['vo']:>7} {tot['te']:>7} "
               f"{tot['tr']*2:>9} {tot['vo']*2:>8} {tot['te']*2:>8}\n")

    train_rules = expand_pool(train_canon)
    valout_rules = expand_pool(valout_canon)
    test_rules = expand_pool(test_canon)

    s_tr, s_vo, s_te = set(train_canon), set(valout_canon), set(test_canon)
    logger.log("=== ISOLATION (canonical) ===")
    logger.log(f"train∩valout={len(s_tr&s_vo)}  train∩test={len(s_tr&s_te)}  valout∩test={len(s_vo&s_te)} (want 0)")
    assert not (s_tr & s_vo) and not (s_tr & s_te) and not (s_vo & s_te), "pool overlap!"
    r_tr, r_vo, r_te = set(train_rules), set(valout_rules), set(test_rules)
    assert not (r_tr & r_vo) and not (r_tr & r_te) and not (r_vo & r_te), "rule overlap after expand!"
    logger.log(f"rules: train={len(train_rules)} valout={len(valout_rules)} test={len(test_rules)}")

    table_cache = {}
    valin_rules = prescreen_coverable(
        train_rules, args.L1, args.L2, args.M, args.valin_pool,
        args.valin_pool * args.valin_scan_factor, args.screen_batch,
        args.seed + 999, table_cache, logger)
    logger.log("")

    np.savez(os.path.join(out_dir, "rules.npz"),
             train_rules=np.array(train_rules, dtype=np.int64),
             valin_rules=np.array(valin_rules, dtype=np.int64),
             valout_rules=np.array(valout_rules, dtype=np.int64),
             test_rules=np.array(test_rules, dtype=np.int64))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"L1": args.L1, "L2": args.L2, "T": args.T, "M": args.M,
                   "num_train": args.num_train, "num_eval": args.num_eval,
                   "seed": args.seed, "use_row_sep": not args.no_row_sep,
                   "n_train_rules": len(train_rules), "n_valin_rules": len(valin_rules),
                   "n_valout_rules": len(valout_rules), "n_test_rules": len(test_rules)}, f, indent=2)
    logger.close()
    return out_dir, dict(train_rules=train_rules, valin_rules=valin_rules,
                         valout_rules=valout_rules, test_rules=test_rules)


def split_specs(args, pools):
    """(name, rules, num_samples, seed, require_all_patterns, rule_pick, sharded)."""
    return [
        ("train",           pools["train_rules"],  args.num_train, args.seed,       False, "uniform",    True),
        ("val_in_nofilter", pools["train_rules"],  args.num_eval,  args.seed + 50,  False, "uniform",    False),
        ("val_in_filter",   pools["valin_rules"],  args.num_eval,  args.seed + 60,  True,  "uniform",    False),
        ("val_out",         pools["valout_rules"], args.num_eval,  args.seed + 100, True,  "k_balanced", False),
        ("test",            pools["test_rules"],   args.num_eval,  args.seed + 150, True,  "k_balanced", False),
    ]


def worker_share(total, w, n):
    base, rem = divmod(total, n)
    return base + (1 if w < rem else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L1", type=int, default=6)
    p.add_argument("--L2", type=int, default=6)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--num_train", type=int, default=50_000_000)
    p.add_argument("--num_eval", type=int, default=500_000)
    p.add_argument("--train_quota", type=int, default=11765)
    p.add_argument("--vt_quota", type=int, default=400)
    p.add_argument("--enum_threshold", type=int, default=3)
    p.add_argument("--num_tries", type=int, default=30)
    p.add_argument("--min_successes", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--shard_size", type=int, default=500_000)
    p.add_argument("--valin_pool", type=int, default=20000, help="target coverable train rules for val_in_filter")
    p.add_argument("--valin_scan_factor", type=int, default=8, help="scan up to valin_pool*this train rules")
    p.add_argument("--screen_batch", type=int, default=64)
    p.add_argument("--no_row_sep", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", type=str, default="data")
    p.add_argument("--output_dir", type=str, default=None)
    # modes
    p.add_argument("--prepare", action="store_true", help="run selection+screen, write rules.npz, print OUTPUT_DIR")
    p.add_argument("--worker_id", type=int, default=None, help="this worker's index (0..num_workers-1)")
    p.add_argument("--num_workers", type=int, default=1)
    args = p.parse_args()
    use_row_sep = not args.no_row_sep
    tpl = build_token_template(args.L1, args.L2, args.T, use_row_sep)

    # ---- PREPARE mode ----
    if args.prepare:
        out_dir, _ = do_prepare(args)
        print(f"OUTPUT_DIR={out_dir}")
        return

    # ---- WORKER mode ----
    if args.worker_id is not None:
        if not args.output_dir:
            raise SystemExit("worker mode requires --output_dir (from `--prepare`)")
        W, N = args.worker_id, args.num_workers
        rz = np.load(os.path.join(args.output_dir, "rules.npz"))
        pools = {k: rz[k].tolist() for k in ("train_rules", "valin_rules", "valout_rules", "test_rules")}
        logger = Logger(os.path.join(args.output_dir, f"stats_w{W}.txt"))
        logger.log(f"=== WORKER {W}/{N}  dir={args.output_dir} ===")
        table_cache = {}
        for name, rules, n_total, sd, raq, pick, sharded in split_specs(args, pools):
            n = worker_share(n_total, W, N)
            if n == 0:
                continue
            logger.log(f"\n-- {name} worker {W}: {n}/{n_total} samples, {len(rules)} rules --")
            generate_split(name, rules, n, args.output_dir, args.L1, args.L2, args.T,
                           args.M, sd + W * 100003, raq, pick, f"_w{W}",
                           args.batch_size, tpl, logger, table_cache)
        logger.log(f"\n=== WORKER {W} DONE ===")
        logger.close()
        return

    # ---- SINGLE-PROCESS mode (smoke / small) ----
    out_dir, pools = do_prepare(args)
    logger = Logger(os.path.join(out_dir, "stats.txt"), mode="a")
    logger.log("\n=== SAMPLE GENERATION (single process) ===")
    table_cache = {}
    for name, rules, n_total, sd, raq, pick, sharded in split_specs(args, pools):
        logger.log(f"\n-- {name}: {n_total} samples, {len(rules)} rules --")
        generate_split(name, rules, n_total, out_dir, args.L1, args.L2, args.T,
                       args.M, sd, raq, pick, "",
                       args.batch_size, tpl, logger, table_cache)
    logger.log(f"\n=== DONE === output_dir={out_dir}")
    logger.close()


if __name__ == "__main__":
    main()
