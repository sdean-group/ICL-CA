"""1D V=2 (ECA) CA data generator — Wolfram-equivalence rule pools, byk five-split OOD.

Elementary Cellular Automata (V=2, K=3, offsets (-1,0,1) -> 256 rules). Rules are
folded into the 88 canonical classes under the FULL symmetry group (reflection ×
0<->1 complement), i.e. Wolfram's minimal representatives + their orbits
(WOLFRAM_EQUIVALENTS). The 88 canonicals are split disjointly into train / valout
/ test, stratified by Wolfram behavioral class (I/II/III/IV), then each canonical
is expanded to its full orbit -> three disjoint rule pools that tile all 256 rules.

Everything else follows the byk architecture of data_generate_1d_byk.py (the V=3
histogram-stratified sibling): batched multi-rule simulation, coverage-filtered
eval splits, byk masking, int8 tokens / uint8 masks, --prepare once then
--worker_id array tasks for parallel sample generation.

  * Cells       : 1D ring of length L, states {0,1}. Neighborhood (L,C,R),
                  config idx = L*4 + C*2 + R. NUM_CONFIGS = 8, NUM_RULES = 256.
  * Equivalence : reflection × complement (Wolfram). 88 canonical classes.
  * Rule split  : train/valout/test disjoint over canonicals, stratified by
                  Wolfram class at train_frac/valout_frac (default .8/.1/.1),
                  then orbit-expanded. valin = coverable subset of train.
  * Five splits (byk masking — train scores EVERY position, evals only step>=M):
        train             train_rules   require_all_patterns=False  mask=all-1    pick=uniform
        val_in_nofilter   train_rules   require_all_patterns=False  mask=all-1    pick=uniform
        val_in_filter     valin_rules   require_all_patterns=True   mask=step>=M  pick=uniform
        val_out           valout_rules  require_all_patterns=True   mask=step>=M  pick=uniform
        test              test_rules    require_all_patterns=True   mask=step>=M  pick=uniform

Two-step parallel run (default grid L=16, T=10, M=4):
    python -m src.datagen.det_1d_wolfram --prepare --L 16 --T 10 --M 4 \
        --num_train 50000000 --num_eval 500000 --seed 42
    python -m src.datagen.det_1d_wolfram --worker_id $TASK --num_workers 32 \
        --output_dir <DIR from step 1> <same grid/sample args>

Single-process smoke test: omit --prepare/--worker_id (use small --num_train).
"""

import os
import json
import argparse
import random
from math import log2
from collections import Counter, defaultdict

import numpy as np
from numpy.lib.format import open_memmap


# ============================================================
# Constants — 1D, V=2, K=3, offsets (-1,0,1), big-endian (L,C,R)
# config idx = L*4 + C*2 + R
# ============================================================
V = 2
K = 3
NUM_CONFIGS = V ** K          # 8
# NUM_RULES = V ** NUM_CONFIGS = 2^8 = 256  (all ECA rules)

TIME_SEP = V                  # 2  (SEP between time rows)
VOCAB = V + 1                 # {0,1,SEP}

POW2 = np.array([V ** i for i in range(NUM_CONFIGS)], dtype=np.int64)
H_MAX = log2(V)               # 1.0 max output entropy (bits)


# ============================================================
# Rule <-> table (base-2, little-endian: table[i] = output for config i)
# ============================================================

def rule_table(rule_int):
    t = np.empty(NUM_CONFIGS, dtype=np.int8)
    r = int(rule_int)
    for i in range(NUM_CONFIGS):
        t[i] = r % V
        r //= V
    return t


def table_to_rule(table):
    return int(np.dot(table.astype(np.int64), POW2))


# ============================================================
# Wolfram equivalence classes (V=2, standard offsets)
#   canonical minimal rule -> [other rules in its reflection×complement orbit]
#   ALL_CLASSES groups the 88 canonicals by Wolfram behavioral class.
# ============================================================

WOLFRAM_CLASS_I = [0, 8, 32, 40, 128, 136, 160, 168]
WOLFRAM_CLASS_II = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26, 27, 28, 29,
                    33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76,
                    77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142, 152, 154, 156, 162, 164,
                    170, 172, 178, 184, 200, 204, 232]
WOLFRAM_CLASS_III = [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150]
WOLFRAM_CLASS_IV = [41, 54, 106, 110]
ALL_CLASSES = [("I", WOLFRAM_CLASS_I), ("II", WOLFRAM_CLASS_II),
               ("III", WOLFRAM_CLASS_III), ("IV", WOLFRAM_CLASS_IV)]

WOLFRAM_EQUIVALENTS = {
    0: [255], 1: [127], 2: [16, 191, 247], 3: [17, 63, 119],
    4: [223], 5: [95], 6: [20, 159, 215], 7: [21, 31, 87],
    8: [64, 239, 253], 9: [65, 111, 125], 10: [80, 175, 245],
    11: [47, 81, 117], 12: [68, 207, 221], 13: [69, 79, 93],
    14: [84, 143, 213], 15: [85], 18: [183], 19: [55],
    22: [151], 23: [], 24: [66, 189, 231], 25: [61, 67, 103],
    26: [82, 167, 181], 27: [39, 53, 83], 28: [70, 157, 199],
    29: [71], 30: [86, 135, 149], 32: [251], 33: [123],
    34: [48, 187, 243], 35: [49, 59, 115], 36: [219], 37: [91],
    38: [52, 155, 211], 40: [96, 235, 249], 41: [97, 107, 121],
    42: [112, 171, 241], 43: [113], 44: [100, 203, 217],
    45: [75, 89, 101], 46: [116, 139, 209], 50: [179], 51: [],
    54: [147], 56: [98, 185, 227], 57: [99], 58: [114, 163, 177],
    60: [102, 153, 195], 62: [118, 131, 145], 72: [237], 73: [109],
    74: [88, 173, 229], 76: [205], 77: [], 78: [92, 141, 197],
    90: [165], 94: [133], 104: [233], 105: [],
    106: [120, 169, 225], 108: [201], 110: [124, 137, 193],
    122: [161], 126: [129], 128: [254], 130: [144, 190, 246],
    132: [222], 134: [148, 158, 214], 136: [192, 238, 252],
    138: [174, 208, 244], 140: [196, 206, 220], 142: [212],
    146: [182], 150: [], 152: [188, 194, 230], 154: [166, 180, 210],
    156: [198], 160: [250], 162: [176, 186, 242], 164: [218],
    168: [224, 234, 248], 170: [240], 172: [202, 216, 228],
    178: [], 184: [226], 200: [236], 204: [], 232: [],
}


def wolfram_orbit(canonical):
    return [canonical] + WOLFRAM_EQUIVALENTS.get(canonical, [])


def expand_wolfram(canon_list):
    rules = set()
    for c in canon_list:
        rules.update(wolfram_orbit(c))
    return sorted(rules)


def split_wolfram_rules(seed, train_frac, valout_frac):
    """Split the 88 canonicals into train/valout/test, stratified by Wolfram
    behavioral class. test_frac = valout_frac (symmetric held-out sets)."""
    rng = random.Random(seed)
    train_canon, valout_canon, test_canon, per_class = [], [], [], []
    for cname, canons in ALL_CLASSES:
        cl = list(canons); rng.shuffle(cl); n = len(cl)
        n_vo = max(1, round(valout_frac * n))
        n_te = max(1, round(valout_frac * n))
        if n - n_vo - n_te < 1:                     # tiny-class safeguard
            n_vo = min(n_vo, (n - 1) // 2)
            n_te = min(n_te, (n - 1) - n_vo)
        vo, te, tr = cl[:n_vo], cl[n_vo:n_vo + n_te], cl[n_vo + n_te:]
        valout_canon += vo; test_canon += te; train_canon += tr
        per_class.append({"cls": cname, "n": n, "train": len(tr),
                          "valout": len(vo), "test": len(te)})
    return train_canon, valout_canon, test_canon, per_class


# ============================================================
# Output histogram / entropy (for logging + the distribution plot)
# ============================================================

def hist_stratum(rule_int):
    counts = np.bincount(rule_table(rule_int), minlength=V)
    return tuple(sorted(counts.tolist(), reverse=True))


def stratum_entropy(strat):
    n = sum(strat)
    h = 0.0
    for c in strat:
        if c:
            pf = c / n
            h -= pf * log2(pf)
    return h


# ============================================================
# Vectorized 1D simulation (base-2 ring)
# ============================================================

def _cfg_idx(g):
    """Neighborhood-config index per cell. g: (..., L). idx = L*4 + C*2 + R."""
    Ln = np.roll(g, 1, axis=-1)     # left  (i-1)
    C = g
    Rn = np.roll(g, -1, axis=-1)    # right (i+1)
    return (Ln * 2 + C) * 2 + Rn


def simulate_batch(B, table, num_steps, L, rng):
    """B trajectories under ONE rule. Returns (B,num_steps,L) int8."""
    states = np.empty((B, num_steps, L), dtype=np.int8)
    states[:, 0] = rng.randint(0, V, size=(B, L)).astype(np.int8)
    for t in range(1, num_steps):
        idx = _cfg_idx(states[:, t - 1].astype(np.int64))
        states[:, t] = table[idx]
    return states


def simulate_batch_multi(tables, num_steps, L, rng):
    """B trajectories, EACH UNDER ITS OWN RULE. tables (B,8) int8.
    Returns (B,num_steps,L) int8."""
    B = tables.shape[0]
    states = np.empty((B, num_steps, L), dtype=np.int8)
    states[:, 0] = rng.randint(0, V, size=(B, L)).astype(np.int8)
    bidx = np.arange(B)[:, None]
    for t in range(1, num_steps):
        idx = _cfg_idx(states[:, t - 1].astype(np.int64))   # (B,L)
        states[:, t] = tables[bidx, idx]
    return states


def coverage_batch(states, M):
    """Per-trajectory: do the first M-1 rows cover all 8 configs? states (B,>=M-1,L)."""
    B = states.shape[0]
    ctx = states[:, :M - 1].astype(np.int64)        # (B, M-1, L)
    flat = _cfg_idx(ctx).reshape(B, -1)
    pres = np.zeros((B, NUM_CONFIGS), dtype=bool)
    pres[np.arange(B)[:, None], flat] = True
    return pres.all(axis=1)


# ============================================================
# Tokenization (1D: T rows of L cells, TIME_SEP between rows; no row-sep)
# ============================================================

def compute_seq_len(L, T):
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


def tokenize_batch(states, tpl, M, require_all_patterns):
    """states (B,T,L) int8 -> inp (B,seq) int8, tgt (B,seq) int8, mask_row (seq,) uint8.
    byk masking: require_all_patterns -> score only rows step>=M; else score all."""
    cell_dst, timesep_dst, step_tpl = tpl
    B = states.shape[0]
    total = len(step_tpl)
    out = np.empty((B, total), dtype=np.int8)
    out[:, cell_dst] = states.reshape(B, -1)
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
# Pre-screen train pool -> coverable subset (for val_in_filter)
# ============================================================

def prescreen_coverable(rules, L, M, target, scan_cap, screen_batch, seed,
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
        states = simulate_batch(screen_batch, tbl, M - 1, L, sim_rng)
        if coverage_batch(states, M).any():
            coverable.append(r)
    logger.log(f"  prescreen val_in pool: scanned {scanned}/{len(rules)} train rules, "
               f"coverable {len(coverable)} (target {target}, screen_batch {screen_batch})")
    return coverable


# ============================================================
# Phase 3 — batched sample generation with memmap writing
# ============================================================

def _sample_rules(B, rng, rule_pick, rules_arr, buckets, n_nonempty):
    """uniform over pool, or uniform-over-stratum (hist_balanced)."""
    if rule_pick == "hist_balanced":
        sidx = rng.randint(n_nonempty, size=B)
        out = np.empty(B, dtype=np.int64)
        for i in range(n_nonempty):
            m = sidx == i
            c = int(m.sum())
            if c:
                bk = buckets[i]
                out[m] = bk[rng.randint(len(bk), size=c)]
        return out
    return rules_arr[rng.randint(len(rules_arr), size=B)]


def generate_split(name, rules, num_samples, out_dir, L, T, M, seed,
                   require_all_patterns, rule_pick, worker_tag,
                   batch_size, tpl, logger, table_cache):
    rng = np.random.RandomState(seed)
    rules_arr = np.array(rules, dtype=np.int64)
    buckets, n_nonempty = None, 0
    if rule_pick == "hist_balanced":
        by_s = defaultdict(list)
        for r in rules:
            by_s[hist_stratum(int(r))].append(int(r))
        strat_keys = sorted(by_s)
        buckets = [np.array(by_s[s], dtype=np.int64) for s in strat_keys]
        n_nonempty = len(buckets)

    seq_len = len(tpl[2]) - 1
    base = os.path.join(out_dir, f"{name}{worker_tag}")
    mm_inp = open_memmap(base + "_inputs.npy", mode="w+", dtype=np.int8, shape=(num_samples, seq_len))
    mm_tgt = open_memmap(base + "_targets.npy", mode="w+", dtype=np.int8, shape=(num_samples, seq_len))
    mm_mask = open_memmap(base + "_masks.npy", mode="w+", dtype=np.uint8, shape=(num_samples, seq_len))
    mm_rules = open_memmap(base + "_rules.npy", mode="w+", dtype=np.int64, shape=(num_samples,))

    collected = n_sim = 0
    rule_counter, strat_counter = Counter(), Counter()

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
        batch_rules = _sample_rules(B, rng, rule_pick, rules_arr, buckets, n_nonempty)
        tables = get_tables(batch_rules)
        states = simulate_batch_multi(tables, T, L, rng)
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
        mm_mask[s] = mask_row
        mm_rules[s] = brules

        u, c = np.unique(brules, return_counts=True)
        for rr, cc in zip(u.tolist(), c.tolist()):
            rule_counter[rr] += cc
            strat_counter[hist_stratum(rr)] += cc
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
    ebins = defaultdict(int)
    for st, cc in strat_counter.items():
        ebins[round(stratum_entropy(st), 2)] += cc
    e_line = "  ".join(f"H{h}:{100*ebins[h]/collected:.1f}%" for h in sorted(ebins))
    logger.log(f"    entropy distribution (H bin -> % samples): {e_line}")
    logger.log(f"    unique strata hit: {len(strat_counter)}")


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
    return os.path.join(
        args.output_root,
        f"V{V}K{K}_ECA_L{args.L}_T{args.T}_M{args.M}"
        f"_wolfram{len(train_canon)}-{len(valout_canon)}-{len(test_canon)}can"
        f"_n{count_str(args.num_train)}_test{count_str(args.num_eval)}_seed{args.seed}")


def do_prepare(args, logger_path_override=None):
    print("=== PREPARE: Wolfram-equivalence rule split (reflection×complement) ===")
    train_canon, valout_canon, test_canon, per_class = split_wolfram_rules(
        args.seed, args.train_frac, args.valout_frac)

    out_dir = resolve_output_dir(args, train_canon, valout_canon, test_canon)
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(logger_path_override or os.path.join(out_dir, "stats.txt"))

    logger.log("=== CONFIG ===")
    logger.log(f"seed={args.seed}  L={args.L}  T={args.T}  M={args.M}  V={V}  K={K}  offsets=(-1,0,1)")
    logger.log(f"seq_len={compute_seq_len(args.L,args.T)}  vocab={VOCAB}")
    logger.log(f"samples: train={args.num_train}, each eval={args.num_eval}; batch_size={args.batch_size}")
    logger.log(f"split: train_frac={args.train_frac} valout_frac=test_frac={args.valout_frac}; "
               f"stratified by Wolfram class; each canonical -> full orbit")
    logger.log(f"equivalence: reflection × 0<->1 complement (88 canonical classes)\noutput_dir={out_dir}\n")

    logger.log("=== RULE SPLIT (canonical counts per Wolfram class) ===")
    logger.log(f"{'class':>6} {'n_canon':>8} {'train':>7} {'valout':>7} {'test':>6}")
    for pc in per_class:
        logger.log(f"{pc['cls']:>6} {pc['n']:>8} {pc['train']:>7} {pc['valout']:>7} {pc['test']:>6}")
    logger.log(f"{'SUM':>6} {88:>8} {len(train_canon):>7} {len(valout_canon):>7} {len(test_canon):>6}\n")

    train_rules = expand_wolfram(train_canon)
    valout_rules = expand_wolfram(valout_canon)
    test_rules = expand_wolfram(test_canon)

    s_tr, s_vo, s_te = set(train_canon), set(valout_canon), set(test_canon)
    logger.log("=== ISOLATION (canonical) ===")
    logger.log(f"train∩valout={len(s_tr&s_vo)}  train∩test={len(s_tr&s_te)}  valout∩test={len(s_vo&s_te)} (want 0)")
    assert not (s_tr & s_vo) and not (s_tr & s_te) and not (s_vo & s_te), "canonical overlap!"
    r_tr, r_vo, r_te = set(train_rules), set(valout_rules), set(test_rules)
    assert not (r_tr & r_vo) and not (r_tr & r_te) and not (r_vo & r_te), "rule overlap after expand!"
    logger.log(f"rules: train={len(train_rules)} valout={len(valout_rules)} test={len(test_rules)} "
               f"(total {len(r_tr | r_vo | r_te)}/256)")

    table_cache = {}
    valin_rules = prescreen_coverable(
        train_rules, args.L, args.M, args.valin_pool,
        args.valin_pool * args.valin_scan_factor, args.screen_batch,
        args.seed + 999, table_cache, logger)
    logger.log("")

    np.savez(os.path.join(out_dir, "rules.npz"),
             train_rules=np.array(train_rules, dtype=np.int64),
             valin_rules=np.array(valin_rules, dtype=np.int64),
             valout_rules=np.array(valout_rules, dtype=np.int64),
             test_rules=np.array(test_rules, dtype=np.int64))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({"V": V, "K": K, "L": args.L, "T": args.T, "M": args.M,
                   "num_train": args.num_train, "num_eval": args.num_eval,
                   "seed": args.seed, "vocab": VOCAB, "equivalence": "wolfram",
                   "n_train_rules": len(train_rules), "n_valin_rules": len(valin_rules),
                   "n_valout_rules": len(valout_rules), "n_test_rules": len(test_rules)}, f, indent=2)
    logger.close()
    return out_dir, dict(train_rules=train_rules, valin_rules=valin_rules,
                         valout_rules=valout_rules, test_rules=test_rules)


def split_specs(args, pools):
    return [
        ("train",           pools["train_rules"],  args.num_train, args.seed,       False, "uniform", True),
        ("val_in_nofilter", pools["train_rules"],  args.num_eval,  args.seed + 50,  False, "uniform", False),
        ("val_in_filter",   pools["valin_rules"],  args.num_eval,  args.seed + 60,  True,  "uniform", False),
        ("val_out",         pools["valout_rules"], args.num_eval,  args.seed + 100, True,  "uniform", False),
        ("test",            pools["test_rules"],   args.num_eval,  args.seed + 150, True,  "uniform", False),
    ]


def worker_share(total, w, n):
    base, rem = divmod(total, n)
    return base + (1 if w < rem else 0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, default=16)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--num_train", type=int, default=50_000_000)
    p.add_argument("--num_eval", type=int, default=500_000)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--valout_frac", type=float, default=0.1,
                   help="held-out canonical fraction per Wolfram class for val_out "
                        "(test gets the same); train gets the rest")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--valin_pool", type=int, default=20000, help="target coverable train rules for val_in_filter")
    p.add_argument("--valin_scan_factor", type=int, default=8, help="scan up to valin_pool*this train rules")
    p.add_argument("--screen_batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", type=str, default="data")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--prepare", action="store_true", help="run split+screen, write rules.npz, print OUTPUT_DIR")
    p.add_argument("--worker_id", type=int, default=None, help="this worker's index (0..num_workers-1)")
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--splits", type=str, nargs="+", default=None,
                   choices=["train", "val_in_nofilter", "val_in_filter", "val_out", "test"],
                   help="Only generate this subset of splits (worker mode). Default: all 5. "
                        "Used to regenerate specific eval splits after a partial failure "
                        "without touching the (huge) train shards.")
    args = p.parse_args()
    tpl = build_token_template(args.L, args.T)

    if args.prepare:
        out_dir, _ = do_prepare(args)
        print(f"OUTPUT_DIR={out_dir}")
        return

    if args.worker_id is not None:
        if not args.output_dir:
            raise SystemExit("worker mode requires --output_dir (from `--prepare`)")
        W, N = args.worker_id, args.num_workers
        rz = np.load(os.path.join(args.output_dir, "rules.npz"))
        pools = {k: rz[k].tolist() for k in ("train_rules", "valin_rules", "valout_rules", "test_rules")}
        logger = Logger(os.path.join(args.output_dir, f"stats_w{W}.txt"))
        logger.log(f"=== WORKER {W}/{N}  dir={args.output_dir} ===")
        table_cache = {}
        wanted_splits = set(args.splits) if args.splits else None
        for name, rules, n_total, sd, raq, pick, sharded in split_specs(args, pools):
            if wanted_splits is not None and name not in wanted_splits:
                continue
            n = worker_share(n_total, W, N)
            if n == 0:
                continue
            logger.log(f"\n-- {name} worker {W}: {n}/{n_total} samples, {len(rules)} rules --")
            generate_split(name, rules, n, args.output_dir, args.L, args.T,
                           args.M, sd + W * 100003, raq, pick, f"_w{W}",
                           args.batch_size, tpl, logger, table_cache)
        logger.log(f"\n=== WORKER {W} DONE ===")
        logger.close()
        return

    # single-process smoke
    out_dir, pools = do_prepare(args)
    logger = Logger(os.path.join(out_dir, "stats.txt"), mode="a")
    logger.log("\n=== SAMPLE GENERATION (single process) ===")
    table_cache = {}
    for name, rules, n_total, sd, raq, pick, sharded in split_specs(args, pools):
        logger.log(f"\n-- {name}: {n_total} samples, {len(rules)} rules --")
        generate_split(name, rules, n_total, out_dir, args.L, args.T,
                       args.M, sd, raq, pick, "",
                       args.batch_size, tpl, logger, table_cache)
    logger.log(f"\n=== DONE === output_dir={out_dir}")
    logger.close()


if __name__ == "__main__":
    main()
