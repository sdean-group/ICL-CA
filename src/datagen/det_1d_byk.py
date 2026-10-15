"""1D V=3 CA data generator — histogram-stratified, S3-only equivalence, held-out-rule OOD.

V=3 / K=3 analogue of data_generate_2d_byk.py (which is V=2, K=5, Von Neumann).
Same five-split byk architecture; only the per-cell math and the two folding
mechanisms change:

  * Cells       : 1D ring of length L, states in {0,1,2}. Neighborhood = (left,
                  center, right), offsets (-1,0,1). config idx = L*9 + C*3 + R,
                  so NUM_CONFIGS = 3^3 = 27, NUM_RULES = 3^27 (fits int64).
  * Equivalence : ONLY S3 colour permutations (6). The V=3 analogue of the V=2
                  "complement" (0<->1 swap). NO spatial reflection (byk dropped
                  spatial symmetry deliberately). canonical = min over the 6-orbit.
                  Each canonical -> up to 6 rules (expand_pool dedups stabilisers).
  * Stratify    : by the SORTED output histogram of the 27-entry rule table,
                  (n_a >= n_b >= n_c), n_a+n_b+n_c = 27. This is S3-invariant and
                  is the exact V=3 generalisation of fold-k (which in V=2 was the
                  sorted {popcount, 32-popcount}). ~75 strata; entropy = H(hist/27).
  * Quotas      : counted in CANONICAL rules per stratum. Low-entropy strata are
                  intrinsically scarce (near-constant rules cannot cover all 27
                  configs); val_out/test each get min(vt_quota, #passers//2) per
                  stratum so the two stay symmetric (same scarce-k logic as byk).
  * Five splits :
        train             train_rules   require_all_patterns=False  mask=all-1   pick=uniform
        val_in_nofilter   train_rules   require_all_patterns=False  mask=all-1   pick=uniform
        val_in_filter     valin_rules   require_all_patterns=True   mask=step>=M pick=uniform
        val_out           valout_rules  require_all_patterns=True   mask=step>=M pick=hist-balanced
        test              test_rules    require_all_patterns=True   mask=step>=M pick=hist-balanced
    valin_rules = the coverable subset of train_rules (PRE-SCREENED).
  * Pools disjoint at canonical level (train vs valout vs test).

Performance / parallelism: identical to byk (batched multi-rule simulation,
--prepare once then --worker_id array tasks, int8 tokens / uint8 masks).

Two-step parallel run:
    python -m src.datagen.det_1d_byk --prepare --L 32 --T 12 --M 8 \
        --num_train 50000000 --num_eval 500000 --train_quota 11765 --vt_quota 400 --seed 42
    python -m src.datagen.det_1d_byk --worker_id $TASK --num_workers 64 \
        --output_dir <DIR from step 1> <same grid/sample args>

Single-process smoke test: omit --prepare/--worker_id.
"""

import os
import json
import argparse
import random
from math import comb, log2
from itertools import combinations, permutations
from collections import Counter, defaultdict

import numpy as np
from numpy.lib.format import open_memmap


# ============================================================
# Constants — 1D, V=3, K=3, offsets (-1,0,1), big-endian (L,C,R)
# config idx = L*9 + C*3 + R
# ============================================================
V = 3
K = 3
NUM_CONFIGS = V ** K          # 27
# NUM_RULES = V ** NUM_CONFIGS = 3^27 = 7_625_597_484_987  (fits int64)

# tokens: states 0,1,2 ; SEP between time rows = V
TIME_SEP = V                  # 3
VOCAB = V + 1                 # {0,1,2,SEP}

POW3 = np.array([V ** i for i in range(NUM_CONFIGS)], dtype=np.int64)
H_MAX = log2(V)               # 1.5849625... max output entropy (bits)


# ============================================================
# Rule <-> table (base-3, little-endian: table[i] = output for config i)
# ============================================================

def rule_table(rule_int):
    t = np.empty(NUM_CONFIGS, dtype=np.int8)
    r = int(rule_int)
    for i in range(NUM_CONFIGS):
        t[i] = r % V
        r //= V
    return t


def table_to_rule(table):
    return int(np.dot(table.astype(np.int64), POW3))


# ============================================================
# S3 colour-permutation equivalence (no reflection)
#   For state_perm p (symbol s -> p[s]) with inverse inv:
#   new_table[cfg] = p[ table[ map_idx_p[cfg] ] ]
#   where map_idx_p[cfg] = index of (inv applied to each cell L,C,R of cfg)
# ============================================================

PERMS = list(permutations(range(V)))          # 6 tuples


def _cfg_digits(cfg):
    return cfg // 9, (cfg // 3) % 3, cfg % 3   # (L, C, R)


def _build_perm_maps():
    map_idx, out_relabel = [], []
    for p in PERMS:
        inv = [0] * V
        for s, ps in enumerate(p):
            inv[ps] = s
        mi = np.empty(NUM_CONFIGS, dtype=np.int64)
        for cfg in range(NUM_CONFIGS):
            L, C, R = _cfg_digits(cfg)
            mi[cfg] = inv[L] * 9 + inv[C] * 3 + inv[R]
        map_idx.append(mi)
        out_relabel.append(np.array(p, dtype=np.int8))
    return map_idx, out_relabel


MAP_IDX, OUT_RELABEL = _build_perm_maps()


def _transform_table(table, pi):
    return OUT_RELABEL[pi][table[MAP_IDX[pi]]]


def canonical_rule(rule_int):
    tbl = rule_table(rule_int)
    best = int(rule_int)
    for pi in range(len(PERMS)):
        ri = table_to_rule(_transform_table(tbl, pi))
        if ri < best:
            best = ri
    return best


def orbit(rule_int):
    tbl = rule_table(rule_int)
    return {table_to_rule(_transform_table(tbl, pi)) for pi in range(len(PERMS))}


def expand_pool(canon_list):
    rules = set()
    for c in canon_list:
        rules |= orbit(c)
    return sorted(rules)


# ============================================================
# Output histogram stratum (S3-invariant) + entropy
# ============================================================

def hist_stratum(rule_int):
    """Sorted (desc) output-symbol histogram of the rule table -> stratum key."""
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


def all_strata():
    """All sorted histograms (a>=b>=c>=0, a+b+c=27)."""
    out = []
    for a in range(NUM_CONFIGS, -1, -1):
        for b in range(min(a, NUM_CONFIGS - a), -1, -1):
            c = NUM_CONFIGS - a - b
            if c <= b:
                out.append((a, b, c))
    return out


def stratum_avail(strat):
    """Number of labelled assignments (a configs of one symbol, b of another, c
    of the third) up to which-symbol choice. Upper bound used for enum/sample
    cutoff; canonicalisation only shrinks it."""
    a, b, c = strat
    return comb(NUM_CONFIGS, a) * comb(NUM_CONFIGS - a, b)


# ============================================================
# Vectorized 1D simulation (base-3 ring)
# ============================================================

def _cfg_idx(g):
    """Neighborhood-config index per cell. g: (..., L). idx = L*9 + C*3 + R."""
    Ln = np.roll(g, 1, axis=-1)     # left  (i-1)
    C = g
    Rn = np.roll(g, -1, axis=-1)    # right (i+1)
    return (Ln * 3 + C) * 3 + Rn


def simulate_batch(B, table, num_steps, L, rng):
    """B trajectories under ONE rule. Returns (B,num_steps,L) int8."""
    states = np.empty((B, num_steps, L), dtype=np.int8)
    states[:, 0] = rng.randint(0, V, size=(B, L)).astype(np.int8)
    for t in range(1, num_steps):
        idx = _cfg_idx(states[:, t - 1].astype(np.int64))
        states[:, t] = table[idx]
    return states


def simulate_batch_multi(tables, num_steps, L, rng):
    """B trajectories, EACH UNDER ITS OWN RULE. tables (B,27) int8.
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
    """Per-trajectory: do the first M-1 rows cover all 27 configs? states (B,>=M-1,L)."""
    B = states.shape[0]
    ctx = states[:, :M - 1].astype(np.int64)        # (B, M-1, L)
    flat = _cfg_idx(ctx).reshape(B, -1)
    pres = np.zeros((B, NUM_CONFIGS), dtype=bool)
    pres[np.arange(B)[:, None], flat] = True
    return pres.all(axis=1)


def rule_can_cover(rule_int, L, M, num_tries=30, min_successes=1, rng=None):
    if rng is None:
        rng = np.random.RandomState(int(rule_int) % (2 ** 31))
    table = rule_table(rule_int)
    states = simulate_batch(num_tries, table, M - 1, L, rng)
    return int(coverage_batch(states, M).sum()) >= min_successes


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
    """states (B,T,L) int8 -> inp (B,seq) int8, tgt (B,seq) int8, mask_row (seq,) uint8."""
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
# Phase 2 — rule selection (per stratum, canonical, order valout->test->train)
# ============================================================

def _table_from_assignment(s0, s1):
    """configs in s0 -> 0, in s1 -> 1, rest -> 2."""
    t = np.full(NUM_CONFIGS, 2, dtype=np.int8)
    t[list(s0)] = 0
    t[list(s1)] = 1
    return t


def _enumerate_stratum(strat):
    """All canonical rules with this exact sorted histogram (small strata only)."""
    a, b, _c = strat
    seen = set()
    allcfg = range(NUM_CONFIGS)
    for s0 in combinations(allcfg, a):
        rem = [x for x in allcfg if x not in s0]
        for s1 in combinations(rem, b):
            seen.add(canonical_rule(table_to_rule(_table_from_assignment(s0, s1))))
    return seen


def _sample_stratum_rule(strat, pyr):
    """Random canonical rule with the given sorted histogram."""
    a, b, _c = strat
    cfgs = list(range(NUM_CONFIGS))
    pyr.shuffle(cfgs)
    return canonical_rule(table_to_rule(_table_from_assignment(cfgs[:a], cfgs[a:a + b])))


def select_rules(L, M, train_quota, vt_quota, enum_threshold,
                 num_tries, min_successes, seed, entropy_prec=2):
    """train_quota / vt_quota are PER DISTINCT ENTROPY VALUE (rounded to
    entropy_prec decimals). An entropy value realised by m histograms splits its
    budget evenly -> each histogram gets quota/m. This makes the pool (and hence
    uniform-sampled samples) flat over discrete entropy; scarce low-entropy values
    fall short by availability (no forced repetition)."""
    pyr = random.Random(seed)
    filter_rng = np.random.RandomState(seed + 1)
    strata = all_strata()
    groups = defaultdict(list)
    for s in strata:
        groups[round(stratum_entropy(s), entropy_prec)].append(s)
    gsize = {s: len(groups[round(stratum_entropy(s), entropy_prec)]) for s in strata}
    train_by_s, valout_by_s, test_by_s, stats = {}, {}, {}, []

    for strat in strata:
        avail = stratum_avail(strat)
        m = gsize[strat]
        t_quota = max(1, round(train_quota / m))     # per-histogram share of the entropy level
        v_quota = max(1, round(vt_quota / m))
        n_tested = 0

        if avail <= enum_threshold:
            all_canon = sorted(_enumerate_stratum(strat))
            passers = [c for c in all_canon
                       if rule_can_cover(c, L, M, num_tries, min_successes, filter_rng)]
            n_tested = len(all_canon)
            pyr.shuffle(passers)
            each = min(v_quota, len(passers) // 2)
            valout, test = passers[:each], passers[each:2 * each]
            used = set(valout) | set(test)
            train_pool = [c for c in all_canon if c not in used]
            pyr.shuffle(train_pool)
            train = train_pool[:t_quota]
        else:
            seen, passers = set(), []
            target_pass = 2 * v_quota
            max_attempts = target_pass * 300 + 2000
            for _ in range(max_attempts):
                if len(passers) >= target_pass:
                    break
                canon = _sample_stratum_rule(strat, pyr)
                if canon in seen:
                    continue
                seen.add(canon); n_tested += 1
                if rule_can_cover(canon, L, M, num_tries, min_successes, filter_rng):
                    passers.append(canon)
            each = min(v_quota, len(passers) // 2)
            valout, test = passers[:each], passers[each:2 * each]
            train_seen = set(valout) | set(test) | set(passers)
            train = []
            t_max = t_quota * 50 + 5000
            for _ in range(t_max):
                if len(train) >= t_quota:
                    break
                canon = _sample_stratum_rule(strat, pyr)
                if canon in train_seen:
                    continue
                train_seen.add(canon); train.append(canon)

        train_by_s[strat], valout_by_s[strat], test_by_s[strat] = train, valout, test
        stats.append({"strat": strat, "H": stratum_entropy(strat), "Hg": round(stratum_entropy(strat), entropy_prec),
                      "m": m, "avail": avail, "tested": n_tested, "train_canon": len(train),
                      "valout_canon": len(valout), "test_canon": len(test)})
    return train_by_s, valout_by_s, test_by_s, stats


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
    # entropy-binned coverage of the sample stream
    ebins = defaultdict(int)
    for st, cc in strat_counter.items():
        ebins[round(stratum_entropy(st), 1)] += cc
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
    voto = len(valout_canon) + len(test_canon)
    return os.path.join(
        args.output_root,
        f"V{V}K{K}_L{args.L}_T{args.T}_M{args.M}"
        f"_train{count_str(len(train_canon))}_voto{count_str(voto)}"
        f"_n{count_str(args.num_train)}_test{count_str(args.num_eval)}_seed{args.seed}")


def do_prepare(args, logger_path_override=None):
    print(f"=== PREPARE: selection (by output histogram, S3-only) ===")
    train_by_s, valout_by_s, test_by_s, stats = select_rules(
        args.L, args.M, args.train_quota, args.vt_quota,
        args.enum_threshold, args.num_tries, args.min_successes, args.seed,
        args.entropy_prec)
    train_canon = [c for s in train_by_s for c in train_by_s[s]]
    valout_canon = [c for s in valout_by_s for c in valout_by_s[s]]
    test_canon = [c for s in test_by_s for c in test_by_s[s]]

    out_dir = resolve_output_dir(args, train_canon, valout_canon, test_canon)
    os.makedirs(out_dir, exist_ok=True)
    logger = Logger(logger_path_override or os.path.join(out_dir, "stats.txt"))

    logger.log("=== CONFIG ===")
    logger.log(f"seed={args.seed}  L={args.L}  T={args.T}  M={args.M}  V={V}  K={K}  offsets=(-1,0,1)")
    logger.log(f"enum/sample boundary avail<= {args.enum_threshold}; filter=({args.num_tries},{args.min_successes})")
    logger.log(f"seq_len={compute_seq_len(args.L,args.T)}  vocab={VOCAB}")
    logger.log(f"samples: train={args.num_train}, each eval={args.num_eval}; batch_size={args.batch_size}")
    logger.log(f"quotas (canonical): train={args.train_quota}/entropy-level, val_out=test={args.vt_quota}/entropy-level each")
    logger.log(f"entropy grouping: round H to {args.entropy_prec} decimals; a level with m histograms splits quota/m")
    logger.log(f"equivalence: S3 colour perms only; each canonical -> up to 6 rules\noutput_dir={out_dir}\n")

    logger.log("=== RULE STATS (per output-histogram stratum; canonical counts) ===")
    logger.log(f"{'stratum':>14} {'H':>6} {'Hlvl':>6} {'m':>3} {'avail':>16} {'tested':>9} {'tr_can':>8} {'vo_can':>7} {'te_can':>7}")
    tot = {"tr": 0, "vo": 0, "te": 0}
    for s in stats:
        logger.log(f"{str(s['strat']):>14} {s['H']:>6.3f} {s['Hg']:>6.2f} {s['m']:>3} {s['avail']:>16,} {s['tested']:>9} "
                   f"{s['train_canon']:>8} {s['valout_canon']:>7} {s['test_canon']:>7}")
        tot["tr"] += s["train_canon"]; tot["vo"] += s["valout_canon"]; tot["te"] += s["test_canon"]
    logger.log(f"{'SUM':>14} {'':>6} {'':>16} {'':>9} {tot['tr']:>8} {tot['vo']:>7} {tot['te']:>7}\n")

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
                   "seed": args.seed, "vocab": VOCAB,
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
    p.add_argument("--L", type=int, default=32)
    p.add_argument("--T", type=int, default=12)
    p.add_argument("--M", type=int, default=8)
    p.add_argument("--num_train", type=int, default=50_000_000)
    p.add_argument("--num_eval", type=int, default=500_000)
    p.add_argument("--train_quota", type=int, default=11765)
    p.add_argument("--vt_quota", type=int, default=400)
    p.add_argument("--enum_threshold", type=int, default=100_000,
                   help="enumerate strata with avail<=this, sample the rest")
    p.add_argument("--entropy_prec", type=int, default=2,
                   help="round H to this many decimals to define entropy levels; "
                        "a level with m histograms splits its quota across them (flat over entropy)")
    p.add_argument("--num_tries", type=int, default=30)
    p.add_argument("--min_successes", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--valin_pool", type=int, default=20000, help="target coverable train rules for val_in_filter")
    p.add_argument("--valin_scan_factor", type=int, default=8, help="scan up to valin_pool*this train rules")
    p.add_argument("--screen_batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_root", type=str, default="data")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--prepare", action="store_true", help="run selection+screen, write rules.npz, print OUTPUT_DIR")
    p.add_argument("--worker_id", type=int, default=None, help="this worker's index (0..num_workers-1)")
    p.add_argument("--num_workers", type=int, default=1)
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
        for name, rules, n_total, sd, raq, pick, sharded in split_specs(args, pools):
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
