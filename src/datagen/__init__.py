"""The dataset generators used for the paper's six tasks (unchanged code).

Training does NOT need these: `src.train` simulates data on the fly
(`src/data/streaming.py`), and the rule splits they produce ship in
src/assets/rule_pools.npz. Use them to rebuild the exact on-disk datasets.

Named by how rules are chosen, not by V/K (V is a flag for the stochastic
generators; the deterministic ones set V/K as module constants):

    module                         rules                                       original script
    src.datagen.det_1d_wolfram     elementary CA, split by Wolfram class       data_generate_1d.py
    src.datagen.det_1d_byk         stratified by output entropy (paper: V=3)   data_generate_1d_byk.py
    src.datagen.det_2d_byk         von Neumann, stratified by fold-k (V=2)     data_generate_2d_byk.py
    src.datagen.stoch_1d           Dirichlet(1,...,1) tables, any V, K=3       data_generate_1d_stoch.py
    src.datagen.stoch_2d           Dirichlet(1,...,1) tables, any V, K=5 (VN)  data_generate_2d_stoch.py

Every released dataset used the generator's defaults and seed 42:

    python -m src.datagen.det_1d_wolfram --L 16 --T 10 --M 4 --seed 42
    python -m src.datagen.det_1d_byk --L 32 --T 12 --M 8 --seed 42
    python -m src.datagen.det_2d_byk --L1 6 --L2 6 --T 12 --M 8 --seed 42
    python -m src.datagen.stoch_1d --V 2 --L 16 --T 20 --M 10 --num_eval_test 500000 --seed 42
    python -m src.datagen.stoch_1d --V 3 --L 16 --T 20 --M 10 --num_eval_test 500000 --seed 42
    python -m src.datagen.stoch_2d --V 2 --L1 6 --L2 6 --T 20 --M 10 --num_eval_test 500000 --seed 42

(50M train + 500k per eval split each; 15-130 GB per dataset.) For speed the
originals ran `--prepare` once and then a Slurm array of
`--worker_id i --num_workers N` processes over the same --output_dir; see each
module's docstring. Deterministic generators write rules.npz with the
train/valin/valout/test rule splits; `--prepare` alone (seconds to minutes)
reproduces exactly the pools in src/assets/rule_pools.npz.
"""
