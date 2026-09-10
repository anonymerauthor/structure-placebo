"""Neighbourhood-density sweep at the correct problem dimensions.

Extends the density law beyond the 9 points available from the placebo study.
Two corrections over the repo's own protocol:

  * F14-F23 are fixed-dimension problems whose implementations slice x[:2..6].
    The repo runs them at config.DIM=30, leaving 24-28 coordinates inert.
    Here each is run at its native dimension.
  * Budget scales with dimension (3500*D), matching the paper's 105k at D=30
    instead of giving a 2-D problem the same budget as a 30-D one.

Results are written per run so an interrupted sweep resumes where it stopped.
"""
import os, sys, json, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from social.config import Config
from social.optimizer import SOCIALOptimizer
from social.budget import BudgetedObjective
from social.functions import BenchmarkFunctions

SCALABLE = ["Sphere", "Schwefel_2_22", "Schwefel_1_2", "Schwefel_2_21", "Rosenbrock",
            "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank",
            "Penalized", "Penalized2"]
# Native dimensions, read off the x[:n] slices in social/functions.py
FIXED_DIM = {"Foxholes": 2, "Kowalik": 4, "Camel-Back": 2, "Branin": 2,
             "Goldstein-Price": 2, "Hartman": 3, "Shekel1": 6,
             "Shekel2": 4, "Shekel3": 4, "Shekel4": 4}
K_LEVELS = [4, 8, 16, 32, 59]
EVALS_PER_DIM = 3500
OUTDIR = "outputs/ksweep"


def configs(dims):
    for f in SCALABLE:
        for d in dims:
            yield f, d
    for f, d in FIXED_DIM.items():
        yield f, d


def run_one(fname, dim, K, seed, nodes):
    tag = f"{fname}_D{dim}_K{K}_S{seed}"
    path = os.path.join(OUTDIR, tag + ".json")
    if os.path.exists(path):
        return None
    bf = BenchmarkFunctions()
    func, bounds, _ = bf.functions[fname]
    cfg = Config()
    cfg.NUM_NODES, cfg.SHOW_PROGRESS, cfg.VECTORIZED = nodes, False, True
    cfg.K = K
    if K >= nodes - 1:                     # complete graph: nothing left to rewire
        cfg.K, cfg.P_BASE, cfg.REWIRE_MODE = nodes - 1, 0.0, "none"
    opt = SOCIALOptimizer(cfg, np.random.default_rng(seed))
    t0 = time.perf_counter()
    res = opt.optimize(BudgetedObjective(func, EVALS_PER_DIM * dim), list(bounds), dim, seed=seed)
    rec = dict(function=fname, dim=dim, K=K, seed=seed,
               best=float(res["best_fitness"]), evals=int(res["evals_used"]),
               seconds=round(time.perf_counter() - t0, 3))
    with open(path, "w") as fh:
        json.dump(rec, fh)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--dims", type=int, nargs="*", default=[10, 30, 50])
    ap.add_argument("--out", default="outputs/ksweep.csv")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    jobs = [(f, d, K, s) for f, d in configs(a.dims) for K in K_LEVELS for s in range(a.runs)]
    done = len(os.listdir(OUTDIR))
    print(f"{len(jobs)} runs total ({done} already on disk) | K={K_LEVELS} | "
          f"budget={EVALS_PER_DIM}*D | N={a.nodes} | jobs={a.jobs}", flush=True)

    t0 = time.perf_counter()
    Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
        delayed(run_one)(f, d, K, s, a.nodes) for f, d, K, s in jobs)
    print(f"\nwall={(time.perf_counter()-t0)/60:.1f} min", flush=True)

    rows = [json.load(open(os.path.join(OUTDIR, p))) for p in os.listdir(OUTDIR)]
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} runs -> {a.out}")
    piv = df.pivot_table(index=["function", "dim"], columns="K", values="best", aggfunc="median")
    pd.set_option("display.width", 220)
    print(piv.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
