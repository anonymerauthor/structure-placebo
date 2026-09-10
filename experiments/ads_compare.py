"""ADS against fixed densities, the published algorithm, and its own placebo.

random_K is the control for the contribution being claimed here: it switches
density on the same epoch schedule but ignores the reward. ADS must beat it,
otherwise the mechanism is density cycling rather than adaptation.
"""
import os, sys, json, argparse, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from social.config import Config
from social.optimizer import SOCIALOptimizer
from social.adaptive import AdaptiveDensityOptimizer, make_adaptive_config
from social.budget import BudgetedObjective
from social.functions import BenchmarkFunctions

FUNCTIONS = ["Sphere", "Schwefel_2_22", "Schwefel_1_2", "Schwefel_2_21", "Rosenbrock",
             "Step", "Quartic", "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank",
             "Penalized", "Penalized2"]
VARIANTS = ["social", "uniform_K4", "fixed_K16", "fixed_K59", "random_K", "ads"]
OUTDIR = "outputs/ads"


def build(variant, nodes, dim, seed):
    if variant == "social":
        cfg = Config()
        cfg.NUM_NODES, cfg.DIM, cfg.VECTORIZED, cfg.SHOW_PROGRESS = nodes, dim, True, False
        return SOCIALOptimizer(cfg, np.random.default_rng(seed))
    cfg = make_adaptive_config(nodes, dim)
    if variant.startswith(("uniform_K", "fixed_K")):
        K = int(variant.split("K")[1])
        cfg.K = K
        if K >= nodes - 1:
            cfg.K, cfg.P_BASE = nodes - 1, 0.0
        return SOCIALOptimizer(cfg, np.random.default_rng(seed))
    policy = "random" if variant == "random_K" else "ucb"
    return AdaptiveDensityOptimizer(cfg, np.random.default_rng(seed), policy=policy)


def run_one(variant, fname, seed, nodes, dim, evals):
    path = os.path.join(OUTDIR, f"{variant}_{fname}_S{seed}.json")
    if os.path.exists(path):
        return
    func, bounds, _ = BenchmarkFunctions().functions[fname]
    opt = build(variant, nodes, dim, seed)
    t0 = time.perf_counter()
    res = opt.optimize(BudgetedObjective(func, evals), list(bounds), dim, seed=seed)
    rec = dict(variant=variant, function=fname, seed=seed,
               best=float(res["best_fitness"]), seconds=round(time.perf_counter()-t0, 3))
    json.dump(rec, open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--evals", type=int, default=105000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="outputs/ads_compare.csv")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    jobs = [(v, f, s) for v in VARIANTS for f in FUNCTIONS for s in range(a.runs)]
    print(f"{len(jobs)} runs | {VARIANTS} | D={a.dim} evals={a.evals} jobs={a.jobs}", flush=True)
    t0 = time.perf_counter()
    Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
        delayed(run_one)(v, f, s, a.nodes, a.dim, a.evals) for v, f, s in jobs)
    print(f"wall={(time.perf_counter()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame([json.load(open(os.path.join(OUTDIR, p))) for p in os.listdir(OUTDIR)])
    df.to_csv(a.out, index=False)
    piv = df.pivot_table(index="function", columns="variant", values="best", aggfunc="median")
    pd.set_option("display.width", 220)
    print(piv[[v for v in VARIANTS if v in piv.columns]].to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
