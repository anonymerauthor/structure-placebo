"""Does SOCIAL's structure actually inform the search, or only its weight spread?

Six variants share every hyperparameter, budget and seed; they differ only in
where the neighbour weights come from and what graph carries them.

  social      published configuration: WS graph, betweenness weights
  shuffled    WS graph, betweenness values randomly permuted across nodes
              -> identical weight distribution, no structure/position link
  uniform     WS graph, no centrality weighting (plain neighbour mean)
  degree      WS graph, degree centrality (cheap structural proxy)
  ring        p=0 lattice: no small-world shortcuts (betweenness is constant)
  complete    fully connected: no topology at all

'social' beating 'shuffled' is the only evidence that betweenness carries
search-relevant information. 'social' == 'shuffled' means the mechanism is a
placebo dressed as structure.
"""
import os, sys, argparse, time
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

VARIANTS = {
    "social":   dict(CENTRALITY_MODE="betweenness",          NEIGHBOR_MODE="centrality_weighted"),
    "shuffled": dict(CENTRALITY_MODE="shuffled_betweenness", NEIGHBOR_MODE="centrality_weighted"),
    "uniform":  dict(CENTRALITY_MODE="betweenness",          NEIGHBOR_MODE="uniform"),
    "degree":   dict(CENTRALITY_MODE="degree",               NEIGHBOR_MODE="centrality_weighted"),
    "ring":     dict(CENTRALITY_MODE="betweenness",          NEIGHBOR_MODE="centrality_weighted", P_BASE=0.0),
    "complete": dict(CENTRALITY_MODE="betweenness",          NEIGHBOR_MODE="centrality_weighted", COMPLETE=True),
    # Neighbourhood-size controls: separate "more neighbours" from "no topology".
    "ws_K20":   dict(CENTRALITY_MODE="betweenness",          NEIGHBOR_MODE="centrality_weighted", K=20),
    "ws_K40":   dict(CENTRALITY_MODE="betweenness",          NEIGHBOR_MODE="centrality_weighted", K=40),
}

FUNCTIONS = ["Sphere", "Rosenbrock", "Step", "Schwefel_2_26", "Rastrigin",
             "Ackley", "Griewank", "Penalized", "Quartic", "Schwefel_1_2"]


def run_one(variant, fname, seed, max_evals, n_nodes, dim):
    bf = BenchmarkFunctions()
    func, bounds, _ = bf.functions[fname]
    cfg = Config()
    cfg.NUM_NODES, cfg.SHOW_PROGRESS, cfg.VECTORIZED = n_nodes, False, True
    spec = dict(VARIANTS[variant])
    complete = spec.pop("COMPLETE", False)
    for k, v in spec.items():
        setattr(cfg, k, v)
    if complete:
        cfg.K = n_nodes - 1
        cfg.P_BASE = 0.0
        cfg.REWIRE_MODE = "none"   # a complete graph has no topology left to rewire

    opt = SOCIALOptimizer(cfg, np.random.default_rng(seed))
    t0 = time.perf_counter()
    res = opt.optimize(BudgetedObjective(func, max_evals), list(bounds), dim, seed=seed)
    return dict(variant=variant, function=fname, seed=seed,
                best=res["best_fitness"], evals=res["evals_used"],
                seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--max_evals", type=int, default=20000)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--dim", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="outputs/placebo_study.csv")
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()

    variants = a.variants or list(VARIANTS)
    jobs = [(v, f, s) for v in variants for f in FUNCTIONS for s in range(a.runs)]
    print(f"{len(jobs)} runs | {len(variants)} variants x {len(FUNCTIONS)} functions "
          f"x {a.runs} seeds | budget={a.max_evals} N={a.nodes} D={a.dim} | jobs={a.jobs}")
    print("variants:", ", ".join(variants))
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=a.jobs, backend="loky", verbose=5)(
        delayed(run_one)(v, f, s, a.max_evals, a.nodes, a.dim) for v, f, s in jobs)
    wall = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwall={wall/60:.1f} min -> {a.out}")

    piv = df.pivot_table(index="function", columns="variant", values="best", aggfunc="median")
    piv = piv[[v for v in variants if v in piv.columns]]
    pd.set_option("display.width", 220)
    print("\n=== median best fitness (lower is better) ===")
    print(piv.to_string(float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
