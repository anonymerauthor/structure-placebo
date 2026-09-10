"""Second audit target: degree-weighted FIPS on a scale-free topology.

Same protocol as the first target, same statistic, same seeds. The arms are

    degree     the published structural variant, w_n = deg(n)
    permuted   the same degree multiset, permuted across nodes   [control]
    uniform    equal weights, the ablation

A second target matters because a single null tells you about one algorithm;
two targets tell you whether the protocol distinguishes between them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "audit"))
sys.path.insert(0, HERE)

from budget import Budget          # noqa: E402
from fips import FIPS              # noqa: E402

CEC_IDS = [1] + list(range(3, 30))          # F2 per errata, F30 unavailable
ARMS = ["degree", "permuted", "uniform"]
OUTDIR = os.path.join(HERE, "outputs", "fips")
EVALS_PER_DIM = 10_000


def get_problem(fid: int, dim: int):
    import opfunu
    return getattr(opfunu.cec_based.cec2017, f"F{fid}2017")(ndim=dim)


def run_one(arm: str, fid: int, dim: int, seed: int, n: int = 60) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{arm}_F{fid}_D{dim}_S{seed}.json")
    if os.path.exists(path):
        return
    prob = get_problem(fid, dim)
    opt = FIPS(n_particles=n, m_attach=2, weighting=arm,
               rng=np.random.default_rng(seed))
    t0 = time.perf_counter()
    res = opt.optimize(Budget(prob.evaluate, EVALS_PER_DIM * dim),
                       [float(prob.lb[0]), float(prob.ub[0])], dim, seed=seed)
    json.dump(dict(variant=arm, fid=fid, dim=dim, seed=seed,
                   error=float(res["best_fitness"] - prob.f_global),
                   seconds=round(time.perf_counter() - t0, 3)),
              open(path, "w"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--dims", type=int, nargs="*", default=[30])
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "fips.csv"))
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    jobs = [(v, f, d, s) for v in a.arms for f in CEC_IDS
            for d in a.dims for s in range(a.runs)][a.shard::a.nshards]
    print(f"[shard {a.shard}/{a.nshards}] {len(jobs)} jobs", flush=True)

    t0 = time.perf_counter()
    for i, (v, f, d, s) in enumerate(jobs, 1):
        try:
            run_one(v, f, d, s, a.nodes)
        except Exception as exc:
            print(f"  FAILED {v} F{f} D{d} S{s}: {type(exc).__name__}: {exc}",
                  flush=True)
        if i % 25 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)}, {(time.perf_counter()-t0)/60:.1f} min",
                  flush=True)

    if a.shard == 0:
        rows = [json.load(open(os.path.join(OUTDIR, p)))
                for p in os.listdir(OUTDIR)]
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f"{len(rows)} runs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
