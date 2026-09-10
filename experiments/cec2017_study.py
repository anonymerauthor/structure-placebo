"""CEC2017 replication of the structure-placebo findings.

Protocol follows the CEC2017 technical report: error f(x) - f*, budget
10000*D, box [-100, 100]^D. F2 is omitted, as the organisers advise, for
unstable behaviour in higher dimensions -- leaving 29 functions.

The six arms are the same ones the classic-benchmark study used, so the
question is whether the placebo results replicate on a modern suite:

  social      published configuration (WS graph, betweenness weighting)
  shuffled    betweenness values permuted across nodes  [control 1]
  uniform_K4  same graph, no centrality weighting
  fixed_K16   uniform mixing at a fixed, denser neighbourhood
  random_K    density cycled per epoch, reward ignored   [control 2]
  ads         density chosen per epoch by sliding-window UCB
"""
import os, sys, json, argparse, time, functools
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from social.config import Config
from social.optimizer import SOCIALOptimizer
from social.adaptive import AdaptiveDensityOptimizer, make_adaptive_config
from social.budget import BudgetedObjective

# F2 is excluded per the CEC2017 errata (unstable in higher dimensions).
# F30 is not provided by opfunu 1.0.4, leaving 28 of the suite's 30 functions.
CEC_IDS = [1] + list(range(3, 30))
VARIANTS = ["social", "shuffled", "uniform_K4", "fixed_K16", "random_K", "ads",
            "shuffled_inf"]
OUTDIR = "outputs/cec2017"
EVALS_PER_DIM = 10_000


@functools.lru_cache(maxsize=None)
def get_problem(fid, dim):
    """Cached per worker: every construction re-reads the suite's shift and
    rotation tables, and each (fid, dim) is reused across all 30 seeds."""
    import opfunu
    cls = getattr(opfunu.cec_based.cec2017, f"F{fid}2017")
    return cls(ndim=dim)


def build(variant, nodes, dim, seed):
    if variant in ("social", "shuffled", "shuffled_inf"):
        cfg = Config()
        cfg.NUM_NODES, cfg.DIM, cfg.VECTORIZED, cfg.SHOW_PROGRESS = nodes, dim, True, False
        if variant == "shuffled":
            cfg.CENTRALITY_MODE = "shuffled_betweenness"
        elif variant == "shuffled_inf":
            # Positive control: permute the fitness-influence score instead.
            cfg.INFLUENCE_MODE = "shuffled_rank"
        return SOCIALOptimizer(cfg, np.random.default_rng(seed))
    cfg = make_adaptive_config(nodes, dim)
    if variant.startswith(("uniform_K", "fixed_K")):
        cfg.K = int(variant.split("K")[1])
        return SOCIALOptimizer(cfg, np.random.default_rng(seed))
    return AdaptiveDensityOptimizer(cfg, np.random.default_rng(seed),
                                    policy="random" if variant == "random_K" else "ucb")


def run_one(variant, fid, dim, seed, nodes):
    path = os.path.join(OUTDIR, f"{variant}_F{fid}_D{dim}_S{seed}.json")
    if os.path.exists(path):
        return
    prob = get_problem(fid, dim)
    bounds = [float(prob.lb[0]), float(prob.ub[0])]
    opt = build(variant, nodes, dim, seed)
    t0 = time.perf_counter()
    res = opt.optimize(BudgetedObjective(prob.evaluate, EVALS_PER_DIM * dim),
                       bounds, dim, seed=seed)
    rec = dict(variant=variant, fid=fid, dim=dim, seed=seed,
               error=float(res["best_fitness"] - prob.f_global),
               seconds=round(time.perf_counter() - t0, 3))
    json.dump(rec, open(path, "w"))


def run_serial(jobs, nodes, tag=""):
    """Execute this shard's jobs in-process.

    Parallelism is obtained by launching several sharded processes rather than
    a worker pool: loky's Windows spawn path fails on this machine with
    WinError 5 from DuplicateHandle, and each run already checkpoints to its
    own file, so splitting the job list across processes is equivalent.
    """
    t0 = time.perf_counter()
    for i, (v, f, d, sd) in enumerate(jobs, 1):
        try:
            run_one(v, f, d, sd, nodes)
        except Exception as exc:                     # keep the shard alive
            print(f"{tag}  FAILED {v} F{f} D{d} S{sd}: {type(exc).__name__}: {exc}",
                  flush=True)
        if i % 25 == 0 or i == len(jobs):
            print(f"{tag}  {i}/{len(jobs)} done, {(time.perf_counter()-t0)/60:.1f} min",
                  flush=True)


_T0 = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=60)
    ap.add_argument("--dims", type=int, nargs="*", default=[30])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="outputs/cec2017.csv")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    jobs = [(v, f, d, s) for v in a.variants for f in CEC_IDS
            for d in a.dims for s in range(a.runs)]
    jobs = jobs[a.shard::a.nshards]          # interleaved so shards finish together
    print(f"{len(jobs)} runs | {len(CEC_IDS)} functions | dims={a.dims} | "
          f"budget=10000*D | {len(a.variants)} variants | jobs={a.jobs}", flush=True)
    t0 = time.perf_counter()
    run_serial(jobs, a.nodes, tag=f"[shard {a.shard}/{a.nshards}]")
    print(f"wall={(time.perf_counter()-t0)/60:.1f} min", flush=True)

    if a.shard == 0:
        df = pd.DataFrame([json.load(open(os.path.join(OUTDIR, p)))
                           for p in os.listdir(OUTDIR)])
        df.to_csv(a.out, index=False)
        print(f"{len(df)} runs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
