"""Which neighbourhood density wins, and can it be predicted from the landscape?

The placebo study showed centrality weighting is inert but neighbourhood
density is not. This asks whether the best density is predictable from a
quantity an optimizer can measure online -- a prerequisite for adapting
density during the search instead of fixing it by a Watts-Strogatz prior.

Ruggedness proxy: lag-1 autocorrelation of fitness along a random walk.
Needs no known optimum, so it is computable during a real run.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from social.functions import BenchmarkFunctions

# Neighbourhood degree behind each variant in placebo_105k.csv
K_OF = {"social": 5, "shuffled": 5, "ring": 5, "ws_K20": 20, "ws_K40": 40, "complete": 59}


def ruggedness(func, bounds, dim, n_steps=4000, seed=0):
    """Lag-1 autocorrelation of fitness along a random walk (higher = smoother)."""
    rng = np.random.default_rng(seed)
    step = 0.02 * (bounds[1] - bounds[0])
    x = rng.uniform(bounds[0], bounds[1], dim)
    vals = np.empty(n_steps)
    for i in range(n_steps):
        vals[i] = func(x)
        x = np.clip(x + rng.normal(0, step, dim), bounds[0], bounds[1])
    v = (vals - vals.mean()) / (vals.std() + 1e-30)
    return float(np.mean(v[:-1] * v[1:]))


def main():
    df = pd.read_csv("outputs/placebo_105k.csv")
    df["K"] = df["variant"].map(K_OF)
    bf = BenchmarkFunctions()

    rows = []
    for f, g in df.groupby("function"):
        # Per-seed ranking across K levels, averaged: robust to scale differences
        piv = g.pivot_table(index="seed", columns="K", values="best", aggfunc="mean")
        Ks = list(piv.columns)
        if piv.isna().any().any() or piv.shape[0] < 5:
            continue
        r = np.apply_along_axis(rankdata, 1, piv.values).mean(axis=0)
        best_K = Ks[int(np.argmin(r))]
        # signed preference: negative = sparse better, positive = dense better
        pref = r[0] - r[-1]          # rank(K=5) - rank(K=59)
        func, bounds, _ = bf.functions[f]
        rug = ruggedness(func, bounds, 30)
        rows.append(dict(function=f, best_K=best_K, dense_pref=pref,
                         autocorr=rug, **{f"rank_K{k}": v for k, v in zip(Ks, r)}))

    t = pd.DataFrame(rows).sort_values("dense_pref")
    pd.set_option("display.width", 220)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    ok = t.dense_pref.abs() > 1e-9
    rho, p = spearmanr(t.autocorr[ok], t.dense_pref[ok])
    print(f"\nSpearman(landscape autocorrelation, preference for dense) "
          f"rho={rho:.3f}  p={p:.4f}  n={ok.sum()}")
    print("negative rho => rugged (low autocorr) landscapes prefer DENSE neighbourhoods")
    t.to_csv("outputs/density_law.csv", index=False)


if __name__ == "__main__":
    main()
