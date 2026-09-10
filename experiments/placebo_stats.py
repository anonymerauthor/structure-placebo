"""Statistics for the placebo study: is the structure doing any work?"""
import sys, argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="outputs/placebo_study.csv")
ap.add_argument("--refs", nargs="*", default=["shuffled", "complete"])
args = ap.parse_args()

df = pd.read_csv(args.csv)
VAR = [v for v in df["variant"].unique()]
print(f"{args.csv}: {len(df)} runs | variants: " + ", ".join(VAR))


def cliffs_delta(a, b):
    """Cliff's delta; >0 means a is larger (worse, for minimization)."""
    n = len(a) * len(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / n


def holm(pvals):
    order = np.argsort(pvals)
    m, adj = len(pvals), np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


print("=== Friedman across all 6 variants (per function, paired by seed) ===")
ranks = []
for f, g in df.groupby("function"):
    piv = g.pivot(index="seed", columns="variant", values="best")[VAR]
    stat, p = friedmanchisquare(*[piv[v].values for v in VAR])
    r = np.apply_along_axis(rankdata, 1, piv.values).mean(axis=0)
    ranks.append(r)
    print(f"  {f:15s} chi2={stat:8.2f} p={p:.2e}   ranks " +
          " ".join(f"{v}={x:.2f}" for v, x in zip(VAR, r)))
mean_ranks = np.mean(ranks, axis=0)
print("\n  MEAN RANK (lower=better): " + "  ".join(f"{v}={x:.3f}" for v, x in zip(VAR, mean_ranks)))

for ref in [r for r in args.refs if r in VAR]:
    print(f"\n=== social vs {ref} : paired Wilcoxon per function (30 seeds) ===")
    rows = []
    for f, g in df.groupby("function"):
        piv = g.pivot(index="seed", columns="variant", values="best")
        a, b = piv["social"].values, piv[ref].values
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0
        rows.append(dict(function=f, social=np.median(a), other=np.median(b),
                         delta=cliffs_delta(a, b), p=p))
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].values)
    t["verdict"] = np.where(t.p_holm >= 0.05, "ns",
                     np.where(t.social < t.other, "social WINS", f"{ref} WINS"))
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    w = (t.verdict == "social WINS").sum(); l = (t.verdict == f"{ref} WINS").sum()
    print(f"  --> social wins {w}, {ref} wins {l}, ties {len(t)-w-l}  (Holm-corrected, alpha=0.05)")
