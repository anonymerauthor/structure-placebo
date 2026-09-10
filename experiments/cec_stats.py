"""Paired statistics for the CEC2017 arms (error = f(x) - f*)."""
import argparse
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

GROUP = {**{i: "unimodal" for i in (1, 2, 3)},
         **{i: "simple multimodal" for i in range(4, 11)},
         **{i: "hybrid" for i in range(11, 21)},
         **{i: "composition" for i in range(21, 31)}}


def cliffs(a, b):
    n = len(a) * len(b)
    return (sum(x > y for x in a for y in b) - sum(x < y for x in a for y in b)) / n


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); r = 0.0
    for k, i in enumerate(o):
        r = max(r, (m - k) * p[i]); adj[i] = min(r, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/cec2017_placebo.csv")
    ap.add_argument("--left", default="social")
    ap.add_argument("--refs", nargs="*", default=["shuffled"])
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    variants = sorted(df.variant.unique())
    print(f"{a.csv}: {len(df)} runs | dims={sorted(df.dim.unique())} | "
          f"variants={variants} | functions={df.fid.nunique()}\n")

    if len(variants) > 2:
        ranks = []
        for f, g in df.groupby("fid"):
            piv = g.pivot_table(index="seed", columns="variant", values="error")[variants]
            ranks.append(np.apply_along_axis(rankdata, 1, piv.values).mean(axis=0))
        mr = np.mean(ranks, axis=0)
        print("MEAN RANK (lower=better): " + "  ".join(f"{v}={x:.3f}" for v, x in zip(variants, mr)) + "\n")

    for ref in a.refs:
        rows = []
        for f, g in df.groupby("fid"):
            piv = g.pivot_table(index="seed", columns="variant", values="error")
            if a.left not in piv or ref not in piv:
                continue
            x, y = piv[a.left].values, piv[ref].values
            try:
                _, p = wilcoxon(x, y)
            except ValueError:
                p = 1.0
            rows.append(dict(F=f, group=GROUP[f], left=np.median(x), right=np.median(y),
                             delta=cliffs(x, y), p=p))
        t = pd.DataFrame(rows)
        t["p_holm"] = holm(t.p.values)
        t["verdict"] = np.where(t.p_holm >= .05, "ns",
                        np.where(t.left < t.right, f"{a.left} WINS", f"{ref} WINS"))
        t = t.rename(columns={"left": a.left, "right": ref})
        pd.set_option("display.width", 220)
        print(f"=== {a.left} vs {ref} : paired Wilcoxon per function ===")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        w = (t.verdict == f"{a.left} WINS").sum(); l = (t.verdict == f"{ref} WINS").sum()
        print(f"  --> {a.left} {w} - {ref} {l}, ties {len(t)-w-l}  (Holm, alpha=0.05)")
        print("\n  by function group:")
        print(t.groupby("group").verdict.value_counts().to_string(), "\n")


if __name__ == "__main__":
    main()
