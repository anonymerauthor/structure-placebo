"""Does the density law survive at n=49, and does it depend on dimension?"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata, friedmanchisquare
from social.functions import BenchmarkFunctions
from experiments.density_law import ruggedness

K_LEVELS = [4, 8, 16, 32, 59]
df = pd.read_csv("outputs/ksweep.csv")
bf = BenchmarkFunctions()

rows = []
for (f, d), g in df.groupby(["function", "dim"]):
    piv = g.pivot_table(index="seed", columns="K", values="best", aggfunc="mean")
    piv = piv.reindex(columns=K_LEVELS)
    if piv.isna().any().any():
        continue
    r = np.apply_along_axis(rankdata, 1, piv.values).mean(axis=0)
    try:
        _, p_fried = friedmanchisquare(*[piv[k].values for k in K_LEVELS])
    except ValueError:
        p_fried = np.nan
    func, bounds, _ = bf.functions[f]
    rows.append(dict(function=f, dim=d, best_K=K_LEVELS[int(np.argmin(r))],
                     dense_pref=r[0] - r[-1], p_fried=p_fried,
                     autocorr=ruggedness(func, bounds, d),
                     **{f"r{k}": v for k, v in zip(K_LEVELS, r)}))

t = pd.DataFrame(rows)
pd.set_option("display.width", 240)
print(t.sort_values(["dim", "dense_pref"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

print("\n" + "=" * 78)
sig = t[t.p_fried < 0.05]
print(f"configs where K matters at all (Friedman p<0.05): {len(sig)}/{len(t)}")

for label, sub in [("ALL", t), ("K matters only", sig)]:
    rho, p = spearmanr(sub.autocorr, sub.dense_pref)
    print(f"  {label:16s} n={len(sub):3d}  Spearman(autocorr, dense_pref) rho={rho:+.3f}  p={p:.2e}")

print("\n--- within each dimension ---")
for d, sub in t.groupby("dim"):
    rho, p = spearmanr(sub.autocorr, sub.dense_pref)
    print(f"  D={d:<3d} n={len(sub):3d}  rho={rho:+.3f}  p={p:.3f}   median best_K={sub.best_K.median():.0f}")

print("\n--- does optimal density grow with dimension? (scalable functions only) ---")
sc = t[t.function.isin(t.groupby('function').dim.nunique()[lambda s: s > 1].index)]
w = sc.pivot_table(index="function", columns="dim", values="best_K")
print(w.to_string())
print("  median best_K by D:", {int(c): float(w[c].median()) for c in w.columns})
t.to_csv("outputs/density_law_full.csv", index=False)
