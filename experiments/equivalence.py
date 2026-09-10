"""Equivalence testing for the control comparisons.

A non-significant difference test is not evidence of equivalence: it is equally
consistent with a study too small to detect anything. We therefore test the
complementary hypothesis directly with two one-sided tests (TOST).

The outcome scale varies across the suite by more than eighteen orders of
magnitude (errors from 1e-12 to 1e6), so an equivalence margin on the raw
difference is meaningless. We place the margin on Cliff's delta instead, a
scale-free dominance measure, and use the conventional negligible-effect
threshold as the default bound.

For a paired comparison with effect size delta and margin m, TOST tests

    H0:  |delta| >= m        against       H1:  |delta| < m

Both one-sided p-values are obtained from the bootstrap distribution of the
effect size over resampled seed pairs; the TOST p-value is their maximum.
Equivalently, equivalence holds at level alpha when the 100(1-2*alpha)%
confidence interval lies entirely inside (-m, m), and we report that interval
so a reader can apply any margin they prefer.
"""
import argparse
import numpy as np
import pandas as pd

# Romano et al.'s thresholds for Cliff's delta: |d| < 0.147 negligible,
# < 0.33 small, < 0.474 medium.
MARGIN_NEGLIGIBLE = 0.147
MARGIN_SMALL = 0.330


def cliffs_delta(a, b):
    """Dominance of a over b; positive means a takes larger (worse) values."""
    return float(np.sign(np.subtract.outer(a, b)).mean())


def bootstrap_delta(a, b, n_boot=4000, seed=0):
    """Bootstrap the effect size, resampling seed pairs to respect pairing."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.array([cliffs_delta(a[i], b[i]) for i in idx])


def tost(a, b, margin=MARGIN_NEGLIGIBLE, n_boot=4000, seed=0):
    """Two one-sided tests for |delta| < margin.

    Returns the observed effect, the 90% interval (the one whose containment
    in the margin is equivalent to TOST at alpha = 0.05), and the TOST p-value.
    """
    d = cliffs_delta(a, b)
    boot = bootstrap_delta(a, b, n_boot=n_boot, seed=seed)
    # Evidence against delta >= +margin, and against delta <= -margin.
    p_upper = float(np.mean(boot >= margin))
    p_lower = float(np.mean(boot <= -margin))
    lo90, hi90 = np.percentile(boot, [5, 95])
    lo95, hi95 = np.percentile(boot, [2.5, 97.5])
    return dict(delta=d, lo90=lo90, hi90=hi90, lo95=lo95, hi95=hi95,
                p_tost=max(p_upper, p_lower))


def verdict(row, margin):
    """Equivalent, different, or inconclusive at alpha = 0.05."""
    inside = row.lo90 > -margin and row.hi90 < margin
    excludes_zero = row.lo95 > 0 or row.hi95 < 0
    if inside:
        return "equivalent"
    if excludes_zero:
        return "different"
    return "inconclusive"


def analyse(df, left, right, fcol, vcol, margin=MARGIN_NEGLIGIBLE, n_boot=4000):
    rows = []
    for f, g in df.groupby(fcol):
        piv = g.pivot_table(index="seed", columns="variant", values=vcol)
        if left not in piv or right not in piv:
            continue
        a, b = piv[left].to_numpy(), piv[right].to_numpy()
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        r = tost(a, b, margin=margin, n_boot=n_boot, seed=int(abs(hash(str(f))) % 2**31))
        r["f"] = f"F{f}" if isinstance(f, (int, np.integer)) else str(f)
        rows.append(r)
    t = pd.DataFrame(rows)
    t["verdict"] = [verdict(r, margin) for r in t.itertuples()]
    # Precision: the narrowest effect the design could have ruled out.
    t["ci_halfwidth"] = (t.hi90 - t.lo90) / 2
    return t[["f", "delta", "lo90", "hi90", "p_tost", "ci_halfwidth", "verdict"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--fcol", default="fid")
    ap.add_argument("--vcol", default="error")
    ap.add_argument("--margin", type=float, default=MARGIN_NEGLIGIBLE)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    t = analyse(df, a.left, a.right, a.fcol, a.vcol, a.margin, a.boot)
    pd.set_option("display.width", 200)
    print(f"{a.csv}: {a.left} vs {a.right} | margin |delta| < {a.margin} | "
          f"{a.boot} bootstrap resamples\n")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    counts = t.verdict.value_counts().to_dict()
    n = len(t)
    print(f"\n  equivalent   {counts.get('equivalent', 0):3d}/{n}")
    print(f"  inconclusive {counts.get('inconclusive', 0):3d}/{n}")
    print(f"  different    {counts.get('different', 0):3d}/{n}")
    print(f"\n  median 90% CI half-width: {t.ci_halfwidth.median():.3f} "
          f"(effects larger than this were detectable)")
    print(f"  largest |delta| observed: {t.delta.abs().max():.3f}")

    # Sensitivity to the margin, so the conclusion is not an artefact of 0.147.
    print("\n  margin sensitivity:")
    for m in (0.10, 0.147, 0.20, 0.33):
        eq = sum(1 for r in t.itertuples() if r.lo90 > -m and r.hi90 < m)
        print(f"    |delta| < {m:<5} : {eq:3d}/{n} equivalent")

    if a.out:
        t.to_csv(a.out, index=False)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
