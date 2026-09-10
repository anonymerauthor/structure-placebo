"""Build the manuscript figures from the experiment CSVs."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import figstyle as F

F.apply()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cliffs(a, b):
    n = len(a) * len(b)
    return (sum(x > y for x in a for y in b) - sum(x < y for x in a for y in b)) / n


def boot_ci(a, b, n_boot=2000, seed=0):
    """Percentile bootstrap CI for Cliff's delta over paired runs."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(a))
    vals = [cliffs(a[s], b[s]) for s in (rng.choice(idx, len(idx)) for _ in range(n_boot))]
    return np.percentile(vals, [2.5, 97.5])


def effects(df, left, right, fcol, vcol):
    rows = []
    for f, g in df.groupby(fcol):
        piv = g.pivot_table(index="seed", columns="variant", values=vcol)
        if left not in piv or right not in piv:
            continue
        a, b = piv[left].values, piv[right].values
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        lo, hi = boot_ci(a, b)
        try:
            _, p = wilcoxon(a, b)
        except ValueError:
            p = 1.0
        label = f"F{f}" if isinstance(f, (int, np.integer)) else str(f).replace("_", " ")
        rows.append(dict(f=label, delta=cliffs(a, b), lo=lo, hi=hi, p=p))
    return pd.DataFrame(rows)


def fig_placebo_forest():
    """Effect of destroying the structure-weight association, three suites."""
    panels = []
    p20 = os.path.join(ROOT, "outputs/placebo_study.csv")
    p105 = os.path.join(ROOT, "outputs/placebo_105k.csv")
    pcec = os.path.join(ROOT, "outputs/cec2017_placebo.csv")
    if os.path.exists(p20):
        panels.append(("Classic $F_{1..23}$, 20 000 evals",
                       effects(pd.read_csv(p20), "social", "shuffled", "function", "best")))
    if os.path.exists(p105):
        panels.append(("Classic $F_{1..23}$, 105 000 evals",
                       effects(pd.read_csv(p105), "social", "shuffled", "function", "best")))
    if os.path.exists(pcec):
        panels.append(("CEC2017 $D{=}30$, 300 000 evals",
                       effects(pd.read_csv(pcec), "social", "shuffled", "fid", "error")))
    p50 = os.path.join(ROOT, "outputs/cec2017_D50.csv")
    if os.path.exists(p50):
        panels.append(("CEC2017 $D{=}50$, 500 000 evals",
                       effects(pd.read_csv(p50), "social", "shuffled", "fid", "error")))

    n = len(panels)
    heights = [max(1.4, 0.14 * len(t) + 0.55) for _, t in panels]
    fig, axes = plt.subplots(1, n, figsize=(F.COL_DOUBLE, max(heights)),
                             gridspec_kw={"width_ratios": [1] * n})
    axes = np.atleast_1d(axes)

    for ax, (title, t) in zip(axes, panels):
        t = t.reset_index(drop=True)
        y = np.arange(len(t))
        ax.axvspan(-0.147, 0.147, color=F.c("grey"), alpha=0.15, lw=0,
                   label="negligible" if ax is axes[0] else None)
        ax.axvline(0, color=F.c("black"), lw=0.6, zorder=1)
        ax.hlines(y, t.lo, t.hi, color=F.c("blue"), lw=0.9, zorder=2)
        ax.plot(t.delta, y, "o", color=F.c("blue"), mfc="white", mew=0.8, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(t.f, fontsize=5.5)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xlabel(r"Cliff's $\delta$   (published $-$ permuted)")
        ax.set_title(title, fontsize=7)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
    axes[0].legend(loc="lower left", fontsize=6)
    fig.tight_layout(w_pad=0.6)
    return F.save(fig, "fig_placebo_forest", os.path.join(ROOT, "paper/figures"))


def fig_mechanism_magnitude():
    """The mechanism is not inert: it displaces the update substantially."""
    path = os.path.join(ROOT, "outputs/mechanism_probe.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d.p > 0]                       # the ring lattice is a degenerate control
    lbl = [f"$N$={int(r.N)}, $K$={int(r.K)}\n$p$={r.p:g}" for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(F.COL_SINGLE, 1.9))
    x = np.arange(len(d))
    ax.bar(x, d.rel_shift, color=F.c("blue"), width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(lbl, fontsize=5.5)
    ax.set_ylabel("$\\|\\bar{x}_{\\mathrm{cent}}-\\bar{x}_{\\mathrm{unif}}\\|\\,/\\,\\sigma_{\\mathrm{pop}}$")
    ax.set_ylim(0, max(0.35, d.rel_shift.max() * 1.15))
    ax.grid(axis="y", color=F.c("grey"), alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return F.save(fig, "fig_mechanism_magnitude", os.path.join(ROOT, "paper/figures"))


def fig_density_law():
    """Optimal neighbourhood density against landscape ruggedness."""
    path = os.path.join(ROOT, "outputs/density_law_full.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[d.p_fried < 0.05]                 # only where density matters at all
    fig, ax = plt.subplots(figsize=(F.COL_SINGLE, 2.3))
    dims = sorted(d.dim.unique())
    for i, dim in enumerate(dims):
        s = d[d.dim == dim]
        ax.scatter(s.autocorr, s.dense_pref, s=16,
                   marker=F.MARKERS[i % len(F.MARKERS)],
                   facecolor="none", edgecolor=F.c(F.ORDER[i % len(F.ORDER)]),
                   linewidth=0.9, label=f"$D={dim}$")
    x = d.autocorr.to_numpy(); y = d.dense_pref.to_numpy()
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, a + b * xs, color=F.c("black"), lw=0.8, ls="--", zorder=0)
    from scipy.stats import spearmanr
    rho, p = spearmanr(x, y)
    ax.axhline(0, color=F.c("grey"), lw=0.5, zorder=0)
    ax.set_xlabel("landscape autocorrelation  (lower = more rugged)")
    ax.set_ylabel("preference for dense\n(rank$_{K=4}$ $-$ rank$_{K=59}$)")
    ax.set_title("Spearman $\\rho=%.2f$, $p=%.1e$, $n=%d$" % (rho, p, len(d)),
                 fontsize=7)
    ax.legend(loc="upper right", ncol=1)
    fig.tight_layout()
    return F.save(fig, "fig_density_law", os.path.join(ROOT, "paper/figures"))


def fig_ranks():
    """Mean Friedman ranks of the six arms on CEC2017."""
    path = os.path.join(ROOT, "outputs/cec2017_D30_all.csv")
    if not os.path.exists(path):
        return None
    from scipy.stats import rankdata
    df = pd.read_csv(path)
    arms = ["social", "shuffled", "uniform_K4", "fixed_K16", "ads", "random_K"]
    arms = [a for a in arms if a in set(df.variant)]
    per = []
    for _, g in df.groupby("fid"):
        piv = g.pivot_table(index="seed", columns="variant", values="error")
        if not set(arms).issubset(piv.columns):
            continue
        per.append(np.apply_along_axis(rankdata, 1, piv[arms].values).mean(axis=0))
    per = np.array(per)
    mr, se = per.mean(axis=0), per.std(axis=0, ddof=1) / np.sqrt(len(per))
    names = ["Social", "Perm", "Unif$_4$", "Fix$_{16}$", "Ads", "Rand"]
    # The two arms whose difference is the paper's primary claim
    colors = [F.c("blue"), F.c("blue"), F.c("grey"), F.c("grey"),
              F.c("orange"), F.c("orange")]
    fig, ax = plt.subplots(figsize=(F.COL_SINGLE, 2.0))
    y = np.arange(len(arms))
    ax.barh(y, mr, xerr=se, color=colors, height=0.6,
            error_kw=dict(lw=0.8, capsize=2, ecolor=F.c("black")))
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("mean Friedman rank over 28 functions (lower is better)")
    ax.set_xlim(0, max(mr) * 1.25)
    ax.grid(axis="x", color=F.c("grey"), alpha=0.3)
    ax.set_axisbelow(True)
    for i, (v, e) in enumerate(zip(mr, se)):
        ax.text(v + e + 0.06, i, f"{v:.2f}", va="center", fontsize=6)
    fig.tight_layout()
    return F.save(fig, "fig_ranks", os.path.join(ROOT, "paper/figures"))


def fig_equivalence():
    """TOST intervals per function against the negligibility margin.

    A difference test can only fail to reject; equivalence has to be shown.
    An interval lying wholly inside the margin establishes it, one straddling
    the margin leaves the question open, and the distinction is the point of
    the figure.
    """
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    from experiments.equivalence import analyse, MARGIN_NEGLIGIBLE as M

    panels = []
    for label, path, fcol, vcol in [
        ("$D=10$", "outputs/cec2017_D10_all.csv", "fid", "error"),
        ("$D=30$", "outputs/cec2017_D30_51.csv", "fid", "error"),
        ("$D=50$", "outputs/cec2017_D50.csv", "fid", "error"),
    ]:
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            t = analyse(pd.read_csv(p), "social", "shuffled", fcol, vcol, M, 4000)
            panels.append((label, t.reset_index(drop=True)))
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), sharey=True,
                             figsize=(F.COL_DOUBLE, 0.15 * len(panels[0][1]) + 0.9))
    axes = np.atleast_1d(axes)
    style = {"equivalent": (F.c("green"), "o"),
             "inconclusive": (F.c("grey"), "s"),
             "different": (F.c("red"), "D")}

    for ax, (label, t) in zip(axes, panels):
        y = np.arange(len(t))
        ax.axvspan(-M, M, color=F.c("grey"), alpha=0.18, lw=0)
        for m in (-M, M):
            ax.axvline(m, color=F.c("black"), lw=0.5, ls=":")
        ax.axvline(0, color=F.c("black"), lw=0.6)
        for i, r in enumerate(t.itertuples()):
            col, mk = style[r.verdict]
            ax.hlines(i, r.lo90, r.hi90, color=col, lw=1.0)
            ax.plot(r.delta, i, mk, color=col, mfc="white", mew=0.8, ms=2.8)
        n_eq = int((t.verdict == "equivalent").sum())
        ax.set_title(f"{label}: {n_eq}/{len(t)} equivalent", fontsize=7)
        ax.set_xlabel("Cliff's $\\delta$ with 90% CI")
        ax.set_xlim(-0.75, 0.75)
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(np.arange(len(panels[0][1])))
    axes[0].set_yticklabels(panels[0][1].f, fontsize=5.5)

    handles = [plt.Line2D([], [], color=c, marker=m, ls="-", mfc="white",
                          mew=0.8, ms=3, label=k) for k, (c, m) in style.items()]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=6)
    fig.tight_layout(w_pad=0.5)
    return F.save(fig, "fig_equivalence", os.path.join(ROOT, "paper/figures"))


def fig_calibration():
    """The instrument on three mechanisms of known and unknown informativeness.

    One panel per contrast, all on the same suite, sample size and statistic,
    so the reader can see that the same test that returns nothing on the
    centrality assignment returns a large effect where the permuted score is
    informative by construction. Without the calibration a null result is
    indistinguishable from a test with no power.
    """
    contrasts = [
        ("Centrality assignment", "social", "shuffled",
         "outputs/cec2017_D30_51_all.csv"),
        ("Adaptive selection", "ads", "random_K",
         "outputs/cec2017_D30_51_all.csv"),
        ("Fitness influence", "social", "shuffled_inf",
         "outputs/cec2017_poscontrol.csv"),
    ]
    rows = []
    for label, left, right, path in contrasts:
        p = os.path.join(ROOT, path)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        if not {left, right}.issubset(set(df.variant)):
            continue
        deltas = []
        for _, g in df.groupby("fid"):
            piv = g.pivot_table(index="seed", columns="variant", values="error")
            if left not in piv or right not in piv:
                continue
            a, b = piv[left].to_numpy(), piv[right].to_numpy()
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            deltas.append(cliffs(a, b))
        if not deltas:
            continue
        d = np.array(deltas)
        rng = np.random.default_rng(0)
        boot = np.array([d[rng.integers(0, len(d), len(d))].mean()
                         for _ in range(20000)])
        lo, hi = np.percentile(boot, [5, 95])
        rows.append(dict(label=label, mean=d.mean(), lo=lo, hi=hi,
                         d=d, n=len(d)))
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(F.COL_SINGLE, 0.62 * len(rows) + 1.0))
    y = np.arange(len(rows))
    ax.axvspan(-0.147, 0.147, color=F.c("grey"), alpha=0.18, lw=0,
               label="negligible")
    ax.axvline(0, color=F.c("black"), lw=0.6)
    for i, r in enumerate(rows):
        # individual functions behind the suite estimate
        ax.plot(r["d"], np.full(len(r["d"]), i) + 0.22, ".",
                color=F.c("grey"), ms=2.2, alpha=0.75)
        col = F.c("green") if r["lo"] > -0.147 and r["hi"] < 0.147 else F.c("red")
        ax.hlines(i, r["lo"], r["hi"], color=col, lw=1.8)
        ax.plot(r["mean"], i, "o", color=col, mfc="white", mew=1.0, ms=4)
        ax.text(r["mean"], i - 0.26, f"{r['mean']:+.3f}", ha="center",
                fontsize=6, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Cliff's $\\delta$: published score $-$ permuted score")
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    return F.save(fig, "fig_calibration", os.path.join(ROOT, "paper/figures"))


if __name__ == "__main__":
    for fn in (fig_placebo_forest, fig_mechanism_magnitude,
               fig_density_law, fig_ranks, fig_equivalence, fig_calibration):
        out = fn()
        print(("wrote " + out) if out else f"skipped {fn.__name__}: input missing")
