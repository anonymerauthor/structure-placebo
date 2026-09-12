"""Emit the LaTeX result tables from the run files."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _data(name: str) -> str:
    """Resolve a results file.

    Published copies live in data/; a working tree may still have them under
    outputs/. Checking both keeps the figures reproducible from a fresh clone.
    """
    base = os.path.basename(name)
    for d in ("data", "outputs"):
        p = os.path.join(ROOT, d, base)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, "data", base)
OUT = os.path.join(ROOT, "paper", "tables")

ARMS = ["social", "shuffled", "uniform_K4", "fixed_K16", "random_K", "ads"]
LABEL = {"social": r"\textsc{Social}", "shuffled": r"\permcontrol{}",
         "uniform_K4": r"\textsc{Unif}$_4$", "fixed_K16": r"\textsc{Fix}$_{16}$",
         "random_K": r"\polcontrol{}", "ads": r"\textsc{Ads}"}
GROUP = {**{i: "Unimodal" for i in (1, 2, 3)},
         **{i: "Simple multimodal" for i in range(4, 11)},
         **{i: "Hybrid" for i in range(11, 21)},
         **{i: "Composition" for i in range(21, 31)}}


def sci(v):
    """Compact scientific notation for a table cell."""
    if not np.isfinite(v):
        return "--"
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10.0 ** e
    return f"{m:.2f}e{e:+03d}"


def results_table(csv, dim, out):
    df = pd.read_csv(csv)
    df = df[df.dim == dim]
    arms = [a for a in ARMS if a in set(df.variant)]
    rows, ranks = [], []
    for fid, g in df.groupby("fid"):
        piv = g.pivot_table(index="seed", columns="variant", values="error")
        if not set(arms).issubset(piv.columns) or piv[arms].isna().any().any():
            continue
        med = {a: float(np.median(piv[a])) for a in arms}
        best = min(med, key=med.get)
        # Which arms are statistically tied with the best, per paired Wilcoxon
        tied = set()
        for a in arms:
            if a == best:
                continue
            try:
                if wilcoxon(piv[best].to_numpy(), piv[a].to_numpy()).pvalue >= 0.05:
                    tied.add(a)
            except ValueError:
                tied.add(a)
        cells = []
        for a in arms:
            s = sci(med[a])
            if a == best or a in tied:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        rows.append((fid, GROUP[fid], cells))
        ranks.append(np.apply_along_axis(rankdata, 1, piv[arms].values).mean(axis=0))

    mr = np.mean(ranks, axis=0)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Median error on CEC2017 at $D=" + str(dim) + r"$ over 30 runs, "
        r"budget $10^{4}D$. Bold marks the best arm and any arm not "
        r"distinguishable from it by a paired Wilcoxon test at $\alpha=0.05$. "
        r"The last row gives mean Friedman ranks over the suite.}",
        r"\label{tab:cec" + str(dim) + "}",
        r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}ll" + "r" * len(arms) + r"@{}}", r"\toprule",
        "F & Class & " + " & ".join(LABEL[a] for a in arms) + r" \\", r"\midrule",
    ]
    prev = None
    for fid, grp, cells in rows:
        shown = grp if grp != prev else ""
        prev = grp
        lines.append(f"$F_{{{fid}}}$ & {shown} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"\multicolumn{2}{@{}l}{Mean rank} & "
        + " & ".join(f"{v:.3f}" for v in mr) + r" \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table*}",
    ]
    os.makedirs(OUT, exist_ok=True)
    open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"wrote {out}  ({len(rows)} functions, {len(arms)} arms)")
    print("  mean ranks: " + "  ".join(f"{a}={v:.3f}" for a, v in zip(arms, mr)))


if __name__ == "__main__":
    csv = os.path.join(ROOT, "outputs", "cec2017_D30_all.csv")
    if os.path.exists(csv):
        results_table(csv, 30, os.path.join(OUT, "cec2017_D30.tex"))
    else:
        print("missing", csv)
