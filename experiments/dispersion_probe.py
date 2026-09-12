"""Does the permutation control preserve the weighting's dispersion pathwise?

Proposition 1 says the permuted operator equals uniform mixing *in expectation*.
That is a statement about the mean over permutations, and it leaves a fair
question open: the global score multiset is preserved by construction, but each
node's *local* normalized weight vector is not, because a node's neighbourhood
receives a random subset of the multiset rather than its original one. If
permuting systematically flattened the local weight vectors, the control would
be quietly weakening the mechanism instead of only re-assigning it, and "the
mechanism is still running at full strength" would be the wrong description.

This probe settles it without running the optimizer, since all three quantities
below depend only on the graph and the score vector:

  chi        sqrt(d_i) * ||w_i - u_i||_2, the leverage of Proposition 2
  n_eff      1 / sum_j w_ij^2, as a fraction of the degree d_i: how many
             neighbours the aggregate effectively averages over
  entropy     H(w_i) / log(d_i), the normalized dispersion of the weights

Reported as per-node distributions for the published weighting and for the
permutation control, over many permutation draws.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import networkx as nx
import pandas as pd

EPS = 1e-10  # the audited implementation's zero-guard; see audit/fast_ops.py

# The graph configurations the main text reports, matching mechanism_probe.py:
# the original report's defaults, the reference implementation's defaults, its
# multimodal category, and the vertex-transitive ring the leverage bound
# predicts to be inert.
CONFIGS = [
    ("paper defaults", 200, 4, 0.3),
    ("reference impl.", 60, 5, 0.1),
    ("multimodal category", 150, 5, 0.1),
    ("ring lattice", 200, 4, 0.0),
]


def node_stats(G, nodes, score):
    """Per-node leverage, effective neighbourhood size and weight entropy."""
    idx = {n: i for i, n in enumerate(nodes)}
    c = np.asarray(score, dtype=float) + EPS
    chi, neff, ent = [], [], []
    for n in nodes:
        nb = list(G.neighbors(n))
        d = len(nb)
        if d < 2:
            continue
        w = c[[idx[m] for m in nb]]
        w = w / w.sum()
        u = 1.0 / d
        chi.append(np.sqrt(d) * np.linalg.norm(w - u))
        neff.append((1.0 / np.sum(w ** 2)) / d)
        p = w[w > 0]
        ent.append(float(-np.sum(p * np.log(p)) / np.log(d)))
    return np.array(chi), np.array(neff), np.array(ent)


def run(n_perm=200, seed=0):
    rows = []
    for label, N, K, p in CONFIGS:
        G = nx.watts_strogatz_graph(N, K, p, seed=seed)
        nodes = list(G.nodes)
        bc = nx.betweenness_centrality(G)
        c = np.array([bc[n] for n in nodes])

        chi, neff, ent = node_stats(G, nodes, c)
        for a, b, e in zip(chi, neff, ent):
            rows.append(dict(config=label, N=N, K=K, p=p, arm="social",
                             chi=a, n_eff=b, entropy=e))

        rng = np.random.default_rng(seed + 1)
        for _ in range(n_perm):
            chi, neff, ent = node_stats(G, nodes, rng.permutation(c))
            for a, b, e in zip(chi, neff, ent):
                rows.append(dict(config=label, N=N, K=K, p=p, arm="shuffled",
                                 chi=a, n_eff=b, entropy=e))
        print(f"{label}: N={N} K={K} p={p} done", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = run()
    out = os.path.join(root, "data", "dispersion_probe.csv")
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")

    s = df.groupby(["config", "arm"])[["chi", "n_eff", "entropy"]].agg(["mean", "std"])
    print(s.round(4).to_string())
