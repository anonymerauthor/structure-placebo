"""How much does centrality weighting actually change the SOCIAL update?

The paper's central claim is that betweenness centrality makes agents learn
preferentially from structurally important neighbours. That claim is testable
without running the optimizer at all: compare the centrality-weighted neighbour
aggregate x_cent against the plain neighbour mean x_unif on the very same graph
and population. If the two coincide, the mechanism is inert by construction.

Reported per configuration:
  cv_bc        coefficient of variation of betweenness over nodes
  w_ratio      mean (max/min) neighbour weight within a node's neighbourhood
  rel_shift    ||x_cent - x_unif|| / population std, averaged over nodes
               -- the displacement the mechanism actually causes, in units of
               the population's own spread
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import networkx as nx
import pandas as pd

from social.fast_ops import build_mixing_matrices


def probe(N, K, p, dim=30, bounds=(-5.12, 5.12), seed=0, n_pop=20):
    rng = np.random.default_rng(seed)
    G = nx.watts_strogatz_graph(N, K, p, seed=seed)
    nodes = list(G.nodes)
    bc = nx.betweenness_centrality(G)
    bc_vals = np.array([bc[n] for n in nodes])

    ones = np.ones(N)
    shifts, wratios = [], []
    for _ in range(n_pop):
        X = rng.uniform(bounds[0], bounds[1], size=(N, dim))
        infl = rng.permutation(np.linspace(0, 1, N))  # placeholder, not used here
        Wc, _, _, _ = build_mixing_matrices(G, nodes, bc, infl, "centrality_weighted")
        Wu, _, _, _ = build_mixing_matrices(G, nodes, bc, infl, "uniform")
        x_cent, x_unif = Wc @ X, Wu @ X
        pop_std = X.std(axis=0).mean()
        shifts.append(np.linalg.norm(x_cent - x_unif, axis=1).mean() / (pop_std * np.sqrt(dim)))

    # weight spread inside each neighbourhood
    for n in nodes:
        nb = list(G.neighbors(n))
        if len(nb) < 2:
            continue
        w = np.array([bc[m] for m in nb]) + 1e-10
        wratios.append(w.max() / w.min())

    return {
        "N": N, "K": K, "p": p,
        "cv_bc": bc_vals.std() / bc_vals.mean() if bc_vals.mean() > 0 else np.nan,
        "bc_min": bc_vals.min(), "bc_max": bc_vals.max(),
        "w_ratio": float(np.mean(wratios)),
        "rel_shift": float(np.mean(shifts)),
    }


if __name__ == "__main__":
    configs = [
        # (N, K, p)  -- paper Table 2, repo config, and the sensitivity grid
        (200, 4, 0.3),    # paper defaults
        (60, 5, 0.1),     # repo Config defaults
        (150, 5, 0.1),    # repo multimodal category
        (200, 4, 0.0),    # ring lattice: no shortcuts
        (200, 4, 1.0),    # fully randomized
        (200, 10, 0.3),   # denser
        (200, 20, 0.3),   # much denser
    ]
    rows = [probe(*c) for c in configs]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/mechanism_probe.csv", index=False)
    print("\nrel_shift = displacement caused by centrality weighting, "
          "in units of population spread.")
