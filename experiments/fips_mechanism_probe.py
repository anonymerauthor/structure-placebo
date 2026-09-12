"""Is the FIPS weighting strong enough for its control to mean anything?

A control only carries information if the mechanism it neutralises actually
moves the search. For the first audit target we reported the normalised
displacement rho alongside every comparison; the same quantity is needed here,
and it has to be measured rather than assumed, because FIPS distributes a
fixed budget phi among neighbours and a weighting that barely redistributes it
would be inert by construction.

For particle i with neighbours N(i), FIPS forms

    Delta_i^w = sum_n (phi_n^w / phi) * (p_n - x_i)

with phi_n^w proportional to the weight w_n. We compare the weighted
aggregate against the uniform one and report

    rho   = mean_i || Delta_i^w - Delta_i^unif || / sigma_pop
    chi_i = sqrt(d_i) * || w_i - u_i ||_2          (the leverage of the bound)

on the Barabasi-Albert topology the study uses.
"""
from __future__ import annotations

import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "audit"))

from fips import FIPS  # noqa: E402


def probe(n=60, m=2, dim=30, bounds=(-100.0, 100.0), n_draws=40, seed=0):
    rng = np.random.default_rng(seed)
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    nbrs = [sorted(g.neighbors(i)) for i in range(n)]
    deg = np.array([g.degree(i) for i in range(n)], float)

    rows = []
    for scheme in ("degree", "goodness"):
        shifts, chis = [], []
        for _ in range(n_draws):
            x = rng.uniform(*bounds, size=(n, dim))
            p = rng.uniform(*bounds, size=(n, dim))
            pf = rng.random(n)                     # stand-in personal-best quality
            sigma = x.std(axis=0).mean() * np.sqrt(dim)
            for i in range(n):
                nb = nbrs[i]
                if len(nb) < 2:
                    continue
                if scheme == "degree":
                    w = deg[nb]
                else:
                    order = np.argsort(np.argsort(pf[nb]))
                    w = (len(nb) - order).astype(float)
                w = w / w.sum()
                u = np.full(len(nb), 1.0 / len(nb))
                diff = p[nb] - x[i]
                shifts.append(np.linalg.norm((w - u) @ diff) / sigma)
                chis.append(np.sqrt(len(nb)) * np.linalg.norm(w - u))
        rows.append(dict(scheme=scheme, rho=float(np.mean(shifts)),
                         chi=float(np.mean(chis)),
                         chi_max=float(np.max(chis))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = probe()
    pd.set_option("display.width", 160)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out = os.path.join(HERE, "outputs", "fips_mechanism_probe.csv")
    t.to_csv(out, index=False)
    print(f"\n-> {out}")
    print("rho is the displacement the weighting causes, in units of the "
          "population spread.\nFor the first target the same quantity was "
          "0.25-0.31; a much smaller value here would\nmean the control is "
          "uninformative by construction rather than by measurement.")
