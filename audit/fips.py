"""Fully Informed Particle Swarm, with the same controls applied to it.

FIPS (Mendes, Kennedy and Neves, IEEE TEVC 2004) replaces the personal/global
best pair by a contribution from *every* neighbour, and the paper's weighted
variants scale each contribution by a per-neighbour score. The degree-weighted
variant is a structural claim of exactly the kind this repository audits: a
node's position in the graph is asserted to say how much its experience is
worth to its neighbours.

    v_i <- chi ( v_i + sum_{n in N(i)} U(0, phi_n) * (p_n - x_i) )
    x_i <- x_i + v_i
    phi_n = PHI * w_n / sum_{m in N(i)} w_m

with chi = 0.7298 and PHI = 4.1 as in the original.

Three weightings are provided:

    "degree"    w_n = deg(n), the published structural variant
    "permuted"  the same degree values, permuted across nodes  [control]
    "uniform"   w_n = 1, the ablation

The topology is Barabasi-Albert rather than a regular lattice, deliberately:
on a regular graph every degree is equal, the leverage chi_i of the bound in
the accompanying paper is identically zero, and the control would be vacuous
by construction rather than by measurement.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import networkx as nx
import numpy as np

CHI = 0.7298
PHI = 4.1


class FIPS:
    def __init__(self, n_particles: int = 60, m_attach: int = 2,
                 weighting: str = "degree", chi: float = CHI, phi: float = PHI,
                 rng: Optional[np.random.Generator] = None):
        if weighting not in ("degree", "permuted", "uniform"):
            raise ValueError(f"unknown weighting: {weighting}")
        self.n = n_particles
        self.m_attach = m_attach
        self.weighting = weighting
        self.chi = chi
        self.phi = phi
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------ graph
    def _topology(self, seed: int) -> tuple[List[List[int]], np.ndarray]:
        """Neighbour lists and the per-node weight vector."""
        g = nx.barabasi_albert_graph(self.n, self.m_attach, seed=seed)
        nbrs = [sorted(g.neighbors(i)) for i in range(self.n)]
        deg = np.array([g.degree(i) for i in range(self.n)], dtype=float)

        if self.weighting == "uniform":
            w = np.ones(self.n)
        elif self.weighting == "degree":
            w = deg
        else:
            # Same multiset of degrees, no correspondence to graph position.
            # Drawn from a stream seeded by the graph so the run stays
            # reproducible without consuming the optimizer's own randomness.
            w = np.random.default_rng(seed + 10_000_007).permutation(deg)
        return nbrs, w

    # ------------------------------------------------------------------- run
    def optimize(self, obj, bounds: Sequence[float], dim: int,
                 seed: Optional[int] = None) -> dict:
        rng = self.rng
        lo, hi = float(bounds[0]), float(bounds[1])
        span = hi - lo
        nbrs, w = self._topology(0 if seed is None else int(seed))

        x = rng.uniform(lo, hi, size=(self.n, dim))
        v = rng.uniform(-0.1 * span, 0.1 * span, size=(self.n, dim))
        p = x.copy()
        pf = np.array([obj(xi) for xi in x])

        gi = int(np.argmin(pf))
        history = [float(pf[gi])]

        # Per-particle contribution weights, normalised to sum to phi.
        phis = []
        for i in range(self.n):
            wi = w[nbrs[i]] if nbrs[i] else np.array([1.0])
            s = wi.sum()
            phis.append(self.phi * wi / s if s > 0 else
                        np.full(len(wi), self.phi / max(len(wi), 1)))

        while not obj.exhausted:
            for i in range(self.n):
                nb = nbrs[i]
                if not nb:
                    continue
                # U(0, phi_n) drawn independently per neighbour and dimension
                u = rng.random((len(nb), dim)) * phis[i][:, None]
                v[i] = self.chi * (v[i] + (u * (p[nb] - x[i])).sum(axis=0))
                x[i] = np.clip(x[i] + v[i], lo, hi)

            for i in range(self.n):
                if obj.exhausted:
                    break
                f = obj(x[i])
                if f < pf[i]:
                    pf[i] = f
                    p[i] = x[i].copy()
            history.append(float(pf.min()))

        gi = int(np.argmin(pf))
        return {"best_fitness": float(pf[gi]), "best_position": p[gi],
                "convergence_history": history, "evals_used": obj.used}
