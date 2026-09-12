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

Weightings provided:

    "uniform"    w_n = 1; this is FIPS as published
    "goodness"   w_n from the quality of neighbour n's previous best; this is
                 wFIPS as published
    "goodness_permuted"  the same goodness values permuted across the
                 neighbourhood, recomputed each iteration        [control]
    "degree"     w_n = deg(n); NOT a published variant, constructed here as a
                 synthetic mechanism to check the instrument detects a harmful
                 one as well as a null one
    "permuted"   the degree values permuted across nodes         [control]

The published FIPS family is FIPS (equal), wFIPS (goodness), wdFIPS
(distance), Self and wSelf. There is no degree-weighted member; "degree" here
is ours and is labelled as such wherever it is reported.

Goodness is taken as the within-neighbourhood rank of the neighbour's personal
best, best receiving the largest share. The original states that contributions
are weighted by goodness without fixing the transform in the sources available
to us; a rank transform is scale-free, which matters on a suite whose errors
span many orders of magnitude, and it matches the influence score used by the
other audit target so the two remain comparable.

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
    STATIC = ("degree", "permuted", "uniform")
    DYNAMIC = ("goodness", "goodness_permuted")

    def __init__(self, n_particles: int = 60, m_attach: int = 2,
                 weighting: str = "degree", chi: float = CHI, phi: float = PHI,
                 rng: Optional[np.random.Generator] = None):
        if weighting not in self.STATIC + self.DYNAMIC:
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

        if self.weighting == "degree":
            w = deg
        elif self.weighting == "permuted":
            # Same multiset of degrees, no correspondence to graph position.
            # Drawn from a stream seeded by the graph so the run stays
            # reproducible without consuming the optimizer's own randomness.
            w = np.random.default_rng(seed + 10_000_007).permutation(deg)
        else:
            w = np.ones(self.n)          # uniform, and the base for goodness
        return nbrs, w

    @staticmethod
    def _goodness(pf: np.ndarray, nb: List[int], permute: bool,
                  rng: np.random.Generator) -> np.ndarray:
        """Rank of each neighbour's personal best within the neighbourhood.

        Best gets the largest share. Permuting keeps the same shares but
        detaches them from which neighbour is actually good; it is redrawn
        every iteration because the underlying quantity changes every
        iteration.
        """
        f = pf[nb]
        order = np.argsort(np.argsort(f))          # 0 = best
        w = (len(nb) - order).astype(float)        # best -> len, worst -> 1
        return rng.permutation(w) if permute else w

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

        dynamic = self.weighting in self.DYNAMIC
        # A separate stream for the control's permutations, so that permuting
        # does not shift the optimizer's own draws and desynchronise the
        # paired comparison against the other arms.
        prng = np.random.default_rng((0 if seed is None else int(seed)) + 20_000_011)

        def contribution_weights(i: int) -> np.ndarray:
            nb = nbrs[i]
            if dynamic:
                wi = self._goodness(pf, nb, self.weighting.endswith("permuted"),
                                    prng)
            else:
                wi = w[nb] if nb else np.array([1.0])
            s = wi.sum()
            return (self.phi * wi / s if s > 0
                    else np.full(len(wi), self.phi / max(len(wi), 1)))

        # Static weightings are fixed for the whole run; goodness is not.
        phis = None if dynamic else [contribution_weights(i) for i in range(self.n)]

        while not obj.exhausted:
            for i in range(self.n):
                nb = nbrs[i]
                if not nb:
                    continue
                ph = contribution_weights(i) if dynamic else phis[i]
                # U(0, phi_n) drawn independently per neighbour and dimension
                u = rng.random((len(nb), dim)) * ph[:, None]
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
