"""Adaptive-density search: the neighbourhood size is a decision variable.

Motivation from our own measurements:

  * Centrality weighting is inert. Permuting betweenness values across nodes
    changes nothing on 10/10 functions at both 20k and 105k budgets, so the
    structural weighting is dropped here (uniform mixing), which also removes
    the betweenness computation -- roughly 30% of per-iteration runtime.
  * Neighbourhood density is not inert. It changes the outcome significantly
    in 30 of 49 problem configurations, with rank spreads up to 3.9 out of 5.
  * The best density is not reliably predictable in advance. Correlation with
    a landscape ruggedness proxy is only rho=-0.61, and vanishes at D=10.

Since density matters but cannot be chosen a priori, it is selected online.
Each epoch the optimizer picks a degree K from a fixed arm set with a
sliding-window UCB rule and rebuilds the interaction graph at that degree,
keeping positions and fitness. Credit is the log-scale improvement of the
incumbent over the epoch, which is scale-free across functions.
"""

import numpy as np
import networkx as nx
from collections import deque
from typing import List, Optional

from .config import Config
from .optimizer import SOCIALOptimizer
from .graph_ops import create_watts_strogatz_graph


class AdaptiveDensityOptimizer(SOCIALOptimizer):
    """SOCIAL with online neighbourhood-density selection."""

    def __init__(self, config: Config, rng: Optional[np.random.Generator] = None,
                 arms: Optional[List[int]] = None, epoch: int = 12,
                 window: int = 40, c_ucb: float = 0.35, policy: str = "ucb"):
        super().__init__(config, rng)
        # "random" is the placebo control for this method: it cycles densities
        # on the same schedule but ignores the reward signal. If UCB does not
        # beat it, the benefit is density cycling, not adaptation.
        self.policy = policy
        n = config.NUM_NODES
        self.arms = [k for k in (arms or [4, 8, 16, 32, n - 1]) if 2 <= k <= n - 1]
        self.epoch = epoch
        self.c_ucb = c_ucb
        # Sliding window of (arm, reward) keeps the choice responsive to the
        # exploration->exploitation shift the schedule imposes on the run.
        self.history = deque(maxlen=window)
        self.current_arm = None
        self._epoch_start_best = None
        self._epoch_start_iter = -1
        self.trace = []            # (iteration, K, reward) for post-hoc analysis

    # ------------------------------------------------------------------ bandit
    def _select_arm(self) -> int:
        if self.policy == "random":
            return int(self.arms[self.rng.integers(len(self.arms))])
        untried = [k for k in self.arms if not any(a == k for a, _ in self.history)]
        if untried:
            return untried[self.rng.integers(len(untried))]

        # Raw credits are function- and phase-dependent (mean ~0.03 on Ackley,
        # ~0.12 on Sphere), so an absolute exploration bonus would swamp the
        # differences between arms. Rescaling the window to [0, 1] puts reward
        # and bonus on one scale and makes c_ucb problem-independent.
        rewards = np.array([r for _, r in self.history])
        lo, hi = rewards.min(), rewards.max()
        norm = (rewards - lo) / (hi - lo) if hi > lo else np.zeros_like(rewards)

        counts = {k: 0 for k in self.arms}
        totals = {k: 0.0 for k in self.arms}
        for (k, _), v in zip(self.history, norm):
            counts[k] += 1
            totals[k] += float(v)

        n_total = len(self.history)
        scores = {}
        for k in self.arms:
            if counts[k] == 0:                       # aged out of the window
                scores[k] = np.inf
            else:
                scores[k] = (totals[k] / counts[k]
                             + self.c_ucb * np.sqrt(np.log(n_total) / counts[k]))
        return max(scores, key=scores.get)

    @staticmethod
    def _credit(before: float, after: float) -> float:
        """Scale-free improvement credit in [0, 1]."""
        if not np.isfinite(before) or not np.isfinite(after) or after >= before:
            return 0.0
        span = abs(before) + 1e-12
        return float(np.clip(np.log1p((before - after) / span) / np.log(2.0), 0.0, 1.0))

    # --------------------------------------------------------------- rewiring
    def _rebuild(self, G: nx.Graph, K: int) -> nx.Graph:
        """New Watts-Strogatz topology at degree K, carrying state over."""
        nodes = list(G.nodes)
        n = len(nodes)
        if K >= n - 1:
            H = nx.complete_graph(n)
        else:
            H = create_watts_strogatz_graph(n, K, self.config.P_BASE,
                                            seed=int(self.rng.integers(2**31)))
        H = nx.relabel_nodes(H, dict(zip(sorted(H.nodes), nodes)), copy=True)
        for node in nodes:
            H.nodes[node]['position'] = G.nodes[node]['position']
            H.nodes[node]['fitness'] = G.nodes[node]['fitness']
        H.graph['_rewire_count'] = G.graph.get('_rewire_count', 0) + 1
        self._centrality_cache = {}
        self._last_centrality_iter = -1
        self._adj_cache = None
        return H

    # ------------------------------------------------------------------- hook
    def adapt_topology(self, G, iteration, max_iterations, gbest_fitness):
        if iteration - self._epoch_start_iter < self.epoch and self.current_arm is not None:
            return G
        if self.current_arm is not None:
            self.history.append((self.current_arm,
                                 self._credit(self._epoch_start_best, gbest_fitness)))
            self.trace.append((iteration, self.current_arm, self.history[-1][1]))
        self.current_arm = self._select_arm()
        self._epoch_start_best = gbest_fitness
        self._epoch_start_iter = iteration
        return self._rebuild(G, self.current_arm)


def make_adaptive_config(nodes: int = 60, dim: int = 30) -> Config:
    """Config for ADS: uniform mixing, no betweenness, adaptive degree."""
    cfg = Config()
    cfg.NUM_NODES, cfg.DIM = nodes, dim
    cfg.VECTORIZED, cfg.SHOW_PROGRESS = True, False
    cfg.NEIGHBOR_MODE = "uniform"          # centrality weighting is a measured placebo
    cfg.CENTRALITY_MODE = "degree"         # cheap; unused by uniform mixing
    cfg.REWIRE_MODE = "none"               # the bandit owns the topology
    return cfg
