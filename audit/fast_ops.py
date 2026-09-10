"""Vectorized inner-loop operations for SOCIAL.

Replaces the per-node Python loops in ``SOCIALOptimizer.diffuse`` with sparse
matrix algebra. Neighbour aggregation is a row-stochastic mixing operator, so
the whole population update becomes two sparse matvecs:

    X_cent = Wc @ X      X_inf = Wi @ X

The adjacency structure only changes when the graph is rewired, so it is cached
and keyed on ``G.graph['_rewire_count']`` (edge swaps preserve both edge count
and degree sequence, so neither is a valid cache key). Per iteration only the
weight vectors change, which is a rescaling of the cached CSR data array.

Results agree with the scalar path to floating-point round-off; they are not
bit-identical because the summation order differs.
"""

import numpy as np
import networkx as nx
import scipy.sparse as sp
from typing import Dict, List, Tuple


class AdjacencyCache:
    """Fixed CSR sparsity pattern of the population graph."""

    __slots__ = ("n", "indptr", "cols", "rows", "isolated", "key")

    def __init__(self, G: nx.Graph, node_list: List):
        A = nx.to_scipy_sparse_array(G, nodelist=node_list, format="csr", dtype=np.float64)
        self.n = len(node_list)
        self.indptr = A.indptr.astype(np.int64, copy=True)
        self.cols = A.indices.astype(np.int64, copy=True)
        # Row index of every stored entry, for segment sums via bincount.
        self.rows = np.repeat(np.arange(self.n, dtype=np.int64), np.diff(self.indptr))
        self.isolated = np.diff(self.indptr) == 0
        self.key = cache_key(G, node_list)

    def row_stochastic(self, weights: np.ndarray) -> sp.csr_matrix:
        """Row-normalized matrix whose (i, j) entry is weights[j] for j in N(i)."""
        data = weights[self.cols]
        rowsum = np.bincount(self.rows, weights=data, minlength=self.n)
        inv = np.zeros(self.n, dtype=np.float64)
        good = rowsum > 0
        inv[good] = 1.0 / rowsum[good]
        return sp.csr_matrix((data * inv[self.rows], self.cols, self.indptr),
                             shape=(self.n, self.n))


def cache_key(G: nx.Graph, node_list: List) -> Tuple:
    """Identity of the graph's edge structure for cache validation."""
    return (G.graph.get('_rewire_count', 0), G.number_of_nodes(), G.number_of_edges(), len(node_list))


def build_mixing_matrices(G: nx.Graph,
                          node_list: List,
                          centrality: Dict,
                          influence: np.ndarray,
                          neighbor_mode: str = "centrality_weighted",
                          eps: float = 1e-10,
                          cache: AdjacencyCache = None) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, AdjacencyCache]:
    """Build the centrality- and influence-weighted row-stochastic mixing matrices.

    Args:
        G: Population graph.
        node_list: Node ordering; row/column i corresponds to node_list[i].
        centrality: node -> centrality value.
        influence: Influence values indexed by position in node_list.
        neighbor_mode: "centrality_weighted" or "uniform".
        eps: Additive floor matching the scalar path's zero-guard.
        cache: Previously built cache; rebuilt if stale.

    Returns:
        (Wc, Wi, isolated, cache)
    """
    if cache is None or cache.key != cache_key(G, node_list):
        cache = AdjacencyCache(G, node_list)

    if neighbor_mode == "centrality_weighted":
        c_vec = np.fromiter((centrality.get(node, 0.0) for node in node_list),
                            dtype=np.float64, count=len(node_list)) + eps
        i_vec = np.asarray(influence, dtype=np.float64) + eps
        Wc = cache.row_stochastic(c_vec)
        Wi = cache.row_stochastic(i_vec)
    else:  # uniform: both aggregates are the plain neighbour mean
        Wc = cache.row_stochastic(np.ones(len(node_list), dtype=np.float64))
        Wi = Wc
    return Wc, Wi, cache.isolated, cache


def repair_boundary_vec(X: np.ndarray, bounds: List[float], mode: str = "reflect") -> np.ndarray:
    """Vectorized counterpart of ``SOCIALOptimizer.repair_boundary``.

    Applies to an (N, D) array; semantics match the scalar version element-wise.
    """
    lower, upper = bounds[0], bounds[1]
    if mode == "clip":
        return np.clip(X, lower, upper)

    range_size = upper - lower
    R = X.copy()

    below = R < lower
    if below.any():
        excess = lower - R[below]
        v = lower + np.mod(excess, 2.0 * range_size)
        over = v > upper
        v[over] = 2.0 * upper - v[over]
        R[below] = v

    above = R > upper
    if above.any():
        excess = R[above] - upper
        v = upper - np.mod(excess, 2.0 * range_size)
        under = v < lower
        v[under] = 2.0 * lower - v[under]
        R[above] = v

    return np.clip(R, lower, upper)
