"""Equivalence + speed check for the vectorized diffusion path."""
import os, copy, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from social.config import Config
from social.optimizer import SOCIALOptimizer
from social.budget import BudgetedObjective

RASTRIGIN = lambda x: float(np.sum(x**2 - 10*np.cos(2*np.pi*x) + 10))


def make_state(N, dim, bounds, seed, neighbor_mode, boundary_mode):
    c = Config(); c.NUM_NODES = N; c.SHOW_PROGRESS = False
    c.NEIGHBOR_MODE = neighbor_mode; c.BOUNDARY_MODE = boundary_mode
    o = SOCIALOptimizer(c, np.random.default_rng(seed))
    G = o.initialize_population(dim, bounds, seed=seed)
    obj = BudgetedObjective(RASTRIGIN, 10**7)
    o.evaluate_population(G, obj)
    return c, G, obj


def one_step(vectorized, N, dim, bounds, seed, iteration, neighbor_mode, boundary_mode):
    c, G, obj = make_state(N, dim, bounds, seed, neighbor_mode, boundary_mode)
    c.VECTORIZED = vectorized
    c.ENABLE_MUTATION = False          # isolate the update; mutation consumes RNG
    c.REWIRE_MODE = "none"
    o = SOCIALOptimizer(c, np.random.default_rng(seed))
    nodes = list(G.nodes)
    gbest = G.nodes[nodes[int(np.argmin([G.nodes[n]['fitness'] for n in nodes]))]]['position'].copy()
    elite = gbest.copy()
    G2 = o.diffuse(copy.deepcopy(G), gbest, elite, bounds, iteration, 100, obj)
    return np.array([G2.nodes[n]['position'] for n in nodes])


def test_single_step():
    print("=== Test A: single diffusion step, scalar vs vectorized ===")
    ok = True
    for nm in ("centrality_weighted", "uniform"):
        for bm in ("reflect", "clip"):
            for it in (0, 3):        # it=0 triggers the sync branch
                a = one_step(False, 60, 30, [-5.12, 5.12], 7, it, nm, bm)
                b = one_step(True,  60, 30, [-5.12, 5.12], 7, it, nm, bm)
                d = np.max(np.abs(a - b))
                good = d < 1e-11
                ok &= good
                print(f"  {nm:20s} {bm:7s} iter={it}  max_abs_diff={d:.3e}  {'OK' if good else 'FAIL'}")
    return ok


def test_boundary_repair():
    print("=== Test B: repair_boundary scalar vs vectorized ===")
    from social.fast_ops import repair_boundary_vec
    rng = np.random.default_rng(0)
    bounds = [-5.12, 5.12]
    X = rng.uniform(-40, 40, size=(500, 30))       # heavy out-of-bounds
    ok = True
    for mode in ("reflect", "clip"):
        c = Config(); c.BOUNDARY_MODE = mode
        o = SOCIALOptimizer(c, rng)
        ref = np.array([o.repair_boundary(x.copy(), bounds) for x in X])
        vec = repair_boundary_vec(X, bounds, mode)
        d = np.max(np.abs(ref - vec))
        good = d == 0.0
        ok &= good
        print(f"  {mode:8s} max_abs_diff={d:.3e}  {'OK' if good else 'FAIL'}")
    return ok


def test_full_runs():
    print("=== Test C: full runs, distribution equivalence + speed ===")
    from scipy.stats import mannwhitneyu
    res = {}
    for vec in (False, True):
        vals, t0 = [], time.perf_counter()
        for seed in range(15):
            c = Config(); c.NUM_NODES = 60; c.SHOW_PROGRESS = False; c.VECTORIZED = vec
            o = SOCIALOptimizer(c, np.random.default_rng(seed))
            r = o.optimize(BudgetedObjective(RASTRIGIN, 20000), [-5.12, 5.12], 30, seed=seed)
            vals.append(r['best_fitness'])
        res[vec] = (np.array(vals), time.perf_counter() - t0)
    (vs, ts), (vv, tv) = res[False], res[True]
    u, p = mannwhitneyu(vs, vv)
    print(f"  scalar     : mean={vs.mean():9.4f} std={vs.std():7.4f}  {ts:6.1f}s")
    print(f"  vectorized : mean={vv.mean():9.4f} std={vv.std():7.4f}  {tv:6.1f}s")
    print(f"  Mann-Whitney p={p:.3f} (want > 0.05 -> indistinguishable)")
    print(f"  SPEEDUP    : {ts/tv:.1f}x")
    return p > 0.05


if __name__ == "__main__":
    a = test_single_step()
    b = test_boundary_repair()
    d = test_cache_invalidation()
    c = test_full_runs()
    print(f"
A={'PASS' if a else 'FAIL'}  B={'PASS' if b else 'FAIL'}  "
          f"C={'PASS' if c else 'FAIL'}  D={'PASS' if d else 'FAIL'}")
