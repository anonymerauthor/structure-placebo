# structure-placebo

Placebo controls for structure-aware metaheuristics.

A large family of population-based optimizers places candidate solutions on a
graph, computes a structural score on it — centrality, degree, bridge
potential — and weights the update so that agents learn preferentially from
neighbours the score distinguishes. The justification is causal: the method
works *because* the structure identifies informative agents. Where that claim
is tested at all, it is tested by ablation.

Ablation cannot establish it. Removing a weighting changes the optimizer's
mixing operator, its effective neighbourhood and its convergence rate, so a
difference in outcome is equally consistent with the weights having been
merely unequal. The claim being made is narrower: that the *assignment* of
scores to graph positions is informative.

This repository implements two controls that test that claim directly, the
theory that makes them sound, and a large-scale application of both.

---

## The two controls

**Value permutation.** Replace the structural score `c` by `c ∘ π` for a
random permutation `π`. The multiset of scores, the weight heterogeneity and
the arithmetic cost are preserved exactly; only the correspondence between a
node's position and the weight its neighbours give it is destroyed.

The control is sound because of a small result proved in the paper: under a
uniform permutation the resulting operator equals uniform mixing *in
expectation* while retaining the original's weight heterogeneity in *every
realisation*. It therefore sits exactly between the two alternatives an
ablation confounds, holding heterogeneity fixed while removing assignment.

**Policy randomisation.** For an adaptive component, replace its selection
rule by a uniform draw over the same alternatives on the same schedule. This
separates *selecting* a component from merely *varying* it.

A control is only informative if the mechanism it neutralises is active, so
every comparison is reported alongside the displacement the mechanism induces,
measured in units of the population's own spread. A companion bound identifies
the topologies — vertex-transitive ones, the ring lattice among them — on which
such a mechanism is inert by construction and the comparison therefore vacuous.

## What the audit found

Applied to a published betweenness-weighted small-world optimizer:

| Mechanism | Suite mean Cliff's δ | 90% CI | Reading |
|---|---|---|---|
| Centrality assignment | −0.010 | [−0.038, +0.020] | equivalent to zero |
| Fitness influence | −0.051 | [−0.117, +0.008] | intermediate, class-dependent |
| Adaptive density selection | −0.101 | [−0.149, −0.061] | clear effect |

The centrality mechanism is *active* — it displaces every update by roughly a
third of the population's spread — and *uninformative*: over 76 function
comparisons on two suites, two dimensions and four budgets, permuting the
scores changes nothing that survives correction.

The other two rows are the calibration. The same instrument, unchanged in
suite, sample size and statistic, returns an intermediate effect on one further
mechanism and a clear one on another, so the null cannot be attributed to a
test without power.

Separating the two properties an ablation confounds shows why: unequal weights
help on 3 of 28 functions, while *which* neighbour receives the large weight
helps on none. Betweenness accounts for about 30% of per-iteration runtime and
can be replaced by any heterogeneous weighting at no cost in solution quality.

What does carry information is not a score computed on the graph but the
graph's coarse density, consequential in 30 of 49 problem configurations —
though with an optimum that is problem- and dimension-dependent and resists
prediction from landscape ruggedness.

## Repository layout

```
audit/          the two controls and the vectorized diffusion path
experiments/    one module per study; each writes per-run JSON records
scripts/        bootstrap, sharded runner, pipeline driver, equivalence tests
patches/        the complete set of changes made to the audited implementation
data/           consolidated per-run results as CSV
paper/          manuscript sources, figure and table generators
```

## Reproducing

The audit runs on the published implementation itself, not on a
reimplementation. That is deliberate: a reimplementation would leave open
whether a null result reflects the method or our version of it. The
implementation is published without a licence, so it is not redistributed
here — fetch it and apply the patches:

```bash
pip install -r requirements.txt

git clone https://github.com/MehrdadJalali-AI/SOCIAL-OPTIMIZATION upstream
cd upstream
git apply ../patches/optimizer.patch ../patches/graph_ops.patch ../patches/config.patch
cp ../audit/*.py social/
cp -r ../experiments ../scripts .
touch experiments/__init__.py
```

The patches are the complete diff — 93 lines across three files — and consist
of the two permutation controls, a topology-adaptation hook, a sparse-matrix
diffusion path equivalent to the original to 9e-16 per step, and a fix for an
exception class that crashes the original on dense graphs. Everything else in
this repository is our own.

If `git apply` reports a line-ending mismatch, normalise first:

```bash
python -c "import pathlib
for n in ('optimizer','graph_ops','config'):
    p = pathlib.Path('social')/f'{n}.py'
    p.write_text(p.read_text(encoding='utf-8').replace('\r\n','\n'), encoding='utf-8', newline='\n')"
```

Then run any study. Each writes one JSON file per run, keyed by
`(variant, function, dimension, seed)`, and skips work already on disk, so runs
are resumable and any number in the paper is traceable to the run that produced
it.

```bash
python scripts/pipeline.py                       # the full sequence
python -m experiments.cec2017_study --help       # or one study at a time
python -m experiments.equivalence --csv ... --left social --right shuffled
```

Parallelism comes from independent sharded processes rather than a worker pool;
`launch_cec.py` starts them. On the machine used here a pool backend failed
intermittently during process spawn, and sharding is equivalent because every
run checkpoints separately.

### Running on another machine

Only the per-run records need moving; the code and consolidated CSVs are in
git. They are one small JSON per run and compress to well under a megabyte.

On the machine that holds the current state:

```bash
python scripts/bundle_state.py --out state.tar.gz
```

On the machine that will run the experiments:

```bash
git clone https://github.com/anonymerauthor/structure-placebo
cd structure-placebo
pip install -r requirements.txt
python scripts/bundle_state.py --restore ../state.tar.gz
```

Then start any study; finished runs are skipped, so the work resumes rather
than repeats.

To follow progress from elsewhere, have the runner publish it:

```bash
python scripts/status.py --watch 900     # every 15 minutes
```

This force-updates an orphan `status` branch holding a single `STATUS.md`,
which can be read on the web from anywhere. It needs no SSH, no port
forwarding and no shared filesystem, and `main` keeps a clean history.

### Regenerating figures and tables

```bash
python paper/make_figures.py     # reads data/, writes paper/figures/*.pdf
python paper/make_tables.py      # writes paper/tables/*.tex
cd paper && latexmk -pdf main.tex
```

## Protocol

Statistical comparisons are paired by seed, so a variant and its control see
the same initial population and the same random stream. Per function we report
the median, Cliff's δ with a bootstrap interval, and a Wilcoxon signed-rank
test under Holm correction; across a suite we report the mean effect with its
interval and a one-sample test.

Both views are reported throughout, because either alone can mislead. A
per-function table under Holm correction hides an effect spread thinly across
a suite — that is how our own adaptive scheme first appeared weak when it is
not. A suite mean alone hides an effect concentrated on a few functions.

Because several claims here are of equivalence rather than difference, we test
equivalence directly with two one-sided tests rather than resting on a
non-significant difference. Equivalence at margin *m* requires the point
estimate *and* its interval inside the margin, which is more demanding than
narrowing the interval alone.

Benchmarks are the classic 23-function suite and CEC2017 at *D* ∈ {10, 30, 50}
under the competition protocol: error `f(x) − f*`, budget `10⁴·D`, domain
`[−100, 100]^D`. Two CEC2017 functions are absent — F2 on the organisers'
advice, F30 because the reference implementation omits it.

## Corrections to the audited implementation

Three defects affect measurement and are corrected in the patches. Fixed-
dimension functions F14–F23 were evaluated at *D* = 30 although their
implementations slice the first two to six coordinates, leaving up to 28
dimensions inert; neighbourhood size was set per function category, which is
per-instance tuning; and the rewiring routine caught only `NetworkXError`
while `double_edge_swap` raises `NetworkXAlgorithmError`, which is not a
subclass, so dense graphs crashed the optimizer.

## Raw data

`data/` holds consolidated CSVs. The complete set of individual run records —
roughly 16,000 JSON files — is attached to the corresponding release rather
than tracked in git.

## Citing

See `CITATION.cff`. The audited optimizer is Jalali et al., *Applied Soft
Computing* 194 (2026) 114914; its implementation is at
<https://github.com/MehrdadJalali-AI/SOCIAL-OPTIMIZATION>.

## Licence

MIT for the contents of this repository. The audited implementation is not
included and is subject to its own terms.
