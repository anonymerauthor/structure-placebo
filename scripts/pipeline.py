"""Unattended CEC2017 pipeline: run rounds back to back, analyse between them.

Each round is a set of sharded processes. The next round only starts once the
previous one's processes are gone, so the six cores are never oversubscribed
and never left idle. Every run checkpoints to its own file, so re-running the
pipeline resumes rather than repeats.
"""
import os, subprocess, sys, time, json
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(HERE, "outputs", "pipeline_status.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(STATUS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def runners_alive():
    """Count live shard processes.

    Matching a substring of the whole command line is not enough: any shell
    that merely mentions run_cec.py -- a status check, a grep -- looks like a
    runner and stalls the pipeline indefinitely. Require a Python executable
    with run_cec.py as an argument in its own right.
    """
    me = os.getpid()
    n = 0
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] == me:
                continue
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            cmd = p.info["cmdline"] or []
            if any(os.path.basename(str(c)) == "run_cec.py" for c in cmd):
                n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


def wait_for_idle(poll=60):
    while True:
        n = runners_alive()
        if n == 0:
            return
        time.sleep(poll)


def count(pattern_parts):
    d = os.path.join(HERE, "outputs", "cec2017")
    if not os.path.isdir(d):
        return 0
    return sum(1 for f in os.listdir(d) if all(p in f for p in pattern_parts))


def rebuild_csv(variants, dims, runs, out):
    """Aggregate the per-run files this stage covers.

    Not left to the worker: shard 0 writes its CSV when it finishes, which can
    precede the other shards (an earlier round lost four runs that way), and it
    aggregates every file in the directory regardless of dimension or arm.
    """
    import json as _json
    import pandas as _pd
    d = os.path.join(HERE, "outputs", "cec2017")
    want_d = {f"_D{x}_" for x in dims}
    want_v = set(variants)
    rows = []
    for fn in os.listdir(d):
        if not any(w in fn for w in want_d):
            continue
        try:
            rec = _json.load(open(os.path.join(d, fn)))
        except (ValueError, OSError):
            continue
        # Filter on the record's own variant field. Matching a filename prefix
        # is wrong here: "shuffled_" is a prefix of "shuffled_inf_...", which
        # silently folded the positive-control runs into the control arm.
        if rec.get("variant") in want_v:
            rows.append(rec)
    df = _pd.DataFrame(rows)
    if len(df):
        df = df[df.seed < runs]
        df.to_csv(os.path.join(HERE, out), index=False)
    expected = len(variants) * 28 * len(dims) * runs
    log(f"  aggregated {len(df)}/{expected} runs -> {out}")
    return len(df)


def launch(variants, dims, runs, out, shards=6):
    cmd = [sys.executable, "-u", os.path.join(HERE, "launch_cec.py"),
           "--shards", str(shards), "--runs", str(runs),
           "--dims", *[str(d) for d in dims], "--out", out,
           "--variants", *variants]
    log(f"START variants={variants} dims={dims} runs={runs}")
    subprocess.run(cmd, cwd=HERE, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    log(f"DONE  variants={variants} dims={dims}")
    rebuild_csv(variants, dims, runs, out)


def analyse(csv, refs, tag):
    """Difference tests for one stage.

    CEC result files key functions by `fid` and outcomes by `error`, the
    classic-suite files by `function` and `best`; the two analysers are not
    interchangeable and using the wrong one fails with a KeyError that the
    stage log swallows.
    """
    out = os.path.join(HERE, "outputs", f"stats_{tag}.txt")
    if "cec" in os.path.basename(csv).lower():
        cmd = [sys.executable, "-u", "-m", "experiments.cec_stats",
               "--csv", csv, "--left", "social", "--refs", *refs]
    else:
        cmd = [sys.executable, "-u", "-m", "experiments.placebo_stats",
               "--csv", csv, "--refs", *refs]
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=1800)
        open(out, "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
        log(f"ANALYSIS {tag} -> {out}")
    except Exception as exc:
        log(f"ANALYSIS {tag} FAILED: {type(exc).__name__}: {exc}")


def equivalence(csv, left, right, tag):
    """TOST equivalence report for one contrast."""
    out = os.path.join(HERE, "outputs", f"tost_{tag}.txt")
    cmd = [sys.executable, "-u", "-m", "experiments.equivalence",
           "--csv", csv, "--left", left, "--right", right,
           "--out", os.path.join(HERE, "outputs", f"tost_{tag}.csv")]
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=3600)
        open(out, "w", encoding="utf-8").write(r.stdout + "\n" + r.stderr)
        log(f"TOST {tag} ({left} vs {right}) -> {out}")
    except Exception as exc:
        log(f"TOST {tag} FAILED: {type(exc).__name__}: {exc}")


ALL6 = ["social", "shuffled", "uniform_K4", "fixed_K16", "random_K", "ads"]
REST4 = ["uniform_K4", "fixed_K16", "random_K", "ads"]
# At D=50 only the two control contrasts are run. The density arms would only
# duplicate the 7350-run K-sweep, which already covers D in {10, 30, 50}.
CTRL_MAIN = ["social", "shuffled"]      # primary claim; tests the D-independence
                                        # predicted by the leverage bound
CTRL_ADAPT = ["random_K", "ads"]        # second control, lower priority


def probe_d50(seeds=(0, 1)):
    """Time a few D=50 runs before committing to the round.

    Cost per run varies about twentyfold across the suite, so a rate measured
    on cheap functions badly underestimates the total. The probe samples one
    function from each cost class and extrapolates from the suite's observed
    cost profile at D=30.
    """
    import experiments.cec2017_study as C
    sample = [(1, "unimodal"), (7, "simple multimodal"),
              (16, "hybrid"), (25, "composition")]
    per = {}
    for fid, kind in sample:
        t0 = time.perf_counter()
        for s in seeds:
            C.run_one("social", fid, 50, s, 60)
        per[kind] = (time.perf_counter() - t0) / len(seeds)
        log(f"  probe D=50 F{fid} ({kind}): {per[kind]:.0f} s/run")
    # Suite composition: 2 unimodal, 7 simple multimodal, 10 hybrid, 9 composition
    counts = {"unimodal": 2, "simple multimodal": 7, "hybrid": 10, "composition": 9}
    per_variant = sum(per[k] * n for k, n in counts.items()) * 30      # 30 seeds
    log(f"  probe: {per_variant/3600:.1f} core-h per variant at D=50; "
        f"{2*per_variant/3600/6:.1f} h wall for a 2-variant stage on 6 shards")

if __name__ == "__main__":
    log("pipeline started; waiting for the round already in flight")
    wait_for_idle()
    log(f"round 1 finished: {count(['_D30_'])} D=30 results")

    # Round 2: the other four arms at D=30, completing the main comparison.
    launch(REST4, [30], 30, "outputs/cec2017_D30.csv")
    analyse("outputs/cec2017_D30.csv", ["shuffled", "random_K", "fixed_K16"], "cec_D30")

    # Round 3: D=10, all six arms. Cheap (roughly a tenth of a D=30 run) and
    # it completes a whole dimension of the protocol.
    launch(ALL6, [10], 30, "outputs/cec2017_D10.csv")
    analyse("outputs/cec2017_D10.csv", ["shuffled", "random_K"], "cec_D10")

    # Round 4: raise D=30 to the 51 runs the CEC2017 protocol specifies.
    # TOST showed 30 seeds leave the per-function equivalence claim
    # underpowered: the median 90% interval half-width is 0.181 against a
    # negligibility margin of 0.147, and about 46 seeds are needed. This
    # strengthens the main claim rather than adding a new axis, so it comes
    # before D=50. The primary contrast goes first.
    launch(CTRL_MAIN, [30], 51, "outputs/cec2017_D30_51.csv")
    analyse("outputs/cec2017_D30_51.csv", ["shuffled"], "cec_D30_51_main")
    equivalence("outputs/cec2017_D30_51.csv", "social", "shuffled", "D30_51")

    launch(REST4, [30], 51, "outputs/cec2017_D30_51.csv")
    analyse("outputs/cec2017_D30_51.csv", ["shuffled", "random_K"], "cec_D30_51_full")
    equivalence("outputs/cec2017_D30_51.csv", "ads", "random_K", "D30_51_adaptive")

    # Positive control. The permutation test is applied to the fitness
    # influence score, which is informative by construction, so it must cost
    # performance. Without it a null result on centrality is confounded with
    # the test having no power. This precedes D=50 because it decides whether
    # the negative result means anything.
    launch(["shuffled_inf"], [30], 30, "outputs/cec2017_poscontrol.csv")
    rebuild_csv(["social", "shuffled", "shuffled_inf"], [30], 30,
                "outputs/cec2017_poscontrol.csv")
    analyse("outputs/cec2017_poscontrol.csv", ["shuffled_inf"], "cec_poscontrol")
    equivalence("outputs/cec2017_poscontrol.csv", "social", "shuffled_inf",
                "poscontrol")

    # Round 5: D=50, staged by priority so a stage that has to be cut costs the
    # least important evidence.
    probe_d50()
    launch(CTRL_MAIN, [50], 30, "outputs/cec2017_D50.csv")
    analyse("outputs/cec2017_D50.csv", ["shuffled"], "cec_D50_main")
    equivalence("outputs/cec2017_D50.csv", "social", "shuffled", "D50")

    launch(CTRL_ADAPT, [50], 30, "outputs/cec2017_D50.csv")
    analyse("outputs/cec2017_D50.csv", ["shuffled", "random_K"], "cec_D50_full")

    log("PIPELINE COMPLETE")
