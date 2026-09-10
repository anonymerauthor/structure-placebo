"""Launch the FIPS audit as N independent sharded processes.

Parallelism comes from separate processes rather than a worker pool: the pool
backend fails intermittently during spawn on this platform, and since every
run checkpoints to its own file the two are equivalent.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=6)
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--dims", nargs="*", default=["30"])
ap.add_argument("--arms", nargs="*", default=None)
a = ap.parse_args()

os.makedirs(os.path.join(HERE, "outputs", "fips"), exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
           PYTHONIOENCODING="utf-8")
base = [sys.executable, "-u", os.path.join(HERE, "experiments", "fips_study.py"),
        "--runs", str(a.runs), "--dims", *a.dims, "--nshards", str(a.shards)]
if a.arms:
    base += ["--arms", *a.arms]

procs, logs = [], []
for i in range(a.shards):
    log = open(os.path.join(HERE, "outputs", f"fips_shard{i}.log"), "w")
    logs.append(log)
    procs.append(subprocess.Popen(base + ["--shard", str(i)], cwd=HERE,
                                  env=env, stdout=log, stderr=subprocess.STDOUT))
print(f"launched {a.shards} shards | runs={a.runs} dims={a.dims}", flush=True)

t0 = time.perf_counter()
for p in procs:
    p.wait()
for log in logs:
    log.close()
done = len(os.listdir(os.path.join(HERE, "outputs", "fips")))
print(f"exit {[p.returncode for p in procs]} | {done} results | "
      f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
