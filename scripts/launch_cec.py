"""Launch the CEC2017 study as N independent sharded processes."""
import argparse, os, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--shards", type=int, default=6)
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--dims", nargs="*", default=["30"])
ap.add_argument("--variants", nargs="*", default=None)
ap.add_argument("--out", default="outputs/cec2017.csv")
a = ap.parse_args()

os.makedirs("outputs/cec2017", exist_ok=True)
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
base = [sys.executable, "-u", "run_cec.py", "--runs", str(a.runs),
        "--dims", *a.dims, "--nshards", str(a.shards), "--out", a.out]
if a.variants:
    base += ["--variants", *a.variants]

procs, logs = [], []
for i in range(a.shards):
    log = open(f"outputs/cec_shard{i}.log", "w")
    logs.append(log)
    procs.append(subprocess.Popen(base + ["--shard", str(i)], env=env,
                                  stdout=log, stderr=subprocess.STDOUT))
print(f"launched {a.shards} shards | dims={a.dims} runs={a.runs} "
      f"variants={a.variants or 'all'}", flush=True)

t0 = time.perf_counter()
for p in procs:
    p.wait()
for log in logs:
    log.close()
print(f"exit codes {[p.returncode for p in procs]} | "
      f"{len(os.listdir('outputs/cec2017'))} results | "
      f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
