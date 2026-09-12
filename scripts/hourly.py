"""Emit one status line per interval; exit when the study completes.

Written for a watcher: each line printed becomes one notification, so the
output is deliberately one compact line rather than a report.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "outputs", "fips")
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 4200
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 3600

prev = -1
while True:
    counts, n = {}, 0
    for fn in os.listdir(D):
        try:
            v = json.load(open(os.path.join(D, fn)))["variant"]
        except (ValueError, OSError, KeyError):
            continue
        counts[v] = counts.get(v, 0) + 1
        n += 1
    cut = time.time() - EVERY
    recent = sum(1 for fn in os.listdir(D)
                 if os.path.getmtime(os.path.join(D, fn)) > cut)
    arms = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    rate = recent / (EVERY / 3600)
    eta = (TARGET - n) / rate if rate else float("inf")
    print(f"FIPS {n}/{TARGET} | {arms} | +{recent} this interval | "
          f"ETA {eta:.1f} h", flush=True)
    if n >= TARGET or n == prev:
        print(f"FIPS WATCH ENDING at {n}/{TARGET}"
              + (" (complete)" if n >= TARGET else " (no progress)"), flush=True)
        break
    prev = n
    time.sleep(EVERY)
