"""Report experiment progress, optionally publishing it for remote viewing.

The runner machine writes a short STATUS.md and force-pushes it to an orphan
`status` branch. Anyone can then watch progress by refreshing that branch on
the web, with no SSH, no port forwarding and no shared filesystem. The branch
is orphan and force-updated, so `main` keeps a clean history.

    python scripts/status.py                 # print
    python scripts/status.py --publish       # print and push to `status`
    python scripts/status.py --watch 600     # publish every 10 minutes
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STUDIES = [
    # label, records dir, expected count
    ("CEC2017 audit", os.path.join("outputs", "cec2017"), None),
    ("FIPS audit", os.path.join("outputs", "fips"), 2520),
]


def summarise() -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"# Run status", "", f"Updated {now}", ""]
    for label, rel, expected in STUDIES:
        d = os.path.join(HERE, rel)
        if not os.path.isdir(d):
            continue
        files = os.listdir(d)
        if not files:
            continue
        recs, secs = {}, []
        for fn in files:
            try:
                r = json.load(open(os.path.join(d, fn)))
            except (ValueError, OSError):
                continue
            key = (r.get("variant"), r.get("dim"))
            recs[key] = recs.get(key, 0) + 1
            if "seconds" in r:
                secs.append(r["seconds"])
        total = sum(recs.values())
        head = f"## {label} — {total}" + (f" / {expected}" if expected else "")
        out += [head, "", "| arm | dim | runs |", "|---|---|---|"]
        for (v, dim), n in sorted(recs.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][0]))):
            out.append(f"| {v} | {dim} | {n} |")
        if secs:
            out += ["", f"mean {sum(secs)/len(secs):.0f} s/run"]
        # throughput over the last hour, the only honest basis for an ETA
        cutoff = time.time() - 3600
        recent = sum(1 for fn in files
                     if os.path.getmtime(os.path.join(d, fn)) > cutoff)
        out += [f"last hour: {recent} runs", ""]
        if expected and recent:
            left = expected - total
            if left > 0:
                out.append(f"remaining {left} runs; at the last hour's rate "
                           f"about {left/recent:.1f} h, though cost per run "
                           f"varies by more than tenfold across the suite\n")
    alive = _runners()
    out += [f"worker processes: {alive}", ""]
    return "\n".join(out)


def _runners() -> int:
    try:
        import psutil
    except ImportError:
        return -1
    n = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if not (p.info["name"] or "").lower().startswith("python"):
                continue
            cmd = p.info["cmdline"] or []
            if any(os.path.basename(str(c)) in
                   ("run_cec.py", "fips_study.py") for c in cmd):
                n += 1
        except Exception:
            pass
    return n


def publish(text: str) -> None:
    """Force-update an orphan `status` branch holding only STATUS.md."""
    path = os.path.join(HERE, "STATUS.md")
    open(path, "w", encoding="utf-8").write(text)
    cmds = [
        ["git", "checkout", "--orphan", "_status_tmp"],
        ["git", "reset"],
        ["git", "add", "-f", "STATUS.md"],
        ["git", "-c", "user.name=anonymerauthor",
         "-c", "user.email=anonymerauthor@users.noreply.github.com",
         "commit", "-q", "-m", "status"],
        ["git", "branch", "-M", "status"],
        ["git", "push", "-f", "origin", "status"],
    ]
    prev = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                          cwd=HERE, capture_output=True, text=True).stdout.strip()
    try:
        for c in cmds:
            r = subprocess.run(c, cwd=HERE, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  {' '.join(c)} failed: {r.stderr.strip()}", file=sys.stderr)
                return
        print("  published to branch `status`")
    finally:
        subprocess.run(["git", "checkout", "-f", prev or "main"],
                       cwd=HERE, capture_output=True, text=True)
        subprocess.run(["git", "branch", "-D", "status"],
                       cwd=HERE, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--watch", type=int, metavar="SECONDS")
    a = ap.parse_args()
    while True:
        text = summarise()
        print(text)
        if a.publish or a.watch:
            publish(text)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
