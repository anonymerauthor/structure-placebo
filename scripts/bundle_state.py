"""Pack the per-run records for transfer to another machine.

Only the raw records need moving: the code and the consolidated CSVs are in
git. Records are one small JSON per run and compress to a fraction of their
size, so the whole experimental state fits in a single small archive.

    python scripts/bundle_state.py --out state.tar.gz          # pack
    python scripts/bundle_state.py --restore state.tar.gz      # unpack
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (label, path relative to the repo root)
TREES = [
    ("cec2017", os.path.join("outputs", "cec2017")),
    ("fips", os.path.join("outputs", "fips")),
]


def pack(out: str, extra: list[str]) -> None:
    roots = [(lbl, os.path.join(HERE, rel)) for lbl, rel in TREES]
    roots += [(os.path.basename(p.rstrip("/\\")), os.path.abspath(p))
              for p in extra]
    present = [(lbl, p) for lbl, p in roots if os.path.isdir(p)]
    if not present:
        sys.exit("nothing to pack; no record directories found")

    with tarfile.open(out, "w:gz") as tar:
        for lbl, path in present:
            n = len(os.listdir(path))
            tar.add(path, arcname=os.path.join("outputs", lbl))
            print(f"  packed {lbl:10s} {n:6d} records")
        # Working notes are not in git; they belong with the state.
        notes = os.path.join(HERE, "HANDOFF.md")
        if os.path.isfile(notes):
            tar.add(notes, arcname="HANDOFF.md")
            print("  packed HANDOFF.md")
    print(f"-> {out}  ({os.path.getsize(out)/1e6:.1f} MB)")


def restore(archive: str) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        # Refuse absolute or parent-escaping paths from an untrusted archive.
        for m in tar.getmembers():
            if os.path.isabs(m.name) or ".." in m.name.split("/"):
                sys.exit(f"unsafe path in archive: {m.name}")
        tar.extractall(HERE)
    for lbl, rel in TREES:
        p = os.path.join(HERE, rel)
        if os.path.isdir(p):
            print(f"  restored {lbl:10s} {len(os.listdir(p)):6d} records")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="state.tar.gz")
    ap.add_argument("--restore")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional record directories to include")
    a = ap.parse_args()
    if a.restore:
        restore(a.restore)
    else:
        pack(a.out, a.extra)


if __name__ == "__main__":
    main()
