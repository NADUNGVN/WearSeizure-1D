"""What is every host actually running right now, and does anything collide?

SERVER-02/03/04 share ~/Manh over NFS, so one invocation on any of them sees all
three. Hydra names its run directory `<host>_<timestamp>` and drops the exact
overrides in `.hydra/overrides.yaml`, so the shared disk already records who is
running what -- this just reads it back.

The part worth having is the collision check. Two hosts training the same
(model, seed) write the same cohort-initialisation .pt files, and one can read a
file the other is still writing. That does not raise; it produces a wrong result
quietly, which is the failure mode this project keeps paying for.

    python scripts/check_servers.py [--minutes 20]

Read-only.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

INTERESTING = ("model=", "seed=", "+rung=", "train.run_tag=", "train.seeds=")


def newest_mtime(d: Path) -> float:
    """Most recent mtime anywhere under `d` -- a run dir's own mtime does not
    move while its log file grows."""
    best = 0.0
    for p in d.rglob("*"):
        try:
            best = max(best, p.stat().st_mtime)
        except OSError:
            continue
    return best or d.stat().st_mtime


def overrides_of(run_dir: Path) -> list[str]:
    f = run_dir / ".hydra" / "overrides.yaml"
    if not f.is_file():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        item = line.strip().lstrip("-").strip().strip("'\"")
        if any(item.startswith(k) for k in INTERESTING):
            out.append(item)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=20,
                    help="a run is 'live' if something under it changed this recently")
    args = ap.parse_args()

    art = Path(os.environ.get("WEARSEIZURE_ARTIFACTS_DIR", ""))
    runs = art / "runs"
    if not runs.is_dir():
        print(f"no runs directory under {art}")
        return 2

    now = time.time()
    cutoff = now - args.minutes * 60
    live: list[tuple[str, float, list[str], Path]] = []
    unqualified = 0
    for d in runs.iterdir():
        if not d.is_dir():
            continue
        m = newest_mtime(d)
        if m < cutoff:
            continue
        host = d.name.split("_", 1)[0] if "_" in d.name and not d.name[0].isdigit() else "?"
        if host == "?":
            unqualified += 1
        live.append((host, m, overrides_of(d), d))

    if not live:
        print(f"nothing has written under {runs} in the last {args.minutes} minutes.")
        print("Either everything finished, or everything died. Check the phase logs.")
        return 1

    # Local processes first. A run directory only shows that SOMETHING is
    # writing; it cannot show that two processes on this host write to the same
    # place. `pkill -f run_phase7_capacity` kills the bash script and leaves its
    # `python train.py` child running -- an orphan that keeps training, and that
    # anything reading only the artifacts tree cannot see.
    print("=== processes on this host ===")
    try:
        out = subprocess.run(["ps", "-eo", "pid,etime,args"],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        out = ""
    mine = [x for x in out.splitlines()
            if ("scripts/train.py" in x or "run_leaky_repro.py" in x) and "grep" not in x]
    if not mine:
        print("  no training process here (this host may simply not be the one running it)")
    for line in mine:
        pid, etime, args = line.split(None, 2)
        keep = " ".join(a for a in args.split()
                        if a.startswith(("model=", "seed=", "+rung=", "train.run_tag=")))
        print(f"  pid {pid:<8} up {etime:<12} {keep or args[:70]}")
    if len(mine) > 1:
        print(f"  -> {len(mine)} training processes on this host. If a phase script was")
        print("     pkill'd, its python child survives -- pkill matches the script name,")
        print("     not the child, and the orphan keeps writing.")
    print()

    print(f"live runs (touched in the last {args.minutes} min), newest first:\n")
    for host, m, ov, d in sorted(live, key=lambda r: -r[1]):
        age = (now - m) / 60
        print(f"  {host:<12} {age:5.1f} min ago   {' '.join(ov) if ov else '(no overrides recorded)'}")
        print(f"  {'':<12} {d}")

    if unqualified:
        print(f"\n{unqualified} run director(ies) have no host in the name -- they were started")
        print("before the run dir was host-qualified. Their host cannot be told from disk.")

    # Collisions. Two hosts on the same work unit is the thing that corrupts.
    print("\n=== collision check ===")
    work: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for host, _m, ov, _d in live:
        key = tuple(sorted(o for o in ov if o.startswith(("model=", "seed=", "+rung="))))
        if key:
            work[key].add(host)
    clashes = {k: v for k, v in work.items() if len(v) > 1}
    if clashes:
        for key, hosts in clashes.items():
            print(f"  CLASH  {' '.join(key)}  is live on {', '.join(sorted(hosts))}")
        print("\n  Two hosts on one (model, seed) write the same cohort-init .pt files and")
        print("  can read each other's partial writes. Stop one of them.")
    else:
        print("  none: no (model, seed, rung) is live on more than one host.")

    hosts_seen = {h for h, *_ in live if h != "?"}
    print(f"\nhosts with live work: {', '.join(sorted(hosts_seen)) or 'none identifiable'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
