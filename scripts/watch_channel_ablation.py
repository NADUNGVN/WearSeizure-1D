"""Where is the channel ablation, and is it healthy?

    python scripts/watch_channel_ablation.py <artifacts_dir> [--watch] [--folds 66]

The same four questions a multi-hour run raises -- is exactly ONE of it
running, how far along is each arm, how much longer, and has anything gone
wrong that a count alone would hide.

It deliberately mirrors `watch_dfp_eval.py` rather than sharing code with it.
The two watch different directory layouts and different failure modes, and a
shared abstraction over two callers would have to be rewritten the moment a
third appears. If a third does appear, that is the time to factor it out.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

STALL_AFTER_S = 1800


def running_processes() -> list[str]:
    try:
        out = subprocess.run(["pgrep", "-af", "run_channel_ablation"],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [ln for ln in out.stdout.strip().splitlines() if ln.strip()]


def arm_status(arm_dir: Path) -> dict:
    files = sorted(arm_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    seconds, bad = [], []
    for path in files:
        try:
            seconds.append(json.loads(path.read_text(encoding="utf-8"))["seconds"])
        except (json.JSONDecodeError, KeyError):
            # A file caught mid-write, or one written by an older format.
            bad.append(path.name)
    return {
        "name": arm_dir.name,
        "done": len(files),
        # Per-fold cost measured by the run itself, which is a better estimate
        # than dividing wall-clock by count: it excludes the time the process
        # spent on folds it skipped.
        "sec_per_fold": (sum(seconds) / len(seconds)) if seconds else None,
        "last_write": files[-1].stat().st_mtime if files else None,
        "bad": bad,
    }


def render(root: Path, total_folds: int) -> bool:
    procs = running_processes()
    print(f"\n{time.strftime('%H:%M:%S')}  processes: ", end="")
    if not procs:
        print("none running")
    elif len(procs) == 1:
        print("1 running")
    else:
        print(f"** {len(procs)} RUNNING -- they will overwrite each other **")
        for line in procs:
            print(f"    {line}")

    base = root / "channel_ablation"
    arms = sorted(base.glob("*ch"), key=lambda p: int(p.name.rstrip("ch")))
    if not arms:
        print(f"  no results yet under {base}")
        return False

    all_done = True
    for arm in arms:
        st = arm_status(arm)
        line = f"  {st['name']:>5} {st['done']:3d} / {total_folds}"
        if st["sec_per_fold"]:
            line += f"   {st['sec_per_fold']:.0f}s/fold"
            if st["done"] < total_folds:
                left = (total_folds - st["done"]) * st["sec_per_fold"] / 3600
                line += f"   ~{left:.1f} h left"
        if st["done"] >= total_folds:
            line += "   COMPLETE"
        else:
            all_done = False
            if st["last_write"] and time.time() - st["last_write"] > STALL_AFTER_S:
                line += (f"   STALLED? nothing written for "
                         f"{(time.time() - st['last_write']) / 60:.0f} min")
        print(line)
        for name in st["bad"][:3]:
            print(f"      unreadable: {name}")

    done = all_done and not procs
    if done:
        print("\n  Every arm complete. The comparison:")
        print(f"    python scripts/summarise_channel_ablation.py {root}")
    elif any(arm_status(a)["done"] >= total_folds for a in arms):
        print("\n  Some arms done, something still running. What is finished reads:")
        print(f"    python scripts/summarise_channel_ablation.py {root}")
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("--folds", type=int, default=66,
                    help="folds per arm per seed (66 for the 13-case protocol)")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=600)
    args = ap.parse_args()

    root = Path(args.artifacts_dir)
    while True:
        if render(root, args.folds) or not args.watch:
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
