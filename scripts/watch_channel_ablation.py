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


def running_processes(pattern: str = "run_channel_ablation") -> list[str]:
    """Only the ROOT processes matching `pattern`, never their workers.

    A DataLoader worker is a fork, so it inherits the parent's entire command
    line and `pgrep -af` reports it as another run. One healthy run with twelve
    workers then shows as thirteen "processes", which reads as exactly the
    data-loss condition this watcher exists to catch -- and the natural reaction
    to that alarm is to kill something.

    So: match on the command line, then keep only those whose parent is not
    itself a match. A genuine second run is orphaned to init or owned by a
    shell, never by another matching python.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,etimes=,args="],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    matched: dict[int, tuple[int, int, str]] = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, etimes, args = parts
        if pattern in args and "watch_" not in args:
            matched[int(pid)] = (int(ppid), int(etimes), args)
    return [f"{pid} [{age}s] {args}"
            for pid, (ppid, age, args) in sorted(matched.items())
            if ppid not in matched]


def youngest_run_age_s(procs: list[str]) -> int | None:
    """Seconds since the newest matching run started, or None if none run.

    A run that started five minutes ago cannot have stalled, however long ago
    the previous run's last result file was written. Without this the watcher
    greeted every restart with three STALLED warnings, because the clock it was
    reading was the clock of a run that had finished a day and a half earlier.
    """
    ages = [int(p.split("[", 1)[1].split("s]", 1)[0]) for p in procs if "[" in p]
    return min(ages) if ages else None


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
    run_age = youngest_run_age_s(procs)
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
            idle = time.time() - st["last_write"] if st["last_write"] else 0
            # A stall needs BOTH: nothing written recently, AND a run that has
            # been alive long enough to have written something. On a restart the
            # first condition is true by construction -- the last file belongs to
            # the previous run -- and reporting that as a stall is a false alarm
            # on a healthy run, which is the alarm people learn to ignore.
            if idle > STALL_AFTER_S and (run_age or 0) > STALL_AFTER_S:
                line += f"   STALLED? nothing written for {idle / 60:.0f} min"
            elif not procs:
                line += "   stopped"
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
