"""Where is the integer-datapath run, and is it healthy?

    python scripts/watch_dfp_eval.py <artifacts_dir> [--watch] [--bits 8]

One command instead of a shell block typed out differently each time. It
answers the four questions that actually come up during a multi-hour run:

  is it still going, and is exactly ONE of it going
  how many folds are done, per scoring arm
  how much longer
  did anything go wrong that the progress count alone would hide

The third question is the reason this exists at all: a fold takes about three
minutes and there are 66, so "is it nearly done" is not answerable by looking.

The fourth is the reason it checks more than counts. Two runs writing to one
directory silently replace each other's results, and the only visible symptom
is that totals stop adding up -- which is easy to miss and expensive to
discover later. Both failures are named here explicitly rather than left to be
inferred.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

TOTAL_FOLDS = 66
SECONDS_PER_FOLD_HINT = 170


def running_processes(pattern: str = "export_dfp_hardware") -> list[str]:
    """Only the ROOT export processes, never their workers.

    A DataLoader worker is a fork and inherits the parent's whole command line,
    so `pgrep -af` counts it as another run. One healthy run then reports as
    many "processes", which reads as the overwrite condition this watcher
    exists to catch -- and the natural reaction to that alarm is to kill
    something. Keep only processes whose parent is not itself a match: a
    genuine second run is owned by a shell or by init, never by another
    matching python.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="],
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    matched: dict[int, tuple[int, str]] = {}
    for line in out.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = parts
        if pattern in args and "watch_" not in args:
            matched[int(pid)] = (int(ppid), args)
    return [f"{pid} {args}" for pid, (ppid, args) in sorted(matched.items())
            if ppid not in matched]


def arm_status(arm_dir: Path) -> dict:
    """Count, rate and health for one scoring arm."""
    files = sorted(arm_dir.rglob("*.json"), key=lambda p: p.stat().st_mtime)
    expected_raw = arm_dir.name.endswith("_acc")
    strays = []
    for path in files:
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A file caught mid-write. Real, and not a problem by itself.
            strays.append((path.name, "unreadable"))
            continue
        if bool(d.get("raw_margin", False)) != expected_raw:
            strays.append((path.name, "belongs to the other arm"))

    rate = None
    if len(files) >= 2:
        span = files[-1].stat().st_mtime - files[0].stat().st_mtime
        rate = span / (len(files) - 1)
    return {
        "name": arm_dir.name,
        "done": len(files),
        "rate": rate,
        "last_write": files[-1].stat().st_mtime if files else None,
        "strays": strays,
    }


def arm_label(name: str, bits: int) -> str:
    """Read the two axes back out of a results directory name."""
    suffix = name[len(f"dfp{bits}"):]
    return ("accumulator" if "_acc" in suffix else "logits    ") + (
        " refit " if "_refit" in suffix else " frozen")


def everything_done(base: Path, bits: int, busy: bool) -> bool:
    """Nothing left to wait for: no process running, and every arm at 66.

    The `busy` term is what makes this trustworthy. An arm that has just been
    launched has written no results and so has no directory, and `all` over the
    directories that exist is vacuously true -- which announced completion at
    the moment a fresh arm still had ninety minutes ahead of it. A running
    process is proof that something is still expected.
    """
    if busy:
        return False
    arms = sorted(base.glob(f"dfp{bits}*")) if base.is_dir() else []
    return bool(arms) and all(arm_status(a)["done"] >= TOTAL_FOLDS for a in arms)


def render(root: Path, bits: int) -> None:
    procs = running_processes()
    print(f"\n{time.strftime('%H:%M:%S')}  processes: ", end="")
    if not procs:
        print("none running")
    elif len(procs) == 1:
        print("1 running")
    else:
        # More than one is a data-loss risk, not just a slowdown: they race on
        # the same output paths, and on one machine each also runs slower.
        print(f"** {len(procs)} RUNNING -- they will overwrite each other **")
        for line in procs:
            print(f"    {line}")
        print("    Keep one. Kill the rest by PID.")

    base = root / "dfp_eval"
    arms = sorted(base.glob(f"dfp{bits}*")) if base.is_dir() else []
    if not arms:
        print(f"  no results yet under {base}")
        return

    for arm in arms:
        st = arm_status(arm)
        line = f"  {arm_label(st['name'], bits)} {st['done']:2d} / {TOTAL_FOLDS}"
        if st["rate"]:
            line += f"   {st['rate']:.0f}s/fold"
            if st["done"] < TOTAL_FOLDS:
                remaining = (TOTAL_FOLDS - st["done"]) * st["rate"] / 60
                line += f"   ~{remaining:.0f} min left"
        if st["done"] >= TOTAL_FOLDS:
            line += "   COMPLETE"
        elif st["last_write"] and time.time() - st["last_write"] > 3 * SECONDS_PER_FOLD_HINT:
            # Nothing written for several folds' worth of time. Either the run
            # stopped, or it is on a fold far larger than the rest.
            idle = (time.time() - st["last_write"]) / 60
            line += f"   STALLED? nothing written for {idle:.0f} min"
        print(line)
        for name, why in st["strays"][:5]:
            print(f"      CORRUPT: {name} -- {why}")
        if len(st["strays"]) > 5:
            print(f"      ... and {len(st['strays']) - 5} more")

    if everything_done(base, bits, busy=bool(procs)):
        print("\n  Every arm on disk is complete. The cohort numbers:")
    elif any(arm_status(a)["done"] >= TOTAL_FOLDS for a in arms):
        print("\n  Some arms are done while something is still running. What has "
              "finished can\n  already be read:")
    else:
        return
    print(f"    python scripts/summarise_dfp_eval.py {root} --bits {bits}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--watch", action="store_true",
                    help="refresh until every arm is complete")
    ap.add_argument("--every", type=int, default=300, help="seconds between refreshes")
    args = ap.parse_args()

    root = Path(args.artifacts_dir)
    while True:
        render(root, args.bits)
        if not args.watch:
            return 0
        if everything_done(root / "dfp_eval", args.bits,
                           busy=bool(running_processes())):
            return 0
        time.sleep(args.every)


if __name__ == "__main__":
    raise SystemExit(main())
