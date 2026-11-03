"""What a hardware demonstration on these vectors may and may not claim.

    python scripts/demo_card.py
    python scripts/demo_card.py --model-dir hardware/export_chb15__chb15_40

Reads the published excerpts and prints the numbers a demo is entitled to put
on a slide, with the ones it is not entitled to shown beside them and labelled.

The reason this script exists is that the same vectors support both a 100 %
and a 61 % headline, and the difference is not a matter of emphasis:

* **Event level** -- did an alarm overlap each seizure, and did any alarm fire
  outside one. This is what a seizure detector is for, what the frozen
  post-processing optimises, and what every published comparison in section 10
  of the hardware brief reports. On these excerpts it is 100 %.
* **Window level** -- was each individual 4 s window flagged. The device never
  needs this: the EMA, run-length and hysteresis exist precisely to turn a
  noisy per-window score into one clean event. On the same excerpts it is 61 %.

Neither number is wrong. Quoting the window-level one as the detector's
performance understates it; quoting the event-level one as the MODEL's
performance overstates it, because these excerpts were selected from folds that
scored 1.0 -- the cohort figure is 94.95 % over 66 folds and three seeds, and
that is the number for a paper.

Window-level accuracy also moves with the ictal/interictal mix of whatever
excerpt it is computed on, which this script demonstrates by printing it per
excerpt. A longer interictal tail raises it without the model changing at all.
That is the selection effect the project's own Figure 1 is about, so it is not
available to us either.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Measured over 66 folds x 3 seeds; see docs/PAPER_CONTEXT_BRIEF.md section 4.1.
COHORT_SENS, COHORT_FAR = 0.9495, 0.2500


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="restrict to one exported fold; default is every excerpt found")
    args = ap.parse_args()

    root = str(args.model_dir) if args.model_dir else "hardware"
    # Found by the contract, not by a directory name. A name pattern can only
    # find the names someone thought of, and `whole_recording/` has already sat
    # unscanned once while a scan reported green.
    paths = sorted(glob.glob(os.path.join(root, "**", "*", "contract.json"),
                             recursive=True))
    if not paths:
        print(f"no excerpts under {root}")
        return 1

    seizures, caught, false_alarms = set(), set(), 0
    pre_windows = pre_firing = 0
    tp = fn = fp = tn = 0
    # Window-level sensitivity is close to (duration - delay) / duration, because
    # every window before the alarm is below threshold by construction. Collected
    # here so the card can show that rather than assert it.
    bounds: list[tuple[str, float, float, float]] = []

    print(f"{'excerpt':<30}{'detected':>10}{'delay':>8}{'false':>7}"
          f"{'win sens':>10}{'win spec':>10}{'win acc':>9}")
    for p in paths:
        c = json.loads(Path(p).read_text(encoding="utf-8"))
        d = c["detection"]
        rows = list(csv.DictReader(
            open(os.path.join(os.path.dirname(p), "windows.csv"), encoding="utf-8")))

        key = (c["recording"], c["onset_sec"])
        seizures.add(key)
        if d["detected"]:
            caught.add(key)
        false_alarms += d["false_alarms_in_excerpt"]
        dur = c["offset_sec"] - c["onset_sec"]
        if d["detected"] and dur > 0:
            bounds.append((os.path.basename(os.path.dirname(p)), dur,
                           d["delay_s"], 100.0 * max(0.0, dur - d["delay_s"]) / dur))

        pre = [r for r in rows if float(r["end_rel_onset_s"]) <= 0]
        pre_windows += len(pre)
        pre_firing += sum(1 for r in pre if r["above_on"] == "1")

        a = sum(1 for r in rows if r["label"] == "1" and r["above_on"] == "1")
        b = sum(1 for r in rows if r["label"] == "1" and r["above_on"] == "0")
        e = sum(1 for r in rows if r["label"] == "0" and r["above_on"] == "1")
        f = sum(1 for r in rows if r["label"] == "0" and r["above_on"] == "0")
        tp, fn, fp, tn = tp + a, fn + b, fp + e, tn + f

        tag = os.path.basename(os.path.dirname(p))
        if "_60s" in p:
            tag += " (60s)"
        elif "_release" in p:
            tag += " (rel)"
        print(f"  {tag:<28}{'yes' if d['detected'] else 'NO':>10}"
              f"{(d['delay_s'] or 0):>7.0f}s{d['false_alarms_in_excerpt']:>7}"
              f"{100 * a / (a + b) if a + b else float('nan'):>9.1f}%"
              f"{100 * f / (f + e) if f + e else float('nan'):>9.1f}%"
              f"{100 * (a + f) / len(rows):>8.1f}%")

    n = tp + fn + fp + tn

    # Whole recordings only. Accuracy on a short excerpt is a statement about
    # how the excerpt was cut, so mixing the two would average a property of
    # the model with a property of the scissors.
    whole = [q for q in paths if "whole_recording" in q]
    if whole:
        wtp = wfn = wfp = wtn = 0
        print(f"\n{'=' * 78}")
        print("ACCURACY, AND WHAT IT IS WORTH (whole recordings only)")
        print(f"{'=' * 78}")
        print(f"  {'recording':<16}{'windows':>9}{'ictal':>7}{'model':>9}"
              f"{'never-fires':>13}")
        for q in whole:
            c = json.loads(Path(q).read_text(encoding="utf-8"))
            rows = list(csv.DictReader(
                open(os.path.join(os.path.dirname(q), "windows.csv"), encoding="utf-8")))
            a = sum(1 for r in rows if r["label"] == "1" and int(r["margin_int"]) > 0)
            b = sum(1 for r in rows if r["label"] == "1" and int(r["margin_int"]) <= 0)
            e = sum(1 for r in rows if r["label"] == "0" and int(r["margin_int"]) > 0)
            g = sum(1 for r in rows if r["label"] == "0" and int(r["margin_int"]) <= 0)
            wtp, wfn, wfp, wtn = wtp + a, wfn + b, wfp + e, wtn + g
            ict = a + b
            print(f"  {c['recording']:<16}{len(rows):>9}{ict:>7}"
                  f"{100 * (a + g) / len(rows):>8.2f}%"
                  f"{100 * (len(rows) - ict) / len(rows):>12.2f}%")
        wn = wtp + wfn + wfp + wtn
        acc = 100 * (wtp + wtn) / wn
        base = 100 * (wtn + wfp) / wn
        print(f"  {'POOLED':<16}{wn:>9}{wtp + wfn:>7}{acc:>8.2f}%{base:>12.2f}%")
        print(f"\n  Never report the model's {acc:.2f} % without the {base:.2f} % "
              "beside it.")
        print(f"  A device that never fires scores {base:.2f} % on this data. "
              f"Everything the\n  detector does is worth {acc - base:+.2f} points "
              "here, because the windows it\n  exists to find are a few percent of "
              "the recording. Event sensitivity\n  separates those same two devices "
              "by 94.89 points.")

    print(f"\n{'=' * 78}")
    print("MAY BE CLAIMED FOR THIS DEMONSTRATION")
    print(f"{'=' * 78}")
    print(f"  seizures detected            {len(caught)} of {len(seizures)} distinct "
          f"({100 * len(caught) / max(1, len(seizures)):.0f} %), across "
          f"{len(paths)} excerpt(s)")
    print(f"  false alarms                 {false_alarms}")
    print(f"  silent before every onset    {pre_windows - pre_firing} of {pre_windows} "
          f"windows ({100 * (pre_windows - pre_firing) / max(1, pre_windows):.1f} %)")
    print("\n  And, from the evaluation rather than from these excerpts:")
    print(f"  cohort event sensitivity     {100 * COHORT_SENS:.2f} %, "
          f"FAR {COHORT_FAR:.2f}/h, 66 folds x 3 seeds")

    print(f"\n{'=' * 78}")
    print("MAY NOT BE CLAIMED")
    print(f"{'=' * 78}")
    print(f"  Window-level sensitivity ({100 * tp / (tp + fn):.1f} %) as the detector's")
    print("  performance. The device raises one alarm per seizure; flagging every")
    print("  4 s window is not its job and no compared publication reports it.")
    if bounds:
        print("  It is also close to arithmetic rather than quality: every window")
        print("  before the alarm is below threshold by construction, so the ceiling")
        print("  is (duration - delay) / duration --")
        for tag, dur, delay, ceil in bounds[:4]:
            print(f"    {tag:<22}{dur:>5.0f}s seizure, {delay:>3.0f}s delay "
                  f"-> at most {ceil:.0f} % of its windows can fire")
    print(f"\n  Window-level accuracy ({100 * (tp + tn) / n:.1f} %) as any kind of headline,")
    print(f"  nor window-level specificity ({100 * tn / (tn + fp) if tn + fp else 0:.1f} %).")
    print("  Both move with the ictal/interictal mix of whatever excerpt they are")
    print("  computed on -- compare the per-excerpt columns above, same model")
    print("  throughout. A longer interictal tail raises accuracy toward 100 %")
    print("  while the detector does not change at all.")
    print("\n  100 % as the MODEL's sensitivity. These excerpts were selected from")
    print("  folds the evaluation scored at 1.0, so the figure is a property of the")
    print(f"  selection. The model's number is {100 * COHORT_SENS:.2f} %.")
    print("\n  98.48 % anywhere. It is seed 0 of three, exactly 65 of 66 folds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
