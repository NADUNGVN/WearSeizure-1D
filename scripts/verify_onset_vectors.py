"""Replay the onset vectors and check every expected value in them.

    python scripts/verify_onset_vectors.py
    python scripts/verify_onset_vectors.py --vectors <dir> --model-dir <dir>

This is the acceptance test the RTL has to pass, written against the same
files the RTL is given. It needs **no CHB-MIT corpus, no checkpoint, no GPU and
no config** -- only `hardware/golden_model.py`, the exported weights, and the
vector directory. That is deliberate: a check that runs only on the training
server cannot be run by whoever is bringing up the hardware.

It reads `stream_input.hex` as a testbench would, slices it into windows at the
offsets `windows.csv` declares, and pushes each one through the DFP8 golden
model. Then it checks, row by row:

* both int8 logits and their margin, exactly -- no tolerance, these are integers;
* the probability derived from the margin;
* the EMA, re-run from the warm start the contract carries;
* which side of each threshold the decision score falls on.

A failure in the first group means the datapath disagrees. A failure in the
third with the first intact means the postprocessor was seeded wrong, which is
the mistake these vectors exist to catch early.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent

# `windows.csv` stores the float columns to eight decimals, so a value read back
# can differ from a recomputation by half an ULP of that -- 5e-9. Anything
# tighter fails on the file format rather than on the arithmetic. The integer
# columns are compared exactly, which is where bit-exactness actually lives.
FLOAT_TOL = 1e-8


def load_golden(model_dir: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "golden_model", REPO / "hardware" / "golden_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["golden_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_stream(path: Path, bits: int) -> np.ndarray:
    """Hex bytes to signed integers, the way a testbench reads $readmemh."""
    text = path.read_text(encoding="utf-8").split()
    raw = np.array([int(v, 16) for v in text], dtype=np.int64)
    half = 1 << (bits - 1)
    return np.where(raw >= half, raw - (1 << bits), raw)


def verify(vectors: Path, model_dir: Path, limit: int | None) -> int:
    contract = json.loads((vectors / "contract.json").read_text(encoding="utf-8"))
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))

    if manifest["provenance"]["fold_id"] != contract["provenance"]["fold_id"]:
        print(f"FAIL: the vectors were made from fold "
              f"{contract['provenance']['fold_id']} but {model_dir} holds weights "
              f"from {manifest['provenance']['fold_id']}. Comparing them would "
              "compare two different models.")
        return 1

    # "DFP8" / "DFP16" -- the data width is in the precision field, not in the
    # layers. Defaulting to 8 would quietly read a 16-bit stream as bytes.
    digits = "".join(ch for ch in str(manifest["precision"]) if ch.isdigit())
    if not digits:
        print(f"FAIL: cannot read a data width from precision "
              f"{manifest['precision']!r}")
        return 1
    bits = int(digits)
    p_out = int(contract["scoring"]["logit_exponent"])
    window_samples = int(contract["windows"]["window_samples"])
    pp = contract["postprocess"]
    alpha = float(pp.get("ema_alpha") or 0.0)
    smooth = pp["decision_score_column"] == "ema_probability"

    golden_mod = load_golden(model_dir)
    specs = [golden_mod.LayerSpec.from_manifest(d)
             for d in golden_mod.all_layers(manifest)]
    weights = golden_mod.load_weights(model_dir / "weights", specs, bits=bits)
    golden = golden_mod.GoldenModel(manifest, weights, bits=bits, check_overflow=True)

    stream = read_stream(vectors / "stream_input.hex", bits)
    if len(stream) != contract["excerpt"]["samples"]:
        print(f"FAIL: stream_input.hex holds {len(stream)} samples, the contract "
              f"says {contract['excerpt']['samples']}")
        return 1

    with (vectors / "windows.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    if not rows:
        print("FAIL: windows.csv has no rows")
        return 1

    # The EMA is seeded from the contract, not from the first row. Seeding it
    # from the first row's own score is the obvious shortcut and it is wrong:
    # the recursion ran over the whole recording before this excerpt began.
    ema = float(pp["ema_warm_start_probability"])
    failures: list[str] = []
    checked = {"logits": 0, "probability": 0, "ema": 0, "threshold": 0}

    for row in rows:
        idx = row["window_idx"]
        off = int(row["stream_offset_samples"])
        x = stream[off:off + window_samples]
        if len(x) != window_samples:
            failures.append(f"window {idx}: the stream ends {window_samples - len(x)} "
                            "samples short of this window")
            continue

        out, _ = golden.run(x)
        l0, l1 = int(out[0, 0]), int(out[1, 0])
        margin = l1 - l0
        checked["logits"] += 1
        bad = None
        if (l0, l1) != (int(row["logit0_int"]), int(row["logit1_int"])):
            bad = (f"window {idx}: logits ({l0}, {l1}) but windows.csv says "
                   f"({row['logit0_int']}, {row['logit1_int']})")
        elif margin != int(row["margin_int"]):
            bad = f"window {idx}: margin {margin} vs {row['margin_int']}"
        if bad is not None:
            failures.append(bad)
            # Carry the EMA forward on the FILE's value rather than the one just
            # computed. Otherwise a single wrong window poisons the recursion and
            # every row after it is reported as an EMA failure, burying the one
            # window that actually broke under a hundred that did not.
            ema = float(row["ema_probability"])
            continue

        prob = 1.0 / (1.0 + np.exp(-margin / (2.0 ** p_out)))
        checked["probability"] += 1
        if abs(prob - float(row["probability"])) > FLOAT_TOL:
            failures.append(f"window {idx}: probability {prob:.10f} vs "
                            f"{row['probability']}")

        ema = alpha * prob + (1.0 - alpha) * ema if alpha else prob
        checked["ema"] += 1
        if abs(ema - float(row["ema_probability"])) > FLOAT_TOL:
            failures.append(
                f"window {idx}: EMA {ema:.10f} vs {row['ema_probability']}. If this "
                "is the first failing row, the warm start is the suspect; if the "
                "error grows row by row, alpha is.")

        score = ema if smooth else prob
        checked["threshold"] += 1
        for name, key in (("threshold_on", "above_on"), ("threshold_off", "above_off")):
            if int(score >= float(pp[name])) != int(row[key]):
                failures.append(f"window {idx}: {key} disagrees at score {score:.6f}")

        # The integer form of the same comparison, which is what the hardware
        # runs. Only meaningful on the unsmoothed score -- an EMA of
        # probabilities has no integer margin.
        if not smooth and (margin >= pp["threshold_on_margin_int"]) != bool(int(row["above_on"])):
            failures.append(
                f"window {idx}: the integer threshold {pp['threshold_on_margin_int']} "
                "disagrees with the float one. The rounding in margin_for_probability "
                "is wrong, or p_out is.")

    print(f"vectors : {vectors}")
    print(f"model   : {model_dir}  (fold {contract['provenance']['fold_id']}, "
          f"seed {contract['provenance']['seed']}, tag {contract['provenance']['run_tag']!r})")
    print(f"stream  : {len(stream)} samples, {contract['excerpt']['duration_s']:.1f} s")
    print(f"windows : {len(rows)} checked, {contract['windows']['ictal_count']} ictal "
          "in the full set")
    print(f"peak accumulator over the replay: {golden.peak_acc} "
          f"({golden.peak_acc.bit_length()} bits of magnitude)")

    if failures:
        print(f"\nFAIL: {len(failures)} disagreement(s); first few:")
        for line in failures[:10]:
            print(f"  {line}")
        return 1

    print(f"\nPASS: {checked['logits']} windows reproduce their logits exactly, "
          f"and the probability,\n  EMA and threshold columns all follow from them. "
          "The stream, the weights and\n  the expected outputs are mutually "
          "consistent, so a testbench that fails\n  against these files has an RTL "
          "problem rather than a data problem.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", type=Path, default=None,
                    help="an excerpt directory; by default every one found under "
                         "hardware/export/onset_vectors")
    ap.add_argument("--model-dir", type=Path, default=REPO / "hardware" / "export")
    ap.add_argument("--limit", type=int, default=None,
                    help="check only the first N windows (a quick smoke test)")
    args = ap.parse_args()

    if args.vectors:
        targets = [args.vectors]
    else:
        # Any directory holding excerpts, found by the contract rather than by
        # the name. Matching `onset_vectors*` was the first fix and was still
        # too narrow: `whole_recording/` sat in the repository unverified while
        # the scan reported nine green excerpts. A name pattern can only find
        # the names someone thought of, so this looks for the file that makes a
        # directory an excerpt. `test_vectors/` and `weights/` keep their files
        # flat, so neither matches.
        targets = sorted(p.parent for p in
                         args.model_dir.glob("*/*/contract.json"))
        if not targets:
            print(f"no excerpt directories under {args.model_dir}. Pull them from the "
                  "server, or run\nscripts/export_onset_vectors.py there to make them.")
            return 1

    worst = 0
    for i, target in enumerate(targets):
        if i:
            print()
        worst = max(worst, verify(target, args.model_dir, args.limit))
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
