"""The table the hardware result has to reproduce, built without the hardware.

    python scripts/hardware_verification_table.py

For every complete recording exported under `hardware/export*/whole_recording/`
(found by its contract, not its name), this checks two things and writes them
as one table:

1. **Bit-exactness of the reference.** The recording's `stream_input.hex` is
   replayed through `hardware/golden_model.py` with that fold's DFP8 weights,
   and every window's integer logits, probability, EMA and threshold flags are
   checked against `windows.csv` (`verify_onset_vectors.verify`). This is the
   comparison the board must also pass, against the same files.
2. **End-to-end detection on the deployed pipeline.** The checked EMA scores go
   through the frozen hysteresis / run-length / merge parameters of the fold,
   the alarms are matched to the seizures with the evaluation's own matcher,
   and the result is reported the way the device is used: seizures detected,
   delay, false alarms, hours. The alarms are also compared with the ones the
   server pipeline froze into the contract when the vectors were exported.

What the table is entitled to say
---------------------------------
These recordings were chosen from folds the evaluation detected, so "all
seizures detected" here is a property of the selection, not the model's
sensitivity. The table establishes what the board must reproduce; the model's
sensitivity is the cohort evaluation's. Window-level accuracy is deliberately
absent: it moves with the excerpt (72.4 % on a 130 s excerpt of chb01_03,
98.4 % on the whole hour, same model and same windows) and is not what the
device decides on.

Needs NumPy and this repository only -- no dataset, no checkpoint, no GPU.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from wearseizure.eval.metrics_event import compute_event_metrics
from wearseizure.postprocess.hysteresis import PostprocessParams, hysteresis_alarms

_spec = importlib.util.spec_from_file_location("verify_onset_vectors", REPO / "scripts" / "verify_onset_vectors.py")
verify_onset_vectors = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_onset_vectors)

BOARD_PENDING = "not yet measured"


def load_reference_model(model_dir: Path):
    """The DFP8 golden model of one fold, as the verifier builds it."""
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    bits = int("".join(ch for ch in str(manifest["precision"]) if ch.isdigit()))
    golden_mod = verify_onset_vectors.load_golden(model_dir)
    specs = [golden_mod.LayerSpec.from_manifest(d) for d in golden_mod.all_layers(manifest)]
    weights = golden_mod.load_weights(model_dir / "weights", specs, bits=bits)
    return golden_mod, golden_mod.GoldenModel(manifest, weights, bits=bits), bits


def host_logits(golden_mod, golden, features):
    """GAP and FC applied to a feature map, as the host does after the PL.

    The same arithmetic as `GoldenModel.run` for the layers from GAP onward, so
    a board feature map that equals the golden one yields the golden logits by
    construction, and one that does not yields what the device would actually
    decide on.
    """
    import numpy as np

    host = next(i for i, s in enumerate(golden.specs) if s.type == "gap")
    act = np.asarray(features, dtype=np.int64)
    for spec in golden.specs[host:]:
        if spec.type == "gap":
            acc = golden_mod.gap_int(act)
        else:
            w, b = golden.weights[spec.name]
            acc = golden_mod.conv_int(act, w, b, spec)
        act = golden_mod.requantise(acc, spec.output_shift, golden.bits, spec.relu)
    return int(act[0, 0]), int(act[1, 0])


def board_column(entry: dict, board_path: Path, model_dir: Path, vectors: Path) -> str:
    """Score one board run of a complete recording against the reference.

    Every window's feature map is compared byte for byte with the golden
    model's; the board's own feature maps are then taken through the host
    head and the frozen postprocessor, so the event columns describe what the
    device itself would have alarmed on.
    """
    import numpy as np

    run = json.loads(board_path.read_text(encoding="utf-8"))
    contract = json.loads((vectors / "contract.json").read_text(encoding="utf-8"))
    if run.get("fold_id") != contract["provenance"]["fold_id"]:
        raise SystemExit(f"{board_path}: board run is fold {run.get('fold_id')}, vectors are "
                         f"{contract['provenance']['fold_id']}")
    for name, digest in run.get("vectors_sha256", {}).items():
        if _sha256(vectors / name) != digest:
            raise SystemExit(f"{board_path}: {name} changed since the board ran it")

    with (vectors / "windows.csv").open(encoding="utf-8") as f:
        rows = {int(r["window_idx"]): r for r in csv.DictReader(f)}
    stream = verify_onset_vectors.read_stream(vectors / "stream_input.hex", 8)
    golden_mod, golden, _ = load_reference_model(model_dir)
    pl_last = next(k for k in golden.specs if k.type == "gap").layer_id - 1

    mismatched, margins = [], {}
    for w in run["windows"]:
        idx = int(w["window_idx"])
        row = rows[idx]
        off = int(row["stream_offset_samples"])
        _, trace = golden.run(stream[off:off + 1024])
        ref = next(v for k, v in trace.items() if k.startswith(f"{pl_last:02d}_"))
        board = np.frombuffer(bytes.fromhex(w["features_hex"]), dtype=np.int8).astype(np.int64)
        board = board.reshape(ref.shape)  # channel-major, as the driver reads it back
        if not np.array_equal(board, ref):
            mismatched.append(idx)
        l0, l1 = host_logits(golden_mod, golden, board)
        margins[idx] = l1 - l0

    n_total = len(rows)
    n_run = len(run["windows"])
    entry["board_windows_run"] = n_run
    entry["board_feature_mismatches"] = mismatched[:50]
    entry["board_n_feature_mismatches"] = len(mismatched)
    exact = f"{n_run - len(mismatched):,}/{n_run:,} bit-exact"
    if not run.get("complete") or n_run != n_total:
        return f"INCOMPLETE {n_run:,}/{n_total:,} windows; {exact}"

    pp = contract["postprocess"]
    p_out = int(contract["scoring"]["logit_exponent"])
    alpha = float(pp["ema_alpha"])
    ema = float(pp["ema_warm_start_probability"])
    end_sec, scores = [], []
    for idx in sorted(rows):
        prob = 1.0 / (1.0 + np.exp(-margins[idx] / (2.0 ** p_out)))
        ema = alpha * prob + (1.0 - alpha) * ema
        end_sec.append(float(rows[idx]["end_sec"]))
        scores.append(ema if pp["decision_score_column"] == "ema_probability" else prob)
    params = PostprocessParams(
        method=pp["method"], ema_alpha=alpha, threshold_on=float(pp["threshold_on"]),
        threshold_off=float(pp["threshold_off"]), run_length=int(pp["run_length"]),
        event_merge_gap_s=float(pp["event_merge_gap_s"]), alarm_timestamp=pp["alarm_timestamp"],
        window_s=float(pp["window_s"]),
    )
    alarms = hysteresis_alarms(end_sec, scores, params)
    events = [(f"{entry['recording']}_sz{i}", on, off) for i, (on, off) in enumerate(entry["seizures"])]
    m = compute_event_metrics(events, alarms, entry["hours"])
    entry["board_alarms"] = [list(a) for a in alarms]
    entry["board_alarms_match_reference"] = [list(a) for a in alarms] == entry["alarms"]
    delays = ", ".join(f"{d:.0f}" for d in m.delays_s) or "--"
    return (f"{exact}; {m.n_matched}/{m.n_events} detected (delay {delays} s), "
            f"{m.n_false_alarms} false alarms")


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def whole_recordings(repo: Path) -> list[tuple[Path, Path]]:
    """(model_dir, vectors_dir) for every exported COMPLETE recording.

    A complete recording is recognised by what its contract says -- it starts
    at sample 0 and carries the alarms over the whole recording -- because a
    name pattern only finds the names someone thought of.
    """
    found = []
    for contract_path in sorted(repo.glob("hardware/export*/*/*/contract.json")):
        c = json.loads(contract_path.read_text(encoding="utf-8"))
        if c.get("excerpt", {}).get("start_sample") == 0 and "alarms_whole_recording" in c:
            found.append((contract_path.parents[2], contract_path.parent))
    return found


def seizures_from_labels(rows: list[dict], window_s: float, stride_s: float) -> list[tuple[float, float]]:
    """Seizure intervals recovered from the window labels.

    A window is labelled ictal iff it overlaps a seizure, strictly
    (`data/windowing._overlaps`). With a 1 s stride and whole-second
    annotations, the first ictal window starts window_s - 1 s before onset and
    the last ends window_s - 1 s after offset, so both are recoverable exactly.
    The caller cross-checks the result against every interval the contracts
    state, and refuses the recording if any disagrees.
    """
    if stride_s != 1.0:
        raise ValueError(f"label inversion assumes a 1 s stride, got {stride_s}")
    runs: list[list[float]] = []
    prev = 0
    for r in rows:
        y = int(r["label"])
        if y and not prev:
            runs.append([float(r["start_sec"]), float(r["end_sec"])])
        if y:
            runs[-1][1] = float(r["end_sec"])
        prev = y
    lead = window_s - stride_s
    return [(start + lead, end - lead) for start, end in runs]


def stated_intervals(repo: Path) -> dict[str, set[tuple[float, float]]]:
    """Every (onset, offset) any exported contract states, by recording."""
    out: dict[str, set[tuple[float, float]]] = {}
    for p in repo.glob("hardware/export*/*/*/contract.json"):
        c = json.loads(p.read_text(encoding="utf-8"))
        out.setdefault(c["recording"], set()).add((float(c["onset_sec"]), float(c["offset_sec"])))
    return out


def evaluate_recording(model_dir: Path, vectors: Path, stated: dict) -> dict:
    contract = json.loads((vectors / "contract.json").read_text(encoding="utf-8"))
    rec = contract["recording"]

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        verify_status = verify_onset_vectors.verify(vectors, model_dir, None)

    with (vectors / "windows.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pp = contract["postprocess"]
    window_s = float(pp["window_s"])
    stride_s = contract["windows"]["stride_samples"] / contract["stream"]["sampling_rate_hz"]
    seizures = seizures_from_labels(rows, window_s, stride_s)
    unconfirmed = sorted(stated.get(rec, set()) - set(seizures))
    if unconfirmed:
        raise SystemExit(f"{rec}: contract states seizure(s) {unconfirmed} that the labels do not "
                         f"reproduce (labels give {seizures}); refusing to score it")

    params = PostprocessParams(
        method=pp["method"], ema_alpha=float(pp["ema_alpha"]),
        threshold_on=float(pp["threshold_on"]), threshold_off=float(pp["threshold_off"]),
        run_length=int(pp["run_length"]), event_merge_gap_s=float(pp["event_merge_gap_s"]),
        alarm_timestamp=pp["alarm_timestamp"], window_s=window_s,
    )
    end_sec = [float(r["end_sec"]) for r in rows]
    scores = [float(r[pp["decision_score_column"]]) for r in rows]
    alarms = hysteresis_alarms(end_sec, scores, params)
    frozen = [(float(a["start_sec"]), float(a["end_sec"])) for a in contract["alarms_whole_recording"]]

    hours = contract["excerpt"]["duration_s"] / 3600.0
    events = [(f"{rec}_sz{i}", on, off) for i, (on, off) in enumerate(seizures)]
    m = compute_event_metrics(events, alarms, hours)
    prov = contract["provenance"]
    return {
        "recording": rec,
        "fold_id": prov["fold_id"],
        "run_tag": prov["run_tag"],
        "seed": prov["seed"],
        "hours": hours,
        "windows": len(rows),
        "reference_bit_exact": verify_status == 0,
        "verify_log": log.getvalue(),
        "seizures": [list(s) for s in seizures],
        "thresholds": [params.threshold_on, params.threshold_off],
        "alarms": [list(a) for a in alarms],
        "alarms_match_frozen_contract": alarms == frozen,
        "n_seizures": m.n_events,
        "n_detected": m.n_matched,
        "delays_s": m.delays_s,
        "n_false_alarms": m.n_false_alarms,
        "board": BOARD_PENDING,
    }


def render(results: list[dict]) -> str:
    def yes(b: bool) -> str:
        return "yes" if b else "**NO**"

    lines = [
        (
            "| Recording | Fold weights | Hours | Windows | Reference bit-exact | Alarms = frozen | "
            "Seizures | Detected | Delay (s) | False alarms | Board (KV260) |"
        ),
        "|---|---|--:|--:|:-:|:-:|--:|--:|---|--:|---|",
    ]
    for r in results:
        delays = ", ".join(f"{d:.0f}" for d in r["delays_s"]) or "--"
        lines.append(
            f"| {r['recording']} | {r['fold_id']} ({r['run_tag']}, seed {r['seed']}) | {r['hours']:.2f} | "
            f"{r['windows']:,} | {yes(r['reference_bit_exact'])} | {yes(r['alarms_match_frozen_contract'])} | "
            f"{r['n_seizures']} | {r['n_detected']} | {delays} | {r['n_false_alarms']} | {r['board']} |"
        )
    all_delays = [d for r in results for d in r["delays_s"]]
    mean_delay = f"mean {sum(all_delays) / len(all_delays):.1f}" if all_delays else "--"
    lines.append(
        f"| **Total** | | **{sum(r['hours'] for r in results):.2f}** | **{sum(r['windows'] for r in results):,}** | "
        f"{yes(all(r['reference_bit_exact'] for r in results))} | "
        f"{yes(all(r['alarms_match_frozen_contract'] for r in results))} | "
        f"**{sum(r['n_seizures'] for r in results)}** | **{sum(r['n_detected'] for r in results)}** | "
        f"{mean_delay} | **{sum(r['n_false_alarms'] for r in results)}** | "
        f"{sum(r['board'] != BOARD_PENDING for r in results)}/{len(results)} recordings run on the board |"
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path, default=REPO / "docs" / "hardware_verification_table.json")
    ap.add_argument("--markdown", type=Path, default=REPO / "docs" / "hardware_verification_table.md")
    ap.add_argument("--board-run", type=Path, action="append", default=[],
                    help="a JSON written by scripts/board_run_recording.py on the KV260; repeatable")
    args = ap.parse_args()

    targets = whole_recordings(REPO)
    if not targets:
        print("no complete recordings under hardware/export*/; pull them from the server first")
        return 1
    stated = stated_intervals(REPO)
    results = []
    board_runs = {}
    for path in args.board_run:
        board_runs[json.loads(path.read_text(encoding="utf-8"))["recording"]] = path
    for model_dir, vectors in targets:
        print(f"replaying {vectors.relative_to(REPO)} ...", flush=True)
        entry = evaluate_recording(model_dir, vectors, stated)
        if entry["recording"] in board_runs:
            print(f"  comparing board run {board_runs[entry['recording']]} ...", flush=True)
            entry["board"] = board_column(entry, board_runs.pop(entry["recording"]), model_dir, vectors)
        results.append(entry)
    if board_runs:
        raise SystemExit(f"board run(s) for recordings with no exported vectors: {sorted(board_runs)}")

    table = render(results)
    print()
    print(table)
    ok = all(r["reference_bit_exact"] and r["alarms_match_frozen_contract"] for r in results)
    ok = ok and all(r.get("board_n_feature_mismatches", 0) == 0 for r in results)

    args.json.write_text(json.dumps({"results": results, "passed": ok}, indent=2), encoding="utf-8")
    header = (
        "# Hardware verification table\n\n"
        "Generated by `python scripts/hardware_verification_table.py`; do not edit by hand.\n"
        "Read the docstring of that script for what this table may and may not be used to claim.\n\n"
    )
    args.markdown.write_text(header + table + "\n", encoding="utf-8")
    print(f"\n{'PASS' if ok else 'FAIL'}; written to {args.markdown} and {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
