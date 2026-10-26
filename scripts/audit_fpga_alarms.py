"""Audit recorded FPGA features and replay frozen alarms, without using hardware.

Python standard library only. No imports from the deployed application, threshold
search, driver execution, or bitstream programming. A report is recorded evidence,
not attestation of a currently loaded bitstream or a new FPGA run.

    python audit_fpga_alarms.py --embedded-dir /path/to/Embedded_Software \
        --output /path/to/new_alarm_audit.json

For whole-recording event metrics, also supply the original CHB-MIT summary with
--summary. Excerpts retain their reference EMA/initial-state assumptions and never
produce a cohort sensitivity or FAR. Existing output files are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

FLOAT_TOL = 1e-8  # Exported float columns have eight decimal places.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(root, name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe asset path")
    result = (root / path).resolve()
    require(root == result or root in result.parents, "Asset escapes its directory")
    return result


def read_hex(path, count, bits=8):
    tokens = [t for line in path.read_text().splitlines()
              for t in line.split("#", 1)[0].split()]
    values = [int(t, 16) for t in tokens]
    require(len(values) == count and all(0 <= v < 1 << bits for v in values),
            f"Invalid hex length/range: {path.name}")
    return [v - (1 << bits) if v >= 1 << (bits - 1) else v for v in values]


def int8(values, count):
    require(isinstance(values, list) and len(values) == count
            and all(type(v) is int and -128 <= v <= 127 for v in values),
            f"Expected {count} signed INT8 values")
    return values


def feature_sha(values):
    return hashlib.sha256(bytes(v & 255 for v in int8(values, 2048))).hexdigest()


def quantize(value, shift, relu=False):
    # Same ties-away rounding and saturation as board_viewer.classify_features.
    require(type(shift) is int and 0 <= shift <= 48, "Invalid output shift")
    if shift:
        magnitude = (abs(value) + (1 << (shift - 1))) >> shift
        value = -magnitude if value < 0 else magnitude
    return max(0 if relu else -128, min(127, value))


def classify(features, layers, weights, biases):
    int8(features, 2048)
    gap, fc = layers["gap"], layers["fc"]
    pooled = [quantize(sum(features[c * 32:(c + 1) * 32]),
                       gap["output_shift"], gap.get("relu", False))
              for c in range(64)]
    return [quantize(biases[k] + sum(pooled[c] * weights[k * 64 + c]
                                   for c in range(64)),
                     fc["output_shift"], fc.get("relu", False)) for k in range(2)]


def replay_alarms(times, scores, params):
    """Standard-library mirror of postprocess.hysteresis.hysteresis_alarms.

    Parity-tested against the production implementation. Scores are already EMA
    probabilities: smoothing margins or smoothing twice is not equivalent.
    """
    on, off = params["threshold_on"], params["threshold_off"]
    run, gap = params["run_length"], params.get("event_merge_gap_s", 0)
    require(0 <= off < on <= 1 and type(run) is int and run >= 1
            and math.isfinite(gap) and gap >= 0, "Invalid frozen hysteresis")
    active, count, start = False, 0, None
    alarms = []
    for time, score in zip(times, scores):
        if not active:
            count = count + 1 if score >= on else 0
            if count >= run:
                active, start = True, time
        elif score < off:
            alarms.append((start, time))
            active, count, start = False, 0, None
    right_censored = active
    if active:
        alarms.append((start, times[-1]))
    merged = []
    for start, end in alarms:
        if merged and start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged, right_censored


def annotations(path, recording):
    blocks = path.read_text(encoding="utf-8-sig").split("File Name:")[1:]
    blocks = [b for b in blocks if b.strip().splitlines()[0].strip() == recording + ".edf"]
    require(len(blocks) == 1, "Summary must contain exactly one matching recording")
    block = blocks[0]
    starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*([\d.]+)\s*seconds", block)
    ends = re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*([\d.]+)\s*seconds", block)
    count = re.search(r"Number of Seizures in File:\s*(\d+)", block)
    require(count is not None and len(starts) == len(ends) == int(count[1]),
            "Incomplete seizure annotations")
    events = [(float(a), float(b)) for a, b in zip(starts, ends)]
    require(all(math.isfinite(a) and math.isfinite(b) and 0 <= a < b for a, b in events),
            "Invalid seizure interval")
    clocks = []
    for label in ("Start", "End"):
        match = re.search(rf"File {label} Time:\s*(\d+):(\d+):(\d+)", block)
        require(match is not None, "Summary lacks recording duration")
        h, m, s = map(int, match.groups())
        require(0 <= m < 60 and 0 <= s < 60, "Invalid recording clock")
        clocks.append(h * 3600 + m * 60 + s)
    duration = (clocks[1] - clocks[0]) % 86400
    require(duration > 0 and all(b <= duration for _, b in events),
            "Invalid recording duration")
    return sorted(events), duration


def event_metrics(events, alarms, duration):
    # Same one-to-one, first-unused overlapping alarm rule as event_matching.py.
    used, delays, missed = set(), [], 0
    for onset, offset in sorted(events):
        found = next((i for i, (a, b) in enumerate(alarms)
                      if i not in used and onset < b and a < offset), None)
        if found is None:
            missed += 1
        else:
            used.add(found)
            delays.append(max(0, alarms[found][0] - onset))
    return {"n_events": len(events), "n_detected": len(delays), "n_missed": missed,
                "event_sensitivity": len(delays) / len(events) if events else None,
                "n_false_alarms": len(alarms) - len(used),
                "far_per_hour": (len(alarms) - len(used)) / (duration / 3600),
                "delays_s": delays, "exposure_hours": duration / 3600}


def audit(embedded, report_path=None, summary_path=None):
    embedded = embedded.resolve()
    report_path = report_path or embedded / "board_verification.json"
    report = load_json(report_path)
    require(isinstance(report, dict) and report.get("schema_version") == 2,
            "Need schema_version=2 recorded evidence")
    require(all(report.get(k) is True for k in ("hardware_tested", "passed"))
            and report.get("mode") == "fpga", "Not a passing FPGA report")
    assets = report.get("asset_sha256", {})
    required = {"manifest.json", "verify_board_model.py", "board_smoke.c",
                "board_viewer.py", "board_smoke_core", "instructions.hex",
                "weights/15_fc_weights.txt", "weights/15_fc_bias.txt"}
    require(required <= assets.keys(), "Report lacks required asset fingerprints")
    for name, digest in assets.items():
        require(sha(child(embedded, name)) == digest, f"Asset changed since report: {name}")
    dataset = report["dataset"]["name"]
    require(isinstance(dataset, str) and Path(dataset).name == dataset
            and dataset not in ("", ".", ".."), "Invalid dataset name")
    source = child(embedded / "onset_vectors", dataset)
    provenance = load_json(source / "import_verification.json")
    require(provenance == report["source_provenance"], "Source provenance changed")
    pins = provenance.get("sha256", provenance.get("source_sha256", {}))
    require({"contract.json", "windows.csv", "stream_input.hex"} <= pins.keys(),
            "Source lacks required pinned hashes")
    for name, digest in pins.items():
        require(sha(child(source, name)) == digest, f"Pinned source changed: {name}")
    contract = load_json(source / "contract.json")
    manifest = load_json(embedded / "manifest.json")
    require(manifest["precision"] == "DFP8", "Only the deployed DFP8 interface is supported")
    fold = contract["provenance"]["fold_id"]
    require(manifest["provenance"]["fold_id"] == report["dataset"]["fold_id"] == fold,
            "Model/report/vector fold mismatch")
    require(report["dataset"]["recording"] == contract["recording"], "Recording mismatch")
    with (source / "windows.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    items = report["windows"]
    n = contract["windows"]["count"]
    require(type(n) is int and n > 0
            and n == len(rows) == len(items) == report["expected_window_count"]
            == report["completed_window_count"], "Incomplete window coverage")
    pp = contract["postprocess"]
    require(pp["method"] == "hysteresis_runlength" and pp["alarm_timestamp"] == "window_end"
            and pp["decision_score_column"] == "ema_probability",
            "Expected frozen causal EMA+hysteresis+run-length")
    alpha, ema = float(pp["ema_alpha"]), float(pp["ema_warm_start_probability"])
    require(math.isfinite(alpha) and 0 < alpha <= 1 and math.isfinite(ema)
            and 0 <= ema <= 1, "Invalid EMA state")
    layers = {layer["name"]: layer for layer in manifest["software_layers"]}
    exponent = contract["scoring"]["logit_exponent"]
    require(type(exponent) is int and 0 <= exponent <= 32
            and layers["fc"]["p_out"] == exponent, "Logit scale mismatch")
    weights = read_hex(embedded / "weights/15_fc_weights.txt", 128)
    biases = read_hex(embedded / "weights/15_fc_bias.txt", 2, 32)
    stream = read_hex(source / "stream_input.hex", contract["excerpt"]["samples"])
    window, stride = contract["windows"]["window_samples"], contract["windows"]["stride_samples"]
    fs = contract["stream"]["sampling_rate_hz"]
    require(window == 1024 and stride > 0 and fs > 0, "Invalid window geometry")
    require(n == (len(stream) - window) // stride + 1, "Stream/window coverage mismatch")
    times, scores, confusion = [], [], {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
    begin = contract["excerpt"]["start_sample"] / fs
    for i, (row, item) in enumerate(zip(rows, items)):
        require(item["index"] == i and item.get("passed") is True
                and item.get("driver_returncode") == 0, f"Failed driver window {i}")
        payload = item["driver_payload"]
        require(isinstance(payload, dict), f"Invalid driver payload at window {i}")
        require(all(payload.get(k) is True for k in ("success", "ran_hardware", "output_match")),
                f"Window {i} did not confirm real FPGA readback")
        actual = int8(payload["output"], 2048)
        directory = source / "windows" / f"{i:03d}"
        golden_path = directory / "13_context_1_pw.txt"
        golden = read_hex(golden_path, 2048)
        digest = feature_sha(actual)
        require(actual == golden and digest == item["actual_int8_sha256"]
                == item["golden_int8_sha256"] and item["golden_sha256"] == sha(golden_path),
                f"Feature evidence mismatch at window {i}")
        require(item.get("feature_bytes_matched") == 2048, f"Bad byte count at window {i}")
        offset = int(row["stream_offset_samples"])
        require(offset == i * stride
                and read_hex(directory / "00_input.txt", window) == stream[offset:offset + window],
                f"Input continuity mismatch at window {i}")
        start, end = float(row["start_sec"]), float(row["end_sec"])
        require(abs(start - (begin + offset / fs)) <= FLOAT_TOL
                and abs(end - start - window / fs) <= FLOAT_TOL
                and abs(item["start_sec"] - start) <= FLOAT_TOL
                and abs(item["end_sec"] - end) <= FLOAT_TOL,
                f"Timestamp mismatch at window {i}")
        logits = classify(actual, layers, weights, biases)
        expected = [int(row["logit0_int"]), int(row["logit1_int"])]
        require(logits == expected == item["logits"] == item["reference_logits"]
                and item.get("logits_match") is True, f"Logit mismatch at window {i}")
        probability = 1 / (1 + math.exp(-(logits[1] - logits[0]) / (2 ** exponent)))
        ema = alpha * probability + (1 - alpha) * ema
        for key, value in (("probability", probability), ("ema_probability", ema),
                           ("decision_score", ema)):
            require(abs(float(row[key]) - value) <= FLOAT_TOL,
                    f"{key} mismatch at window {i}")
        require(int(row["above_on"]) == int(ema >= pp["threshold_on"])
                and int(row["above_off"]) == int(ema >= pp["threshold_off"]),
                f"Frozen threshold mismatch at window {i}")
        label = int(row["label"])
        require(label in (0, 1) and item["ground_truth"] == label, "Ground truth mismatch")
        pred = int(logits[1] > logits[0])
        confusion[["TN", "FP", "FN", "TP"][label * 2 + pred]] += 1
        times.append(end)
        scores.append(ema)
    require(report["feature_bytes_matched"] == n * 2048, "Summary byte count mismatch")
    require(report["confusion"] == confusion, "Summary confusion mismatch")
    accuracy = (confusion["TN"] + confusion["TP"]) / n
    require(abs(report["accuracy"] - accuracy) <= FLOAT_TOL, "Summary accuracy mismatch")
    # A mid-recording excerpt cannot reconstruct unseen FPGA prehistory. Require
    # an off-state/resetting first score, and expose the reference EMA seed.
    if begin > 0:
        require(scores[0] < pp["threshold_on"]
                and not any(a["start_sec"] <= begin < a["end_sec"]
                            for a in contract["alarms_whole_recording"]),
                "Excerpt begins with ambiguous hysteresis state; use whole recording")
    alarms, censored = replay_alarms(times, scores, pp)
    reference = [(max(a["start_sec"], times[0]), min(a["end_sec"], times[-1]))
                 for a in contract["alarms_whole_recording"]
                 if a["start_sec"] <= times[-1] and times[0] <= a["end_sec"]]
    require(alarms == reference, "Alarm intervals differ from frozen reference")
    onset, offset = contract["onset_sec"], contract["offset_sec"]
    require(math.isfinite(onset) and math.isfinite(offset) and 0 <= onset < offset,
            "Invalid anchor event")
    hits = [(a, b) for a, b in alarms if onset < b and a < offset]
    result = {
        "passed": True, "mode": "audit_recorded_fpga_evidence",
        "hardware_run_performed_by_this_tool": False,
        "board_report_sha256": sha(report_path), "checked_asset_files": len(assets),
        "recording": contract["recording"], "fold_id": fold, "windows": n,
        "feature_bytes_matched": n * 2048, "window_accuracy": accuracy, "confusion": confusion,
        "frozen_postprocess": {k: pp[k] for k in ("ema_alpha", "threshold_on", "threshold_off",
                                              "run_length", "event_merge_gap_s", "alarm_timestamp")},
        "alarms": [{"start_sec": a, "end_sec": b} for a, b in alarms],
        "final_alarm_right_censored": censored,
        "anchor_event": {"onset_sec": onset, "offset_sec": offset, "detected": bool(hits),
                          "delay_s": max(0, hits[0][0] - onset) if hits else None,
                          "fully_observed": begin <= onset and times[-1] >= offset},
        "initial_state": "off; reference EMA warm start" if begin else "recording start",
        "scope": "excerpt_conditional_on_reference_state" if begin else "annotation_coverage_unverified",
        "event_metrics": None,
        "limitations": ["Recorded evidence, not a new FPGA run or loaded-bitstream attestation.",
                     "One recording/fold is not the cohort result."],
    }
    if summary_path:
        events, duration = annotations(summary_path, contract["recording"])
        for row in rows:
            label = int(any(a < float(row["end_sec"]) and float(row["start_sec"]) < b
                            for a, b in events))
            require(label == int(row["label"]), "Summary and CSV labels disagree")
        full = (begin == 0 and abs(len(stream) / fs - duration) <= FLOAT_TOL
                and 0 <= duration - times[-1] < stride / fs + FLOAT_TOL)
        result["annotation_sha256"] = sha(summary_path)
        if full:
            result["scope"] = "single_complete_recording"
            result["event_metrics"] = event_metrics(events, alarms, duration)
        else:
            result["limitations"].append("Incomplete recording; no aggregate event sensitivity/FAR.")
    else:
        result["limitations"].append("No complete annotations supplied; no event sensitivity/FAR.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedded-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    # Reserve exclusively before reading evidence; never overwrite an old result.
    try:
        output = args.output.open("x", encoding="utf-8")
    except OSError as exc:
        print(f"FAIL: cannot create new output: {exc}", file=sys.stderr)
        return 1
    try:
        result = audit(args.embedded_dir, args.report, args.summary)
    except (OSError, ValueError, KeyError, TypeError, OverflowError, IndexError,
            AttributeError) as exc:
        result = {"passed": False, "error": str(exc), "hardware_run_performed_by_this_tool": False}
    with output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    if not result["passed"]:
        print(f"FAIL: {result['error']}; report: {args.output}", file=sys.stderr)
        return 1
    anchor = result["anchor_event"]
    print(f"PASS: {result['windows']} recorded FPGA windows; assets, features, logits and EMA checked")
    print(f"Window accuracy: {100 * result['window_accuracy']:.2f}% (not event sensitivity)")
    print(f"Anchor event: detected={anchor['detected']}, delay={anchor['delay_s']} s")
    print(f"Scope: {result['scope']}; full-recording metrics: {result['event_metrics']}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
