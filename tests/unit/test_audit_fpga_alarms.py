"""Small synthetic fixtures test the audit, never clinical performance."""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit_fpga_alarms", REPO / "scripts/audit_fpga_alarms.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def dump(path, value):
    path.write_text(json.dumps(value))


class AlarmAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "onset_vectors/test_sz0"
        self.source.mkdir(parents=True)
        (self.root / "weights").mkdir()
        self.p = {"method": "hysteresis_runlength", "alarm_timestamp": "window_end",
                      "ema_alpha": 0.125, "ema_warm_start_probability": 1 / (1 + math.exp(-10 / 16)),
                      "decision_score_column": "ema_probability", "threshold_on": 0.6,
                      "threshold_off": 0.1, "run_length": 3, "event_merge_gap_s": 2}
        self.contract = {
            "recording": "test", "provenance": {"fold_id": "test__test"},
            "onset_sec": 5, "offset_sec": 9, "stream": {"sampling_rate_hz": 256},
            "excerpt": {"start_sample": 0, "samples": 2816},
            "windows": {"count": 8, "window_samples": 1024, "stride_samples": 256},
            "scoring": {"logit_exponent": 4}, "postprocess": self.p,
            "alarms_whole_recording": [{"start_sec": 6, "end_sec": 11}],
        }
        dump(self.source / "contract.json", self.contract)
        (self.source / "stream_input.hex").write_text("00\n" * 2816)
        manifest = {
            "precision": "DFP8", "provenance": {"fold_id": "test__test"},
            "software_layers": [{"name": "gap", "output_shift": 5},
                             {"name": "fc", "output_shift": 8, "p_out": 4}],
        }
        dump(self.root / "manifest.json", manifest)
        (self.root / "weights/15_fc_weights.txt").write_text("00\n" * 128)
        (self.root / "weights/15_fc_bias.txt").write_text("FFFFFB00\n00000500\n")
        # These strings stand in for pinned assets; none is executed.
        for name in ("verify_board_model.py", "board_viewer.py", "board_smoke.c",
                     "board_smoke_core", "instructions.hex"):
            (self.root / name).write_text("synthetic test fixture only")
        columns = ["stream_offset_samples", "start_sec", "end_sec", "label", "logit0_int",
                   "logit1_int", "probability", "ema_probability", "decision_score",
                   "above_on", "above_off"]
        rows, items = [], []
        for i in range(8):
            directory = self.source / "windows" / f"{i:03d}"
            directory.mkdir(parents=True)
            (directory / "00_input.txt").write_text("00\n" * 1024)
            golden_path = directory / "13_context_1_pw.txt"
            golden_path.write_text("00\n" * 2048)
            probability = self.p["ema_warm_start_probability"]
            label = int(5 < i + 4 and i < 9)
            rows.append(dict(zip(columns, [
                i * 256, i, i + 4, label, -5, 5,
                probability, probability, probability, 1, 1,
            ])))
            features = [0] * 2048
            items.append({
                "index": i, "passed": True, "driver_returncode": 0,
                "driver_payload": {"success": True, "ran_hardware": True, "output_match": True,
                                    "output": features},
                "actual_int8_sha256": audit.feature_sha(features),
                "golden_int8_sha256": audit.feature_sha(features),
                "golden_sha256": audit.sha(golden_path), "feature_bytes_matched": 2048,
                "start_sec": i, "end_sec": i + 4, "logits": [-5, 5], "reference_logits": [-5, 5],
                "logits_match": True, "ground_truth": label,
            })
        with (self.source / "windows.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, columns)
            writer.writeheader()
            writer.writerows(rows)
        self.provenance = {"sha256": {
            name: audit.sha(self.source / name)
            for name in ("contract.json", "stream_input.hex", "windows.csv")
        }}
        dump(self.source / "import_verification.json", self.provenance)
        self.report = {
            "schema_version": 2, "mode": "fpga", "hardware_tested": True, "passed": True,
            "asset_sha256": {str(p.relative_to(self.root)): audit.sha(p)
                          for p in self.root.iterdir() if p.is_file()},
            "dataset": {"name": "test_sz0", "recording": "test", "fold_id": "test__test"},
            "source_provenance": self.provenance, "windows": items, "expected_window_count": 8,
            "completed_window_count": 8, "feature_bytes_matched": 8 * 2048,
            "confusion": {"TN": 0, "FP": 2, "FN": 0, "TP": 6}, "accuracy": 0.75,
        }
        self.report["asset_sha256"].update({
            "weights/" + p.name: audit.sha(p) for p in (self.root / "weights").iterdir()
        })
        self.save_report()
        self.summary = self.root / "summary.txt"
        self.summary.write_text(
            "File Name: test.edf\nFile Start Time: 00:00:00\nFile End Time: 00:00:11\n"
            "Number of Seizures in File: 1\nSeizure Start Time: 5 seconds\n"
            "Seizure End Time: 9 seconds\n"
        )

    def save_report(self):
        dump(self.root / "board_verification.json", self.report)

    def repin(self, name):
        self.provenance["sha256"][name] = audit.sha(self.source / name)
        dump(self.source / "import_verification.json", self.provenance)
        self.report["source_provenance"] = copy.deepcopy(self.provenance)
        self.save_report()

    def test_success_with_complete_annotations(self):
        result = audit.audit(self.root, summary_path=self.summary)
        self.assertTrue(result["passed"])
        self.assertFalse(result["hardware_run_performed_by_this_tool"])
        self.assertEqual(result["scope"], "single_complete_recording")
        self.assertEqual(result["anchor_event"]["delay_s"], 1)
        self.assertEqual(result["event_metrics"]["n_detected"], 1)
        self.assertEqual(result["event_metrics"]["n_false_alarms"], 0)
        self.assertEqual(result["event_metrics"]["delays_s"], [1])
        self.assertTrue(result["final_alarm_right_censored"])

    def test_no_complete_annotations_no_clinical_metrics(self):
        result = audit.audit(self.root)
        self.assertIsNone(result["event_metrics"])
        self.assertEqual(result["window_accuracy"], 0.75)

    def test_partial_recording_cannot_claim_full_metrics(self):
        self.summary.write_text(self.summary.read_text().replace("00:00:11", "00:00:30"))
        result = audit.audit(self.root, summary_path=self.summary)
        self.assertIsNone(result["event_metrics"])
        self.assertIn("Incomplete recording", " ".join(result["limitations"]))

    def test_replay_start_in_final_window_preserves_zero_length_interval(self):
        params = {"threshold_on": 0.6, "threshold_off": 0.1, "run_length": 1, "event_merge_gap_s": 0}
        alarms, censored = audit.replay_alarms([4, 5], [0, 1], params)
        self.assertEqual(alarms, [(5, 5)])
        self.assertTrue(censored)

    def test_asset_changed_since_report(self):
        (self.root / "board_smoke_core").write_text("different binary")
        with self.assertRaisesRegex(ValueError, "Asset changed"):
            audit.audit(self.root)

    def test_reference_report_rejected(self):
        self.report["mode"] = "reference_only"
        self.save_report()
        with self.assertRaisesRegex(ValueError, "FPGA report"):
            audit.audit(self.root)

    def test_missing_windows_rejected(self):
        self.report["windows"].pop()
        self.save_report()
        with self.assertRaisesRegex(ValueError, "coverage"):
            audit.audit(self.root)

    def test_actual_feature_mismatch_cannot_hide_behind_flags(self):
        self.report["windows"][0]["driver_payload"]["output"][0] = 1
        self.save_report()
        with self.assertRaisesRegex(ValueError, "Feature evidence"):
            audit.audit(self.root)

    def test_short_features_rejected_before_zip(self):
        self.report["windows"][0]["driver_payload"]["output"].pop()
        self.save_report()
        with self.assertRaisesRegex(ValueError, "2048"):
            audit.audit(self.root)

    def test_driver_nonzero_rejected(self):
        self.report["windows"][0]["driver_returncode"] = 1
        self.save_report()
        with self.assertRaisesRegex(ValueError, "Failed driver"):
            audit.audit(self.root)

    def test_summary_confusion_not_trusted(self):
        self.report["confusion"]["TP"] = 8
        self.save_report()
        with self.assertRaisesRegex(ValueError, "confusion"):
            audit.audit(self.root)

    def test_changed_csv_without_repin_rejected(self):
        with (self.source / "windows.csv").open("a") as file:
            file.write("\n")
        with self.assertRaisesRegex(ValueError, "Pinned source changed"):
            audit.audit(self.root)

    def test_wrong_ema_detected_after_repin(self):
        path = self.source / "windows.csv"
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            columns, rows = reader.fieldnames, list(reader)
        rows[0]["ema_probability"] = "0"
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, columns)
            writer.writeheader()
            writer.writerows(rows)
        self.repin("windows.csv")
        with self.assertRaisesRegex(ValueError, "ema_probability"):
            audit.audit(self.root)

    def test_incomplete_annotation_count_rejected(self):
        self.summary.write_text(self.summary.read_text().replace("in File: 1", "in File: 2"))
        with self.assertRaisesRegex(ValueError, "Incomplete seizure"):
            audit.audit(self.root, summary_path=self.summary)

    def test_output_is_exclusive_and_failure_returns_nonzero(self):
        output = self.root / "new_audit.json"
        self.assertEqual(audit.main(["--embedded-dir", str(self.root),
                                     "--output", str(output)]), 0)
        original = output.read_bytes()
        self.assertEqual(audit.main(["--embedded-dir", str(self.root),
                                     "--output", str(output)]), 1)
        self.assertEqual(output.read_bytes(), original)
        self.report["mode"] = "reference_only"
        self.save_report()
        failed = self.root / "failed_audit.json"
        self.assertEqual(audit.main(["--embedded-dir", str(self.root),
                                     "--output", str(failed)]), 1)
        self.assertFalse(json.loads(failed.read_text())["passed"])

    def test_hysteresis_strict_off_and_one_alarm_one_event(self):
        params = {"threshold_on": 0.6, "threshold_off": 0.1, "run_length": 2, "event_merge_gap_s": 0}
        alarms, _ = audit.replay_alarms([1, 2, 3, 4, 5], [0.6, 0.6, 0.1, 0.09, 0], params)
        self.assertEqual(alarms, [(2, 4)])
        metrics = audit.event_metrics([(1, 3), (3, 5)], alarms, 10)
        self.assertEqual(metrics["n_detected"], 1)
        self.assertEqual(metrics["n_missed"], 1)

    def test_randomized_parity_with_production_hysteresis(self):
        import numpy as np
        path = REPO / "src/wearseizure/postprocess/hysteresis.py"
        spec = importlib.util.spec_from_file_location("audit_test_hysteresis", path)
        production = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = production
        spec.loader.exec_module(production)
        rng = random.Random(31)
        for _ in range(50):
            params = {"threshold_on": 0.6, "threshold_off": 0.1,
                          "run_length": rng.randint(1, 4), "event_merge_gap_s": rng.randint(0, 3)}
            scores = [rng.choice([0, 0.1, 0.3, 0.6, 0.9]) for _ in range(100)]
            times = list(range(4, 104))
            expected = production.hysteresis_alarms(
                np.array(times), np.array(scores),
                production.PostprocessParams(method="hysteresis_runlength", **params))
            self.assertEqual(audit.replay_alarms(times, scores, params)[0], expected)

    def test_real_reference_csv_replays_frozen_alarm_without_claiming_fpga(self):
        source = REPO / "hardware/export_chb15__chb15_40/onset_vectors_60s/chb15_40_sz0"
        contract = json.loads((source / "contract.json").read_text())
        with (source / "windows.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        pp = contract["postprocess"]
        ema = pp["ema_warm_start_probability"]
        scores = []
        for row in rows:
            margin = int(row["logit1_int"]) - int(row["logit0_int"])
            probability = 1 / (1 + math.exp(-margin / 2 ** contract["scoring"]["logit_exponent"]))
            ema = pp["ema_alpha"] * probability + (1 - pp["ema_alpha"]) * ema
            self.assertLessEqual(abs(ema - float(row["ema_probability"])), audit.FLOAT_TOL)
            scores.append(ema)
        alarms, censored = audit.replay_alarms([float(r["end_sec"]) for r in rows], scores, pp)
        expected_start = contract["detection"]["alarm_start_sec"]
        self.assertEqual(alarms, [(expected_start, float(rows[-1]["end_sec"]))])
        self.assertEqual(expected_start - contract["onset_sec"], contract["detection"]["delay_s"])
        self.assertTrue(censored)

    def test_real_vectors_through_full_audit_with_simulated_readback_only(self):
        """Golden-generated readback is a test fixture, NOT FPGA evidence."""
        import numpy as np
        model_dir = REPO / "hardware/export_chb15__chb15_40"
        source = model_dir / "onset_vectors_60s/chb15_40_sz0"
        manifest = json.loads((model_dir / "manifest.json").read_text())
        spec = importlib.util.spec_from_file_location(
            "alarm_audit_test_golden", REPO / "hardware/golden_model.py")
        golden_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = golden_module
        spec.loader.exec_module(golden_module)
        specs = [golden_module.LayerSpec.from_manifest(layer)
                 for layer in golden_module.all_layers(manifest)]
        golden = golden_module.GoldenModel(
            manifest, golden_module.load_weights(model_dir / "weights", specs, bits=8),
            bits=8, check_overflow=True)
        for name in ("contract.json", "windows.csv", "stream_input.hex"):
            shutil.copyfile(source / name, self.source / name)
        contract = json.loads((source / "contract.json").read_text())
        manifest["software_layers"] = list(golden_module.all_layers(manifest))[-2:]
        dump(self.root / "manifest.json", manifest)
        for name in ("15_fc_weights.txt", "15_fc_bias.txt"):
            shutil.copyfile(model_dir / "weights" / name, self.root / "weights" / name)
        stream = audit.read_hex(source / "stream_input.hex", contract["excerpt"]["samples"])
        with (source / "windows.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        items = []
        confusion = {"TN": 0, "FP": 0, "FN": 0, "TP": 0}
        for i, row in enumerate(rows):
            start = int(row["stream_offset_samples"])
            vector = stream[start:start + 1024]
            logits, trace = golden.run(np.array(vector, dtype=np.int64))
            features = trace["13_context.1.pw"].reshape(-1).tolist()
            logits = logits.reshape(-1).tolist()
            directory = self.source / "windows" / f"{i:03d}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "00_input.txt").write_text("".join(f"{v & 255:02X}\n" for v in vector))
            path = directory / "13_context_1_pw.txt"
            path.write_text("".join(f"{v & 255:02X}\n" for v in features))
            item = copy.deepcopy(self.report["windows"][0])
            item.update(index=i, start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]), ground_truth=int(row["label"]),
                        logits=logits, reference_logits=logits,
                        actual_int8_sha256=audit.feature_sha(features),
                        golden_int8_sha256=audit.feature_sha(features),
                        golden_sha256=audit.sha(path))
            item["driver_payload"]["output"] = features
            items.append(item)
            key = ["TN", "FP", "FN", "TP"][int(row["label"]) * 2 + int(logits[1] > logits[0])]
            confusion[key] += 1
        self.report.update(
            dataset={"name": "test_sz0", "recording": contract["recording"],
                         "fold_id": contract["provenance"]["fold_id"]},
            windows=items, expected_window_count=len(rows), completed_window_count=len(rows),
            feature_bytes_matched=len(rows) * 2048, confusion=confusion,
            accuracy=(confusion["TN"] + confusion["TP"]) / len(rows))
        for name in list(self.report["asset_sha256"]):
            self.report["asset_sha256"][name] = audit.sha(self.root / name)
        self.provenance["sha256"] = {name: audit.sha(self.source / name)
                                    for name in self.provenance["sha256"]}
        dump(self.source / "import_verification.json", self.provenance)
        self.report["source_provenance"] = self.provenance
        self.save_report()
        result = audit.audit(self.root)
        self.assertEqual(result["anchor_event"]["delay_s"], 24)
        self.assertEqual(result["windows"], 57)
        self.assertEqual(result["confusion"], {"TN": 17, "FP": 0, "FN": 13, "TP": 27})
        self.assertEqual(result["scope"], "excerpt_conditional_on_reference_state")
        self.assertIsNone(result["event_metrics"])
        self.assertTrue(result["final_alarm_right_censored"])


if __name__ == "__main__":
    unittest.main()
