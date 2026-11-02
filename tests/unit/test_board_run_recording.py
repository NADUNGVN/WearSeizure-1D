"""The board runner and the host head it is scored with.

The runner executes on the KV260 and cannot be exercised here end to end, so
these tests pin the two places a board run could be scored wrongly without
anyone noticing: a board holding a different fold's weights, and a host
GAP/FC that does not reproduce the golden model's.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


board = _load("board_run_recording")
hvt = _load("hardware_verification_table")

MODEL_DIR = REPO / "hardware" / "export_chb15__chb15_40"
VECTORS = MODEL_DIR / "whole_recording" / "chb15_40_sz0"


def _embedded(tmp_path: Path, mutate=None) -> Path:
    d = tmp_path / "Embedded_Software"
    d.mkdir()
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(manifest)
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def test_same_fold_is_accepted(tmp_path):
    contract = board.check_same_model(_embedded(tmp_path), VECTORS)
    assert contract["provenance"]["fold_id"] == "chb15__chb15_40"


def test_board_with_another_folds_weights_is_refused(tmp_path):
    def other_fold(m):
        m["provenance"]["fold_id"] = "chb01__chb01_03"
    with pytest.raises(SystemExit, match="different models"):
        board.check_same_model(_embedded(tmp_path, other_fold), VECTORS)


def test_board_with_different_shifts_is_refused(tmp_path):
    def other_shift(m):
        m["layers"][3]["output_shift"] += 1
    with pytest.raises(SystemExit, match="shifts"):
        board.check_same_model(_embedded(tmp_path, other_shift), VECTORS)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    out.write_text("{}")
    monkeypatch.setattr("sys.argv", ["x", "--embedded-dir", str(tmp_path), "--vectors", str(VECTORS),
                                     "--output", str(out)])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        board.main()
    assert out.read_text() == "{}"


def test_host_head_reproduces_the_golden_logits():
    golden_mod, golden, _ = hvt.load_reference_model(MODEL_DIR)
    x = hvt.verify_onset_vectors.read_stream(MODEL_DIR / "test_vectors" / "00_input.txt", 8)
    logits, trace = golden.run(x)
    features = trace["13_context.1.pw"]
    assert hvt.host_logits(golden_mod, golden, features) == (int(logits[0, 0]), int(logits[1, 0]))


def test_feature_bytes_round_trip_channel_major():
    _, golden, _ = hvt.load_reference_model(MODEL_DIR)
    x = hvt.verify_onset_vectors.read_stream(MODEL_DIR / "test_vectors" / "00_input.txt", 8)
    _, trace = golden.run(x)
    ref = trace["13_context.1.pw"]
    on_wire = bytes(int(v) & 0xFF for v in ref.flatten("C")).hex()
    back = np.frombuffer(bytes.fromhex(on_wire), dtype=np.int8).astype(np.int64).reshape(ref.shape)
    np.testing.assert_array_equal(back, ref)
    # and it is the layout of the file the driver compares against
    expected = hvt.verify_onset_vectors.read_stream(MODEL_DIR / "test_vectors" / "13_context_1_pw.txt", 8)
    np.testing.assert_array_equal(back.flatten("C"), expected)


def test_runner_uses_only_the_standard_library():
    text = (REPO / "scripts" / "board_run_recording.py").read_text(encoding="utf-8")
    assert "import numpy" not in text and "from wearseizure" not in text
