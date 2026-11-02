"""Run one exported complete recording through the KV260, window by window.

Runs ON THE BOARD, with the board's system Python and the standard library
only. It drives the RTL team's compiled driver (`board_smoke_core`) exactly as
their own `verify_board_model.py` does -- one `--run --json` call per window --
but over every window of a complete recording instead of a fixed excerpt.
Their checkout is only read: nothing is written into it, no asset is changed,
no bitstream is loaded.

    sudo /usr/bin/python3 -I scripts/board_run_recording.py \\
        --embedded-dir /home/ubuntu/Anhquan/Embedded_Software \\
        --vectors hardware/export_chb15__chb15_40/whole_recording/chb15_40_sz0 \\
        --output /home/ubuntu/Anhquan/board_run_chb15_40.json

It records the raw final PL feature map (64 x 32 bytes, channel-major) the
board returns for each window. Comparing them with the golden model, and
turning them into alarms, is done afterwards by
`scripts/hardware_verification_table.py --board-run <output>` on any machine
with NumPy -- the board needs no NumPy.

About `--expected`: the driver insists on an expected-output file alongside
every input and reports whether the board matched it. A per-window golden file
would need NumPy on the board, so a fixed placeholder is passed instead, and
the driver's own match flag is ignored; what is kept is `ran_hardware` and the
feature bytes it read back. The comparison that matters happens afterwards,
against the golden model, for every byte of every window.

The run refuses to start when the board's assets come from a different fold
than the vectors, since that would compare two different models. It stops at
the first window the driver fails to run, rather than issuing further MMIO
after an unknown failure, and it never overwrites an existing output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FEATURE_BYTES = 2048  # 64 channels x 32 positions, the last layer the PL runs
WINDOW_SAMPLES = 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stream(path: Path) -> list[int]:
    out = []
    for token in path.read_text(encoding="utf-8").split():
        v = int(token, 16)
        if not 0 <= v <= 0xFF:
            raise ValueError(f"{path.name}: {token} is not one byte")
        out.append(v)
    return out


def check_same_model(embedded_dir: Path, vectors: Path) -> dict:
    """The board's assets and the vectors must come from the same trained fold."""
    contract = json.loads((vectors / "contract.json").read_text(encoding="utf-8"))
    board = json.loads((embedded_dir / "manifest.json").read_text(encoding="utf-8"))
    ours = json.loads((vectors.parents[1] / "manifest.json").read_text(encoding="utf-8"))
    want = contract["provenance"]["fold_id"]
    have = board.get("provenance", {}).get("fold_id")
    if have != want:
        raise SystemExit(f"board assets are fold {have!r}, these vectors are fold {want!r}: "
                         "comparing them would compare two different models")

    def shifts(m: dict) -> list:
        return [(layer.get("name"), layer.get("output_shift")) for layer in m.get("layers", [])]

    if shifts(board) != shifts(ours):
        raise SystemExit("board manifest and exported manifest disagree on per-layer shifts; "
                         "the board is not running the exported weights")
    return contract


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--embedded-dir", type=Path, required=True)
    ap.add_argument("--vectors", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N windows only (a smoke test)")
    ap.add_argument("--timeout-ms", type=int, default=2000)
    args = ap.parse_args()

    if args.output.exists():
        raise SystemExit(f"{args.output} exists; refusing to overwrite recorded evidence")
    binary = args.embedded_dir / "board_smoke_core"
    if not binary.is_file():
        raise SystemExit(f"{binary} not found; it is built on the board by the RTL team's Makefile")
    if platform.machine() != "aarch64":
        raise SystemExit(f"this is {platform.node()} ({platform.machine()}); run it on the KV260, "
                         "where the FPGA and the driver are")
    contract = check_same_model(args.embedded_dir, args.vectors)

    stream = read_stream(args.vectors / "stream_input.hex")
    if len(stream) != contract["excerpt"]["samples"]:
        raise SystemExit(f"stream holds {len(stream)} samples, contract says {contract['excerpt']['samples']}")
    with (args.vectors / "windows.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    assets = [binary, args.embedded_dir / "manifest.json", args.embedded_dir / "instructions.hex",
              *sorted((args.embedded_dir / "weights").glob("*"))]
    report = {
        "schema": "wearseizure-board-run-v1",
        "recording": contract["recording"],
        "fold_id": contract["provenance"]["fold_id"],
        "vectors_dir": str(args.vectors),
        "vectors_sha256": {n: sha256(args.vectors / n) for n in ("contract.json", "windows.csv", "stream_input.hex")},
        "embedded_dir": str(args.embedded_dir),
        "board_assets_sha256": {str(p.relative_to(args.embedded_dir)): sha256(p) for p in assets if p.is_file()},
        "host": platform.node(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete": False,
        "windows": [],
    }

    with tempfile.TemporaryDirectory(prefix="ws1d_board_") as tmp:
        tmp = Path(tmp)
        placeholder = tmp / "expected_placeholder.hex"
        placeholder.write_text("00\n" * FEATURE_BYTES, encoding="ascii")
        window_file = tmp / "input.hex"
        try:
            for i, row in enumerate(rows):
                off = int(row["stream_offset_samples"])
                x = stream[off:off + WINDOW_SAMPLES]
                if len(x) != WINDOW_SAMPLES:
                    raise RuntimeError(f"window {row['window_idx']}: stream ends early")
                window_file.write_text("".join(f"{v:02x}\n" for v in x), encoding="ascii")
                proc = subprocess.run(
                    [str(binary), "--run", "--json", "--assets-dir", str(args.embedded_dir),
                     "--input", str(window_file), "--expected", str(placeholder),
                     "--input-p", "3", "--timeout-ms", str(args.timeout_ms)],
                    capture_output=True, text=True, timeout=max(20, args.timeout_ms // 1000 * 5),
                    # Non-zero is expected: the placeholder never matches. ran_hardware decides.
                    check=False,
                )
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
                payload = json.loads(lines[-1]) if lines else {}
                out = payload.get("output") or []
                if payload.get("ran_hardware") is not True or len(out) != FEATURE_BYTES:
                    raise RuntimeError(f"window {row['window_idx']}: driver did not return a feature map "
                                       f"(exit {proc.returncode}): {proc.stderr[-1000:]}")
                if not all(isinstance(v, int) and -128 <= v <= 127 for v in out):
                    raise RuntimeError(f"window {row['window_idx']}: feature bytes out of int8 range")
                report["windows"].append({
                    "window_idx": int(row["window_idx"]),
                    "features_hex": bytes(v & 0xFF for v in out).hex(),
                    "inference_ms": payload.get("inference_ms"),
                })
                if (i + 1) % 200 == 0 or i + 1 == len(rows):
                    print(f"{i + 1}/{len(rows)} windows", flush=True)
            report["complete"] = len(report["windows"]) == len(rows) and not args.limit
        except Exception as exc:  # noqa: BLE001 -- stop at ANY failure, keep what was recorded
            report["error"] = str(exc)
            print(f"STOPPED: {exc}", file=sys.stderr)
        finally:
            report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            tmp_out = args.output.with_suffix(args.output.suffix + ".partial")
            tmp_out.write_text(json.dumps(report) + "\n", encoding="utf-8")
            os.replace(tmp_out, args.output)

    print(f"{'complete' if report['complete'] else 'INCOMPLETE'}: {len(report['windows'])} windows -> {args.output}")
    return 0 if "error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
