"""Measure the ACTUAL parameter and MAC count of every model variant.

Why this script exists: until now the only params/MAC figures anywhere in the
project were (a) the Table 4 *design target* in the `models/wearseizure1d.py`
docstring (13,810 params / 644,000 MACs) and (b) the hard *budget ceilings* in
`configs/model/*.yaml`, enforced as an upper bound by
`tests/unit/test_models_shapes_and_budget.py`. Neither is a measurement.
`docs/EXPERIMENT_LOG_G1a.md` section 4 lists this as a gap: "Actual measured
param count / MAC count for any model variant -- never printed".

The efficiency claim is one of the project's strongest (roughly 30x smaller
than the nearest quantised 1D-CNN seizure detector, arXiv:2607.16296 at
0.44 MB INT8), so it needs a measured number behind it, not a target.

Deliberately does NOT use Hydra: this is a pure measurement over config files
with no data, no artifacts directory, and no run directory to pollute. It reads
`configs/model/*.yaml` and `configs/window/*.yaml` directly.

    python scripts/measure_model_size.py
    python scripts/measure_model_size.py --window w2s_stride1s
    python scripts/measure_model_size.py --json artifacts/model_size.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from wearseizure.eval.metrics_model import count_macs, count_params, weight_bytes
from wearseizure.models.factory import MODEL_FACTORIES, build_model

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _cfg_for(model_name: str, window_name: str, fs_hz: int):
    """Minimal config carrying only what models/factory.build_model reads."""
    return OmegaConf.create(
        {
            "model": OmegaConf.load(CONFIGS / "model" / f"{model_name}.yaml"),
            "window": OmegaConf.load(CONFIGS / "window" / f"{window_name}.yaml"),
            "data": {"fs_hz": fs_hz},
        }
    )


def measure(model_name: str, window_name: str, fs_hz: int) -> dict:
    cfg = _cfg_for(model_name, window_name, fs_hz)
    model = build_model(cfg).eval()
    input_len = round(cfg.window.window_s * fs_hz)
    params = count_params(model)
    macs = count_macs(model, (cfg.model.in_channels, input_len))
    return {
        "model": model_name,
        "window": window_name,
        "input_len": input_len,
        "params": params,
        "macs": macs,
        "int8_weight_bytes": weight_bytes(model, bits=8),
        "param_budget_max": cfg.model.get("param_budget_max"),
        "mac_budget_max": cfg.model.get("mac_budget_max"),
        "param_target": cfg.model.get("param_target"),
        "mac_target": cfg.model.get("mac_target"),
    }


def _verdict(value: int, budget: int | None) -> str:
    if budget is None:
        return "-"
    return "within" if value <= budget else "OVER"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window", default="w4s_stride1s", help="window config name (configs/window/*.yaml)")
    ap.add_argument("--fs-hz", type=int, default=256, help="sampling rate; CHB-MIT and the synthetic generator are both 256")
    ap.add_argument("--json", default=None, help="also write results to this path")
    args = ap.parse_args()

    rows = [measure(name, args.window, args.fs_hz) for name in MODEL_FACTORIES]

    header = f"{'model':<26}{'params':>10}{'':>3}{'MACs':>12}{'':>3}{'INT8 KiB':>10}{'':>4}{'vs budget':>10}"
    print(f"\nwindow={args.window}  input_len={rows[0]['input_len']}  fs={args.fs_hz}Hz\n")
    print(header)
    print("-" * len(header))
    for r in rows:
        verdict = f"{_verdict(r['params'], r['param_budget_max'])}/{_verdict(r['macs'], r['mac_budget_max'])}"
        print(
            f"{r['model']:<26}{r['params']:>10,}{'':>3}{r['macs']:>12,}{'':>3}"
            f"{r['int8_weight_bytes'] / 1024:>10.1f}{'':>4}{verdict:>10}"
        )

    target_row = next((r for r in rows if r["model"] == "wearseizure1d"), None)
    if target_row and target_row["param_target"]:
        print(
            f"\nwearseizure1d vs Table 4 design target: "
            f"params {target_row['params']:,} vs {target_row['param_target']:,} "
            f"({target_row['params'] - target_row['param_target']:+,}), "
            f"MACs {target_row['macs']:,} vs {target_row['mac_target']:,} "
            f"({target_row['macs'] - target_row['mac_target']:+,})"
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
