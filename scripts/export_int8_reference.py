"""Export INT8 weights, activation scales, and a handful of quantized sample
windows from a trained WearSeizure-1D checkpoint -- the artifacts the future
RTL testbench will need to replay bit-exact (memo 5.4 golden chain). Requires
`train.py` to have already produced a checkpoint for at least one fold.
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from wearseizure.data.loader import load_records_from_manifest
from wearseizure.data.manifest import hash_manifest, load_manifest
from wearseizure.data.splits import load_folds
from wearseizure.models.factory import build_model
from wearseizure.quant.qat import QATConv1d, QATLinear, prepare_qat, set_calibrating
from wearseizure.quant.scales import compute_symmetric_scale, compute_symmetric_scale_from_max
from wearseizure.rtl_interface.golden_io_contract import WINDOW_SAMPLES
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import ensure_dir, fold_run_dir, run_tag_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing

# Must run at import time: configs/profile/server.yaml interpolates
# ${oc.env:...} into hydra.run.dir, which Hydra resolves before main().
bootstrap_env(sys.argv)

log = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    if cfg.model.name != "wearseizure1d":
        raise ValueError("export_int8_reference.py is scoped to model=wearseizure1d")

    seed = int(cfg.seed)
    run_dir = fold_run_dir(
        cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name, cfg.window.name, seed, run_tag_from_cfg(cfg)
    )
    manifest_df = load_manifest(str(Path(cfg.data.manifest_path)))
    # Version-lock the split to the manifest it was built from. PROTOCOL.md
    # calls splits version-locked by manifest hash; until now nothing checked
    # it, so a manifest rebuilt from changed data would have been paired with
    # stale folds silently -- and every comparison against earlier rows would
    # have been invalid without anything saying so.
    folds = load_folds(
        str(Path(cfg.split.folds_path)), expected_manifest_hash=hash_manifest(manifest_df)
    )
    if not folds:
        raise RuntimeError("no folds found -- run make_splits.py and train.py first")
    fold = folds[0]
    checkpoint_path = run_dir / f"{fold.fold_id}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"{checkpoint_path} not found -- run train.py first")

    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model = prepare_qat(model, weight_bits=8, act_bits=8)
    model.eval()

    data_dir = cfg.data.generated_dir if cfg.data.name == "synthetic" else None
    raw_dir = cfg.data.raw_dir if cfg.data.name != "synthetic" else None
    records = load_records_from_manifest(manifest_df, data_dir=data_dir, raw_dir=raw_dir)

    sample_record = records[next(iter(fold.test_edf_ids))]
    n_windows = min(10, max(0, (len(sample_record.signal) - WINDOW_SAMPLES) // 256 + 1))

    windows = []
    set_calibrating(model, True)
    with torch.no_grad():
        for i in range(n_windows):
            start = i * 256
            x = torch.from_numpy(sample_record.signal[start : start + WINDOW_SAMPLES]).float().view(1, 1, -1)
            if x.shape[-1] != WINDOW_SAMPLES:
                break
            model(x)
            windows.append(x.numpy())
    set_calibrating(model, False)

    out_dir = ensure_dir(run_dir / "int8_export")
    layer_export: dict[str, np.ndarray] = {}
    for name, module in model.named_modules():
        if isinstance(module, (QATConv1d, QATLinear)):
            weight = module.conv.weight if isinstance(module, QATConv1d) else module.linear.weight
            weight_scale = compute_symmetric_scale(weight.detach(), module.weight_bits)
            act_scale = compute_symmetric_scale_from_max(module.act_running_max, module.act_bits)
            layer_export[f"{name}.weight_int8"] = weight_scale.quantize(weight.detach()).numpy().astype(np.int8)
            layer_export[f"{name}.weight_scale"] = np.array(weight_scale.scale, dtype=np.float64)
            layer_export[f"{name}.act_scale"] = np.array(act_scale.scale, dtype=np.float64)

    np.savez(out_dir / "layers.npz", **layer_export)
    stacked = np.concatenate(windows, axis=0) if windows else np.zeros((0, 1, WINDOW_SAMPLES), dtype=np.float32)
    np.savez(out_dir / "sample_windows.npz", windows=stacked)

    n_layers = sum(1 for k in layer_export if k.endswith(".weight_scale"))
    log.info(f"exported {n_layers} quantized layers and {len(windows)} sample windows -> {out_dir}")


if __name__ == "__main__":
    main()
