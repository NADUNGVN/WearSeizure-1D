"""How far apart are the per-channel weight ranges? Read-only.

This decides the whole PTQ plan, and it cannot be answered from an untrained
model: at initialisation every channel is drawn from the same distribution and
the spread is about 2x, which says nothing. What matters is how far the channels
drift during training.

`wearseizure1d_k5only` is depthwise-separable throughout, and per-tensor
quantisation is known to fail on that family -- Krishnamoorthi (2018) showed
MobileNet losing badly at INT8 per-tensor while per-channel held up. A depthwise
layer has one independent filter per channel and no mixing to even them out, so
a single tensor-wide scale is set by the loudest channel and crushes the quiet
ones.

    python scripts/measure_weight_ranges.py profile=server data=chbmit \\
        model=wearseizure1d_k5only train.run_tag=L8

Reads checkpoints only -- no data, no GPU, no writes.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import nn

from wearseizure.models.factory import build_model
from wearseizure.utils.env import bootstrap_env
from wearseizure.utils.logging import get_logger
from wearseizure.utils.paths import fold_run_dir, run_tag_from_cfg, seeds_from_cfg
from wearseizure.utils.profile_guard import check_profile_data_pairing

log = get_logger(__name__)
bootstrap_env(sys.argv)

# Thresholds from docs/PLAN_ptq_method.md section 2. They decide how much of the
# PTQ ladder has to be built, so they are stated here rather than eyeballed.
UNIFORM, MODERATE = 4.0, 20.0


def spread(conv: nn.Conv1d) -> float:
    """max|w| of the widest output channel divided by that of the narrowest.

    This is exactly the ratio a per-tensor scale has to span. At 10x, the
    narrowest channel keeps about 3 of 8 bits.
    """
    per_ch = conv.weight.detach().flatten(1).abs().amax(dim=1)
    lo = float(per_ch.min())
    return float(per_ch.max()) / lo if lo > 0 else float("inf")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    check_profile_data_pairing(cfg)
    tag = run_tag_from_cfg(cfg)
    seeds = seeds_from_cfg(cfg)

    per_layer: dict[str, list[float]] = {}
    n_ckpt = 0
    for seed in seeds:
        run_dir = fold_run_dir(
            cfg.profile.artifacts_dir, cfg.model.name, cfg.split.name, cfg.window.name, seed, tag
        )
        for ckpt in sorted(Path(run_dir).glob("*.pt")):
            model = build_model(cfg)
            model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
            n_ckpt += 1
            for name, mod in model.named_modules():
                if isinstance(mod, nn.Conv1d) and mod.out_channels > 1:
                    per_layer.setdefault(name, []).append(spread(mod))

    if not per_layer:
        raise SystemExit(
            f"no checkpoints under {run_dir}. This needs TRAINED weights: an untrained "
            "model shows a spread of about 2x, which answers nothing."
        )

    log.info(f"{n_ckpt} checkpoints, {len(seeds)} seed(s), tag={tag or '<control>'}")
    log.info(f"{'layer':<26}{'type':<11}{'out_ch':>7}{'median':>9}{'max':>9}")
    worst = 0.0
    for name, vals in per_layer.items():
        model_mod = dict(build_model(cfg).named_modules())[name]
        kind = "depthwise" if model_mod.groups == model_mod.in_channels != 1 else "conv"
        med, mx = statistics.median(vals), max(vals)
        worst = max(worst, med)
        log.info(f"{name:<26}{kind:<11}{model_mod.out_channels:>7}{med:>9.1f}x{mx:>8.1f}x")

    log.info("=" * 62)
    log.info(f"worst median spread across layers: {worst:.1f}x")
    if worst < UNIFORM:
        log.info("UNIFORM (<4x): per-tensor may be enough. Try P0/P1 and skip CLE --")
        log.info("  building cross-layer equalisation for channels this even is wasted work.")
    elif worst < MODERATE:
        log.info("MODERATE (4-20x): per-channel weight scales are REQUIRED (ladder step P1).")
        log.info("  CLE may add a little on top; measure P1 before deciding to build it.")
    else:
        log.info("SEVERE (>20x): this is the MobileNet failure mode. Per-channel scales are")
        log.info("  mandatory, and cross-layer equalisation plus bias correction (P2, P4) are")
        log.info("  likely needed before INT8 is usable at all.")
    log.info("Thresholds and the ladder they select: docs/PLAN_ptq_method.md")


if __name__ == "__main__":
    main()
