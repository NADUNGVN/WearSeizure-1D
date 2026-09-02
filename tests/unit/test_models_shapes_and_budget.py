from __future__ import annotations

import pytest
import torch
from pathlib import Path

from wearseizure.eval.metrics_model import count_macs, count_params
from wearseizure.models.baselines import Compact1DBaseline, FrontiersBaseline2D
from wearseizure.models.wearseizure1d import WearSeizure1D

INPUT_LEN = 1024
PARAM_BUDGET_MAX = 32000
MAC_BUDGET_MAX = 2_000_000


def test_wearseizure1d_output_shape():
    model = WearSeizure1D(input_len=INPUT_LEN)
    x = torch.randn(4, 1, INPUT_LEN)
    y = model(x)
    assert y.shape == (4, 2)


def test_wearseizure1d_rejects_wrong_input_length():
    model = WearSeizure1D(input_len=INPUT_LEN)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 512))


def test_wearseizure1d_within_hard_budget():
    model = WearSeizure1D(input_len=INPUT_LEN)
    params = count_params(model)
    macs = count_macs(model, (1, INPUT_LEN))
    assert params <= PARAM_BUDGET_MAX, f"{params} params exceeds hard budget {PARAM_BUDGET_MAX}"
    assert macs <= MAC_BUDGET_MAX, f"{macs} MACs exceeds hard budget {MAC_BUDGET_MAX}"


@pytest.mark.parametrize(
    "kernel_mode,dilations",
    [
        ("multi_scale", (1, 2, 4)),  # default (variant D)
        ("k3_only", (1, 1, 1)),      # variant A
        ("k5_only", (1, 2, 4)),      # variant B
        ("multi_scale", (1, 1, 1)),  # variant C: no dilation
    ],
)
def test_wearseizure1d_kernel_ablation_variants_build_and_fit_budget(kernel_mode, dilations):
    model = WearSeizure1D(input_len=INPUT_LEN, kernel_mode=kernel_mode, dilations=dilations)
    x = torch.randn(2, 1, INPUT_LEN)
    y = model(x)
    assert y.shape == (2, 2)
    params = count_params(model)
    macs = count_macs(model, (1, INPUT_LEN))
    assert params <= PARAM_BUDGET_MAX, f"{kernel_mode}/{dilations}: {params} params exceeds {PARAM_BUDGET_MAX}"
    assert macs <= MAC_BUDGET_MAX, f"{kernel_mode}/{dilations}: {macs} MACs exceeds {MAC_BUDGET_MAX}"


def test_wearseizure1d_rejects_unknown_kernel_mode():
    with pytest.raises(ValueError):
        WearSeizure1D(input_len=INPUT_LEN, kernel_mode="not_a_mode")


def test_frontiers_baseline_output_shape():
    model = FrontiersBaseline2D(input_len=INPUT_LEN)
    x = torch.randn(3, 1, INPUT_LEN)
    y = model(x)
    assert y.shape == (3, 2)


def test_compact1d_baseline_output_shape():
    model = Compact1DBaseline(input_len=INPUT_LEN)
    x = torch.randn(3, 1, INPUT_LEN)
    y = model(x)
    assert y.shape == (3, 2)


# ---------------------------------------------------------------------------
# The capacity ladder (configs/model/wearseizure1d_k5only_{ctx16,wide}.yaml)
# ---------------------------------------------------------------------------

# The gate is a quarter of the heaviest reproduced baseline, baseline_frontiers2d
# at 2,523,328 MACs. It is one of the paper's two axes, so a config drifting
# past it silently would cost the claim, not just a number.
FRONTIERS2D_MACS = 2_523_328
MAC_GATE = FRONTIERS2D_MACS // 4  # 630,832

# (context_channels, stage_out_channels) -> (MACs, params), measured with thop
# at input_len=1024. Pinned because the ladder's whole argument is that one
# variable changes per rung and the budget still holds.
LADDER = {
    "k5only":       ((64, (16, 24, 32, 48)), (585_920, 11_786)),
    "k5only_ctx16": ((16, (16, 24, 32, 48)), (367_664, 5_114)),
    "k5only_wide":  ((16, (24, 36, 48, 72)), (626_736, 9_414)),
}


@pytest.mark.parametrize("name", sorted(LADDER))
def test_capacity_ladder_macs_and_params_are_what_the_configs_claim(name):
    from thop import profile

    (ctx, stages), (want_macs, want_params) = LADDER[name]
    model = WearSeizure1D(
        in_channels=1, input_len=1024, stem_out_channels=8,
        stage_out_channels=stages, context_channels=ctx,
        dilations=(1, 2, 4), num_classes=2, kernel_mode="k5_only",
    )
    macs, params = profile(model, inputs=(torch.randn(1, 1, 1024),), verbose=False)
    assert int(macs) == want_macs, f"{name}: {int(macs):,} MACs, config says {want_macs:,}"
    assert int(params) == want_params
    assert int(macs) <= MAC_GATE, f"{name} exceeds the M10 gate of {MAC_GATE:,}"


def test_the_ladder_changes_one_variable_per_rung():
    """`wide` versus `k5only` changes context AND stage width at once, so it is
    only interpretable through `ctx16`. Comparing the two ends directly is the
    same confound that invalidated the row-15 architecture claim, where k5only
    had cohort pre-training and the baselines did not."""
    base_ctx, base_stages = LADDER["k5only"][0]
    ctx16_ctx, ctx16_stages = LADDER["k5only_ctx16"][0]
    wide_ctx, wide_stages = LADDER["k5only_wide"][0]

    assert ctx16_stages == base_stages and ctx16_ctx != base_ctx, "rung 1 must move context alone"
    assert wide_ctx == ctx16_ctx and wide_stages != ctx16_stages, "rung 2 must move stages alone"
    assert all(w == round(b * 1.5) for w, b in zip(wide_stages, base_stages))


def test_widening_the_stages_still_costs_fewer_parameters_than_the_baseline():
    """The counter-intuitive part of the design, and the reason it fits: the
    context block was where the parameters were, so 50% wider stages still come
    out 20% smaller than the model they widen."""
    assert LADDER["k5only_wide"][1][1] < LADDER["k5only"][1][1]


@pytest.mark.parametrize("name", ["wearseizure1d_k5only_ctx16", "wearseizure1d_k5only_wide"])
def test_new_ladder_configs_are_registered_and_buildable(name):
    from omegaconf import OmegaConf

    from wearseizure.models.factory import MODEL_FACTORIES, build_model

    assert name in MODEL_FACTORIES, f"{name} must be in MODEL_FACTORIES or train.py cannot build it"
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "model" / f"{name}.yaml"
    cfg = OmegaConf.create({
        "model": OmegaConf.load(cfg_path),
        "window": {"window_s": 4.0},
        "data": {"fs_hz": 256},
    })
    assert cfg.model.name == name, "the config's own name field must match its filename"
    out = build_model(cfg)(torch.randn(2, 1, 1024))
    assert out.shape == (2, 2)
