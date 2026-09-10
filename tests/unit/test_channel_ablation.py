"""The channel ablation: three arms that must differ in exactly one thing.

The experiment answers "what does the single-channel constraint cost under a
leakage-safe protocol". That answer is only worth anything if the arms are
identical apart from channel count, so these tests pin the two ways they could
silently stop being identical: the montage each arm reads, and the model each
arm builds.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "run_channel_ablation", ROOT / "scripts" / "run_channel_ablation.py")
ablation = importlib.util.module_from_spec(_spec)
sys.modules["run_channel_ablation"] = ablation
_spec.loader.exec_module(ablation)


def _cfg(in_channels: int = 1):
    return OmegaConf.create({
        "model": {
            "name": "wearseizure1d_k5only", "in_channels": in_channels,
            "input_len": 1024, "stem_out_channels": 8,
            "stage_out_channels": [16, 24, 32, 48], "context_channels": 64,
            "kernels": [5], "dilations": [1, 2, 4], "kernel_mode": "k5_only",
            "num_classes": 2, "param_budget_max": 32000, "mac_budget_max": 2000000,
        },
        "window": {"window_s": 4.0}, "data": {"fs_hz": 256},
    })


def test_montages_are_chung_2024s_verbatim():
    """The published ablation is the comparison target, so the montage must be
    theirs. A tidied or reordered list would measure something else and could
    not be placed beside their numbers."""
    assert len(ablation.CHUNG_18) == 18
    assert len(ablation.CHUNG_4) == 4
    assert set(ablation.CHUNG_4) <= set(ablation.CHUNG_18)
    # The four wearable positions this project uses live in the 18-channel set,
    # so the single-channel arm is a subset of the wider arms rather than a
    # different montage.
    for ch in ("FP1-F3", "P7-O1", "P8-O2", "FP2-F4"):
        assert ch in ablation.CHUNG_18
    assert len(set(ablation.CHUNG_18)) == 18, "a duplicated label would silently narrow the arm"


def test_single_channel_arm_uses_the_patients_own_confirmed_position():
    """The 1-channel arm must be the production configuration, not a fixed
    default: each patient's channel is the one clinically confirmed for them."""
    assert ablation.channels_for_arm(1, "chb01") == ["P8-O2"]
    assert ablation.channels_for_arm(1, "chb03") == ["FP1-F3"]
    assert ablation.channels_for_arm(1, "chb15") == ["P7-O1"]
    assert ablation.channels_for_arm(4, "chb01") == ablation.CHUNG_4
    assert ablation.channels_for_arm(18, "chb01") == ablation.CHUNG_18


def test_unknown_arm_is_refused():
    with pytest.raises(SystemExit, match="unknown arm"):
        ablation.channels_for_arm(7, "chb01")


@pytest.mark.parametrize("n", [1, 4, 18])
def test_each_arm_builds_a_model_that_reads_that_many_channels(n):
    model = ablation.build_arm_model(_cfg(), n)
    x = torch.randn(2, n, 1024)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2)
    assert model.stem[0].in_channels == n


def test_only_the_stem_changes_between_arms():
    """Widths, kernels, dilations and the classifier are held fixed, so the
    arms differ in exactly one thing. If a later edit made channel count change
    anything else, the ablation would be measuring two variables."""
    one, eighteen = ablation.build_arm_model(_cfg(), 1), ablation.build_arm_model(_cfg(), 18)
    shapes_one = {k: tuple(v.shape) for k, v in one.state_dict().items()}
    shapes_18 = {k: tuple(v.shape) for k, v in eighteen.state_dict().items()}
    assert set(shapes_one) == set(shapes_18)
    differing = [k for k in shapes_one if shapes_one[k] != shapes_18[k]]
    assert differing == ["stem.0.weight"], differing

    # And the cost of the extra channels is confined to that one tensor.
    delta = sum(v.numel() for k, v in eighteen.state_dict().items() if k in differing)
    delta -= sum(v.numel() for k, v in one.state_dict().items() if k in differing)
    assert delta == 8 * 7 * (18 - 1)


def test_input_length_is_derived_not_hardcoded():
    """`build_model` derives input_len from window_s x fs, so a 4 s window at
    256 Hz must give 1024 -- the length every exported artefact assumes."""
    model = ablation.build_arm_model(_cfg(), 4)
    with torch.no_grad():
        model(torch.randn(1, 4, 1024))
    with pytest.raises(ValueError, match="expected input length"):
        model(torch.randn(1, 4, 512))
