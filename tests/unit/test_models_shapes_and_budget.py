from __future__ import annotations

import torch

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
    import pytest

    model = WearSeizure1D(input_len=INPUT_LEN)
    with pytest.raises(ValueError):
        model(torch.randn(2, 1, 512))


def test_wearseizure1d_within_hard_budget():
    model = WearSeizure1D(input_len=INPUT_LEN)
    params = count_params(model)
    macs = count_macs(model, (1, INPUT_LEN))
    assert params <= PARAM_BUDGET_MAX, f"{params} params exceeds hard budget {PARAM_BUDGET_MAX}"
    assert macs <= MAC_BUDGET_MAX, f"{macs} MACs exceeds hard budget {MAC_BUDGET_MAX}"


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
