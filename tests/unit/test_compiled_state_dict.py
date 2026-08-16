"""`torch.compile` must stay invisible to everything that touches checkpoints.

A compiled module's `state_dict()` keys are prefixed with `_orig_mod.`. If that
leaked out, every checkpoint this project writes would silently stop matching:
the per-fold `<fold_id>.pt` files, and -- worse -- the cohort pre-training cache
in `training/pretrain.py`, which is loaded straight back into a plain
`build_model(cfg)` instance on a later run, possibly one where compilation is
turned off again.

Uses `backend="eager"`, which exercises the same `OptimizedModule` wrapper
without invoking Inductor, so the test needs no C++ toolchain.
"""
from __future__ import annotations

import torch

from wearseizure.models.wearseizure1d import WearSeizure1D
from wearseizure.training.loop import unwrap_compiled


def _model() -> WearSeizure1D:
    return WearSeizure1D(input_len=1024)


def test_compiled_state_dict_really_does_get_prefixed():
    # If this ever stops being true, `unwrap_compiled` is dead weight and the
    # tests below stop testing anything -- so pin the hazard itself.
    compiled = torch.compile(_model(), backend="eager")
    assert any(key.startswith("_orig_mod.") for key in compiled.state_dict())


def test_unwrap_restores_exactly_the_plain_key_set():
    plain = _model()
    compiled = torch.compile(_model(), backend="eager")
    assert set(unwrap_compiled(compiled).state_dict()) == set(plain.state_dict())


def test_unwrapped_state_dict_loads_into_a_fresh_uncompiled_model():
    compiled = torch.compile(_model(), backend="eager")
    fresh = _model()
    fresh.load_state_dict(unwrap_compiled(compiled).state_dict())


def test_unwrap_is_a_no_op_for_an_uncompiled_model():
    plain = _model()
    assert unwrap_compiled(plain) is plain
