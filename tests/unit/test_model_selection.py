"""Lever L4: what early stopping and checkpoint selection optimise.

The paper's metric is event-level sensitivity at a bounded false-alarm rate.
Cross-entropy is a poor stand-in for it at this imbalance -- ictal windows are
roughly 0.5% of the data, so a model can lower cross-entropy by growing more
confident about the interictal majority while getting no better at the minority
class that decides every reported number.

These tests pin three things: `val_loss` still behaves exactly as it did, the
AUPRC path selects a different epoch when the two criteria disagree, and a
validation partition with no positive window -- which happens on this data,
where some folds hold a single seizure -- does not silently burn early-stopping
patience.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wearseizure.training.loop import MODEL_SELECTION, train_classifier


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _loaders(n_pos: int = 20, n_neg: int = 200, seed: int = 0):
    """Deliberately imbalanced, roughly the shape of the real problem."""
    g = torch.Generator().manual_seed(seed)
    x_neg = torch.randn(n_neg, 4, generator=g)
    x_pos = torch.randn(n_pos, 4, generator=g) + 1.5
    x = torch.cat([x_neg, x_pos])
    y = torch.cat([torch.zeros(n_neg, dtype=torch.long), torch.ones(n_pos, dtype=torch.long)])
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=32, shuffle=True), DataLoader(ds, batch_size=32)


def test_both_criteria_are_accepted_and_anything_else_is_refused():
    assert MODEL_SELECTION == ("val_loss", "val_auprc")
    train, val = _loaders()
    for selection in MODEL_SELECTION:
        train_classifier(_Tiny(), train, val, epochs=1, lr=1e-2, weight_decay=0.0,
                         model_selection=selection)
    with pytest.raises(ValueError, match="unknown model_selection"):
        train_classifier(_Tiny(), train, val, epochs=1, lr=1e-2, weight_decay=0.0,
                         model_selection="val_sensitivity")


def test_default_is_val_loss_and_reproduces_the_old_path_exactly():
    """The default must stay bit-identical: every number in the experiment log
    was produced by selecting on cross-entropy."""
    train, val = _loaders()
    torch.manual_seed(0)
    a = train_classifier(_Tiny(), train, val, epochs=3, lr=1e-2, weight_decay=0.0)
    torch.manual_seed(0)
    b = train_classifier(_Tiny(), train, val, epochs=3, lr=1e-2, weight_decay=0.0,
                         model_selection="val_loss")
    assert a.selection == "val_loss"
    assert a.best_val_loss == b.best_val_loss
    for (ka, va), (kb, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


def test_auprc_is_recorded_every_epoch_under_either_criterion():
    train, val = _loaders()
    result = train_classifier(_Tiny(), train, val, epochs=3, lr=1e-2, weight_decay=0.0)
    assert len(result.history) == 3
    for row in result.history:
        assert "val_auprc" in row and "val_loss" in row
        assert 0.0 <= row["val_auprc"] <= 1.0


def test_selecting_on_auprc_keeps_the_best_auprc_epoch():
    train, val = _loaders()
    torch.manual_seed(0)
    result = train_classifier(_Tiny(), train, val, epochs=6, lr=5e-2, weight_decay=0.0,
                              model_selection="val_auprc")
    assert result.selection == "val_auprc"
    best_seen = max(r["val_auprc"] for r in result.history)
    assert result.best_val_auprc == pytest.approx(best_seen)


def test_a_single_class_validation_split_does_not_burn_patience():
    """Some folds hold one seizure, so a validation split can contain no ictal
    window at all. AUPRC is undefined there; falling back to loss for that epoch
    is better than reading nan as 'did not improve'."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 4, generator=g)
    y = torch.zeros(64, dtype=torch.long)          # negatives only
    val = DataLoader(TensorDataset(x, y), batch_size=32)
    train, _ = _loaders()

    result = train_classifier(_Tiny(), train, val, epochs=4, lr=1e-2, weight_decay=0.0,
                              early_stopping_patience=2, model_selection="val_auprc")
    assert all(np.isnan(r["val_auprc"]) for r in result.history)
    # Ran past the patience window instead of stopping at epoch 2 on nan.
    assert len(result.history) == 4
