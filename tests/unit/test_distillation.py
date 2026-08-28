"""Lever L3: multi-channel teacher distilled into the single-channel student.

Lever L5 established that the missing information cannot be bought with more
pre-training data -- widening the corpus changed nothing at four electrode
positions and made things significantly worse at one. L3 goes the other way and
extracts more from the recordings already in hand.

The property everything else depends on is alignment: the teacher's logit array
is matched to the student's windows by ORDER, so if the two ever built their
window lists differently the student would be trained against another window's
soft targets, silently and with no error. These tests pin that, the loss itself,
and that the default path is untouched.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from wearseizure.data.dataset import build_fold_datasets
from wearseizure.data.splits import make_patient_specific_loso_edf
from wearseizure.models.teacher import MultiChannelTeacher, build_teacher
from wearseizure.training.distill import (
    MultiChannelWindowDataset,
    teacher_logits_for_windows,
    windows_for_partition,
)
from wearseizure.training.loop import _distillation_loss, train_classifier

# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def test_distillation_loss_is_zero_when_the_student_matches_the_teacher():
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-3.0, 4.0]])
    assert _distillation_loss(logits, logits.clone(), temperature=2.0).item() == pytest.approx(0.0, abs=1e-6)


def test_distillation_loss_grows_as_the_student_diverges():
    teacher = torch.tensor([[3.0, -3.0]])
    near = _distillation_loss(torch.tensor([[2.0, -2.0]]), teacher, 2.0).item()
    far = _distillation_loss(torch.tensor([[-3.0, 3.0]]), teacher, 2.0).item()
    assert 0 < near < far


def test_temperature_scaling_keeps_the_soft_term_comparable():
    """The T**2 factor is not cosmetic: softening shrinks the soft term's
    gradients by roughly 1/T**2, so without it the hard/soft balance would
    silently depend on the temperature."""
    student = torch.tensor([[2.0, -1.0]])
    teacher = torch.tensor([[1.0, 0.0]])
    at_1 = _distillation_loss(student, teacher, 1.0).item()
    at_4 = _distillation_loss(student, teacher, 4.0).item()
    # Same order of magnitude rather than 16x apart.
    assert 0.25 < at_4 / at_1 < 4.0


def test_invalid_distillation_settings_are_refused():
    x = torch.randn(16, 1, 8)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="distill_alpha"):
            train_classifier(model, loader, loader, epochs=1, lr=1e-3, weight_decay=0.0,
                             distill_alpha=bad)
    with pytest.raises(ValueError, match="distill_temperature"):
        train_classifier(model, loader, loader, epochs=1, lr=1e-3, weight_decay=0.0,
                         distill_temperature=0.0)


# ---------------------------------------------------------------------------
# Alignment -- the property a silent bug would break
# ---------------------------------------------------------------------------


def test_teacher_and_student_build_identical_window_lists(synthetic_cohort):
    """Windows come from `record.meta` and index arithmetic, never from signal
    values, so an 18-channel view must produce exactly the student's windows."""
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    datasets, _, _ = build_fold_datasets(records, fold, 4.0, 1.0)

    teacher_windows = windows_for_partition(records, fold.train_edf_ids, 4.0, 1.0)
    student_windows = datasets["train"].windows
    assert len(teacher_windows) == len(student_windows)
    for a, b in zip(teacher_windows, student_windows):
        assert (a.edf_id, a.start_idx, a.end_idx, a.label) == (b.edf_id, b.start_idx, b.end_idx, b.label)


def test_a_mismatched_logit_array_is_refused_rather_than_broadcast(synthetic_cohort):
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    n = len(windows_for_partition(records, fold.train_edf_ids, 4.0, 1.0))
    with pytest.raises(ValueError, match="teacher_logits has"):
        build_fold_datasets(records, fold, 4.0, 1.0, teacher_logits=np.zeros((n - 1, 2), np.float32))


def test_logits_reach_only_the_train_partition(synthetic_cohort):
    """Distillation is a training signal. Attaching it to val or test would put
    a multi-channel model's opinion inside the evaluation path."""
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    n = len(windows_for_partition(records, fold.train_edf_ids, 4.0, 1.0))
    datasets, _, _ = build_fold_datasets(
        records, fold, 4.0, 1.0, teacher_logits=np.zeros((n, 2), np.float32)
    )
    assert datasets["train"].teacher_logits is not None
    assert datasets["val"].teacher_logits is None
    assert datasets["test"].teacher_logits is None


def test_the_train_dataset_hands_back_the_matching_logit_row(synthetic_cohort):
    manifest_df, records = synthetic_cohort
    fold = make_patient_specific_loso_edf(manifest_df, seed=0)[0]
    n = len(windows_for_partition(records, fold.train_edf_ids, 4.0, 1.0))
    logits = np.arange(n * 2, dtype=np.float32).reshape(n, 2)
    ds = build_fold_datasets(records, fold, 4.0, 1.0, teacher_logits=logits)[0]["train"]

    for idx in (0, n // 2, n - 1):
        item = ds[idx]
        assert len(item) == 3
        assert torch.allclose(item[2], torch.from_numpy(logits[idx]))
    # The batched fast path must agree with the per-item path.
    batch = ds.__getitems__([n - 1, 0, 3])
    assert torch.allclose(batch[0][2], torch.from_numpy(logits[n - 1]))
    assert torch.allclose(batch[1][2], torch.from_numpy(logits[0]))


def test_scored_logits_come_back_in_window_order():
    """`teacher_logits_for_windows` must never shuffle: the order IS the
    alignment."""
    from wearseizure.data.windowing import Window

    signals = {"e0": np.tile(np.arange(100, dtype=np.float32), (3, 1))}
    windows = [
        Window(edf_id="e0", subject_id="s", start_idx=i, end_idx=i + 10,
               start_sec=0.0, end_sec=1.0, label=0)
        for i in range(0, 50, 10)
    ]

    class _FirstSample(nn.Module):
        """Logit row equals the window's first sample, so order is checkable."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            first = x[:, 0, 0].unsqueeze(1)
            return torch.cat([first, -first], dim=1)

    out = teacher_logits_for_windows(_FirstSample(), signals, windows, batch_size=2)
    assert out.shape == (5, 2)
    assert np.allclose(out[:, 0], [0, 10, 20, 30, 40])


# ---------------------------------------------------------------------------
# The teacher itself
# ---------------------------------------------------------------------------


def test_teacher_mixes_channels_in_its_first_layer():
    """The cross-channel mixing is the only thing the teacher has that the
    student does not -- if the first conv were depthwise it would be pointless."""
    teacher = build_teacher(in_channels=18)
    first = teacher.features[0]
    assert isinstance(first, nn.Conv1d)
    assert first.in_channels == 18 and first.groups == 1


def test_teacher_accepts_any_channel_count_and_outputs_two_logits():
    for c in (1, 3, 18, 23):
        out = build_teacher(c)(torch.randn(2, c, 1024))
        assert out.shape == (2, 2)


def test_teacher_refuses_a_single_channel_shaped_batch():
    with pytest.raises(ValueError, match=r"expected \(B, C, L\)"):
        MultiChannelTeacher(in_channels=1)(torch.randn(2, 1024))


def test_multichannel_dataset_slices_all_channels():
    from wearseizure.data.windowing import Window

    signals = {"e0": np.arange(3 * 100, dtype=np.float32).reshape(3, 100)}
    w = Window("e0", "s", 10, 20, 0.0, 1.0, 1)
    x, y = MultiChannelWindowDataset(signals, [w])[0]
    assert x.shape == (3, 10)
    assert y.item() == 1
    assert np.allclose(x.numpy(), signals["e0"][:, 10:20])


# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------


def test_alpha_zero_reproduces_training_without_a_teacher():
    x = torch.randn(64, 1, 16)
    y = torch.randint(0, 2, (64,))
    plain = DataLoader(TensorDataset(x, y), batch_size=16)
    with_t = DataLoader(TensorDataset(x, y, torch.randn(64, 2)), batch_size=16)

    def _model():
        torch.manual_seed(0)
        return nn.Sequential(nn.Flatten(), nn.Linear(16, 2))

    torch.manual_seed(1)
    a = train_classifier(_model(), plain, plain, epochs=2, lr=1e-2, weight_decay=0.0)
    torch.manual_seed(1)
    b = train_classifier(_model(), with_t, plain, epochs=2, lr=1e-2, weight_decay=0.0,
                         distill_alpha=0.0)
    for (_, va), (_, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.equal(va, vb)
