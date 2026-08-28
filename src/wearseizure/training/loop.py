"""Generic binary-classifier training loop with early stopping on a
configurable validation criterion.
Shared by every model (baseline or WearSeizure-1D) and every precision mode
(the QAT wrapper in `quant/qat.py` reuses this by passing a fake-quantized
model in as `model`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from wearseizure.eval.metrics_segment import compute_segment_metrics
from wearseizure.utils.logging import get_logger

log = get_logger(__name__)


MODEL_SELECTION = ("val_loss", "val_auprc")


@dataclass
class TrainResult:
    model: nn.Module
    best_val_loss: float
    history: list[dict] = field(default_factory=list)
    best_val_auprc: float = float("nan")
    selection: str = "val_loss"


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """The original module behind a `torch.compile` wrapper, or the model itself.

    `torch.compile` returns an `OptimizedModule` whose `state_dict()` keys are
    prefixed with `_orig_mod.`. Every checkpoint this project writes -- fold
    checkpoints, and the cohort pre-training cache that `training/pretrain.py`
    loads back into a plain model -- would silently stop matching. Unwrapping
    before anything touches `state_dict` keeps compilation invisible to the
    rest of the codebase.
    """
    return getattr(model, "_orig_mod", model)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str = "cpu",
    early_stopping_patience: int = 8,
    compile_mode: str | None = None,
    model_selection: str = "val_loss",
    distill_alpha: float = 0.0,
    distill_temperature: float = 2.0,
) -> TrainResult:
    """`compile_mode` is passed straight to `torch.compile(mode=...)`.

    Why it is worth having: this model is ~12k parameters, so a batch of 256 is
    a few hundred MFLOP -- microseconds of arithmetic on a Quadro RTX 8000 --
    while the forward/backward issues on the order of a hundred tiny CUDA
    kernels. Wall-clock per step is therefore dominated by launch overhead, not
    compute, which is exactly what makes GPU-utilisation and memory-bandwidth
    counters read low no matter how many DataLoader workers are running.
    `mode="reduce-overhead"` captures the step into CUDA graphs and replays it,
    which is the direct fix for that specific bottleneck.

    Left off by default: it changes nothing numerically in principle, but it is
    a large behavioural change to take on trust in the middle of a run that
    feeds a publication. Turn it on with `train.compile_mode=reduce-overhead`
    and compare against an uncompiled run before adopting it.

    `distill_alpha` (lever L3) mixes a soft-target loss against teacher
    logits the dataset carries per window; 0.0 disables it, which is the
    default and reproduces every recorded result.

    `model_selection` (lever L4) chooses what early stopping and checkpoint
    selection optimise:

    - `"val_loss"` -- cross-entropy. The default, and what every result in
      docs/EXPERIMENT_LOG_G1a.md was produced with.
    - `"val_auprc"` -- area under the precision-recall curve on the validation
      windows.

    Why the second option exists: the paper's metric is event-level sensitivity
    at a bounded false-alarm rate, and cross-entropy is a poor stand-in for it
    here. Ictal windows are roughly 0.5% of the data, so a model can lower
    cross-entropy by growing more confident about the interictal majority while
    getting no better at the minority class that decides every reported number.
    AUPRC is computed against the positive class specifically and ignores the
    true-negative mass, which is why it is the standard choice for detection at
    this kind of imbalance.

    Deliberately NOT selecting on event sensitivity/FAR directly, tempting as
    that is: those need `threshold_on`/`threshold_off`, which
    `training/threshold_selection.py` fits on this same validation partition
    after training. Selecting a checkpoint by a metric that depends on
    thresholds fitted to the checkpoint is circular. AUPRC is threshold-free,
    which breaks the circle while still scoring the axis that matters.
    """
    if not 0.0 <= distill_alpha <= 1.0:
        raise ValueError(f"distill_alpha must be in [0, 1], got {distill_alpha}")
    if distill_temperature <= 0:
        raise ValueError(f"distill_temperature must be > 0, got {distill_temperature}")
    if model_selection not in MODEL_SELECTION:
        raise ValueError(
            f"unknown model_selection {model_selection!r}, expected one of {MODEL_SELECTION}"
        )
    model.to(device)
    if compile_mode:
        log.info(f"torch.compile(mode={compile_mode!r}) enabled")
        model = torch.compile(model, mode=compile_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_auprc = float("nan")
    # Higher is better for AUPRC, lower for loss, so the comparison is written
    # once here rather than branching inside the epoch loop.
    best_score = float("-inf")
    best_state = None
    patience_left = early_stopping_patience
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for batch in train_loader:
            # Two- or three-tuple: the dataset attaches teacher logits only when
            # lever L3 is on, and only to the train partition.
            if len(batch) == 3:
                x, y, teacher = batch
                teacher = teacher.to(device, non_blocking=True)
            else:
                (x, y), teacher = batch, None
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if teacher is not None and distill_alpha > 0:
                loss = (1.0 - distill_alpha) * loss + distill_alpha * _distillation_loss(
                    logits, teacher, distill_temperature
                )
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss /= max(n_train, 1)

        # One pass over validation produces both numbers, so selecting on AUPRC
        # costs no extra forward passes -- only the AUPRC computation itself,
        # which is negligible against a validation partition of 1-2 EDFs.
        val_loss, val_auprc = _eval_val(model, val_loader, criterion, device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auprc": val_auprc}
        )
        log.info(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_auprc={val_auprc:.4f}"
        )

        score = -val_loss if model_selection == "val_loss" else val_auprc
        # A validation partition with no positive window at all makes AUPRC nan
        # -- which happens on this data, since some folds hold a single seizure.
        # Falling back to loss for that epoch is better than treating nan as a
        # failure to improve and burning early-stopping patience on it.
        if model_selection == "val_auprc" and np.isnan(score):
            score = -val_loss

        if score > best_score:
            best_score = score
            best_val_loss = val_loss
            best_val_auprc = val_auprc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = early_stopping_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    # Always hand back the uncompiled module: callers save its state_dict.
    return TrainResult(
        model=unwrap_compiled(model),
        best_val_loss=best_val_loss,
        history=history,
        best_val_auprc=best_val_auprc,
        selection=model_selection,
    )


def _distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Hinton-style soft-target loss: KL(teacher || student) at temperature T.

    Why lever L3 exists at all: the student sees ONE EEG channel, and lever L5
    established that the missing information cannot be bought with more
    pre-training data. A teacher reading all 18 channels solves a materially
    easier problem, and its soft probabilities carry more than the hard label
    does -- how confident it is, and on which windows -- which is exactly the
    signal a single channel is short of.

    The `T**2` factor is from the original formulation: softening by T shrinks
    the gradients of the soft term by roughly 1/T**2, so without it the balance
    between the hard and soft terms would silently depend on the temperature.
    """
    t = temperature
    student_log_p = torch.log_softmax(student_logits / t, dim=1)
    teacher_p = torch.softmax(teacher_logits / t, dim=1)
    return torch.nn.functional.kl_div(
        student_log_p, teacher_p, reduction="batchmean"
    ) * (t * t)


def _eval_val(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str
) -> tuple[float, float]:
    """Validation cross-entropy and AUPRC from a single pass.

    AUPRC is over `softmax(logits)[:, 1]`, the ictal probability -- the same
    quantity `training/threshold_selection.py` later thresholds, so the metric
    the checkpoint is chosen by and the score the operating point is chosen on
    are the same signal.

    Returns nan for AUPRC when the partition holds only one class, which is a
    real case here: some folds have a single seizure, and a validation split
    can end up with no ictal window at all.
    """
    model.eval()
    total_loss, n = 0.0, 0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            probs.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

    val_loss = total_loss / max(n, 1)
    if not probs:
        return val_loss, float("nan")
    y_true = np.concatenate(labels)
    y_score = np.concatenate(probs)
    if y_true.min() == y_true.max():
        return val_loss, float("nan")
    return val_loss, float(compute_segment_metrics(y_true, y_score).auprc)
