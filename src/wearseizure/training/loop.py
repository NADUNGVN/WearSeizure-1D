"""Generic binary-classifier training loop with early stopping on val loss.
Shared by every model (baseline or WearSeizure-1D) and every precision mode
(the QAT wrapper in `quant/qat.py` reuses this by passing a fake-quantized
model in as `model`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from wearseizure.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class TrainResult:
    model: nn.Module
    best_val_loss: float
    history: list[dict] = field(default_factory=list)


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
    """
    model.to(device)
    if compile_mode:
        log.info(f"torch.compile(mode={compile_mode!r}) enabled")
        model = torch.compile(model, mode=compile_mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_left = early_stopping_patience
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        train_loss, n_train = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        train_loss /= max(n_train, 1)

        val_loss = _eval_loss(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        log.info(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
    return TrainResult(model=unwrap_compiled(model), best_val_loss=best_val_loss, history=history)


def _eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)
