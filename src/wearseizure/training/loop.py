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


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str = "cpu",
    early_stopping_patience: int = 8,
) -> TrainResult:
    model.to(device)
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
            x, y = x.to(device), y.to(device)
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
    return TrainResult(model=model, best_val_loss=best_val_loss, history=history)


def _eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)
