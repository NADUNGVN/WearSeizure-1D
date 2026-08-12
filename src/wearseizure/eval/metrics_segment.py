"""Segment-level metrics (memo 5.2): explicitly secondary. Never report
accuracy alone -- always alongside class prevalence, per memo 5.2 "Segment"
row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class SegmentMetrics:
    auprc: float
    auroc: float
    f1: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    prevalence: float


def compute_segment_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> SegmentMetrics:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    has_both_classes = len(set(labels.tolist())) > 1
    auprc = float(average_precision_score(labels, scores)) if has_both_classes else float("nan")
    auroc = float(roc_auc_score(labels, scores)) if has_both_classes else float("nan")

    return SegmentMetrics(
        auprc=auprc,
        auroc=auroc,
        f1=float(f1_score(labels, preds, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(labels, preds)),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        prevalence=float(np.mean(labels)),
    )
