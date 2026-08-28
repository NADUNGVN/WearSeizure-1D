"""Per-fold orchestration: build datasets -> train -> freeze threshold on val
-> evaluate on continuous test. Identical for every model architecture (only
`model` differs), so `engine_wearseizure.py` re-exports `run_fold` rather than
duplicating this procedure for WearSeizure-1D.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from wearseizure.data.dataset import WearSeizureWindowDataset, build_fold_datasets
from wearseizure.data.records import EEGRecord
from wearseizure.data.sampler import make_class_balanced_sampler
from wearseizure.data.splits import Fold
from wearseizure.eval.metrics_event import EventMetrics, compute_event_metrics
from wearseizure.eval.metrics_segment import SegmentMetrics, compute_segment_metrics
from wearseizure.postprocess.pipeline import run_postprocess
from wearseizure.training.loop import train_classifier
from wearseizure.training.threshold_selection import FrozenPostprocessParams, fit_threshold_on_val
from wearseizure.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class FoldResult:
    fold_id: str
    model: torch.nn.Module
    frozen_postprocess: FrozenPostprocessParams
    test_event_metrics: EventMetrics
    test_segment_metrics: SegmentMetrics


def _dataloader_kwargs(device: str, num_workers: int, persistent: bool = True) -> dict:
    # pin_memory only helps (and is only supported) when copying host->CUDA.
    # persistent_workers avoids re-spawning the worker pool every epoch for a
    # loader that's iterated many times (train/val across epochs) -- but for
    # a one-shot loader (scoring a partition exactly once per fold), workers
    # spawned with persistent_workers=True stick around until the DataLoader
    # object is garbage-collected, and creating a fresh loader per fold x66
    # folds x2 partitions was outliving GC and exhausting the OS's open-file
    # limit ("Too many open files"). Only ever request persistent=True for a
    # loader that will actually be reused across multiple iterations.
    return {
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
        "persistent_workers": persistent and num_workers > 0,
    }


def score_partition(
    model: torch.nn.Module,
    dataset: WearSeizureWindowDataset,
    device: str,
    batch_size: int = 128,
    num_workers: int = 0,
):
    # Always num_workers=0 here regardless of what's passed: this loader is
    # iterated exactly once (a single scoring pass, not repeated across
    # epochs), so multiprocessing workers buy nothing but still spawn OS-level
    # locks/semaphores (and, in file_system sharing mode, a temp directory)
    # per DataLoader construction. Across ~66 folds x 2 scoring calls each,
    # that accumulation alone exhausted the OS open-file limit even after
    # switching to the file_system sharing strategy -- this is a second,
    # independent source of the same "Too many open files" failure mode.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        **_dataloader_kwargs(device, num_workers=0, persistent=False),
    )
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1]
            scores.append(probs.cpu().numpy())
            labels.append(y.numpy())
    scores = np.concatenate(scores) if scores else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    end_sec = np.array([w.end_sec for w in dataset.windows])
    edf_ids = np.array([w.edf_id for w in dataset.windows])
    return scores, labels, end_sec, edf_ids


def group_by_edf(end_sec, scores, edf_ids, records: dict[str, EEGRecord], allowed_edf_ids):
    end_sec_by_edf, scores_by_edf, events_by_edf, exposure_by_edf = {}, {}, {}, {}
    for edf_id in allowed_edf_ids:
        mask = edf_ids == edf_id
        order = np.argsort(end_sec[mask])
        end_sec_by_edf[edf_id] = end_sec[mask][order]
        scores_by_edf[edf_id] = scores[mask][order]
        record = records[edf_id]
        events_by_edf[edf_id] = [(e.event_id, e.onset_sec, e.offset_sec) for e in record.meta.seizure_events]
        exposure_by_edf[edf_id] = record.meta.duration_sec / 3600.0
    return end_sec_by_edf, scores_by_edf, events_by_edf, exposure_by_edf


def evaluate_fold(
    model: torch.nn.Module,
    records: dict[str, EEGRecord],
    fold: Fold,
    window_s: float,
    stride_s: float,
    postprocess_method: str,
    postprocess_ema_alpha: float,
    postprocess_run_length: int,
    postprocess_event_merge_gap_s: float,
    threshold_on_grid: list[float],
    threshold_off_grid: list[float],
    batch_size: int,
    device: str = "cpu",
    num_workers: int = 0,
    far_weight: float = 0.1,
    far_cap_per_hour: float | None = None,
    objective: str = "max_sensitivity",
    sensitivity_floor: float | None = None,
    postprocess_alarm_timestamp: str = "window_end",
    compile_mode: str | None = None,
) -> FoldResult:
    """Threshold-freeze on val + evaluate on test for an already-trained
    `model` -- no training happens here. Used by `run_fold` right after
    training, and directly by `scripts/rethreshold.py` to re-run
    postprocessing against a saved checkpoint without re-training (e.g. after
    widening the threshold search grid).
    """
    datasets, _band, _normalizer = build_fold_datasets(records, fold, window_s, stride_s)
    val_ds, test_ds = datasets["val"], datasets["test"]
    model.to(device)

    val_scores, _val_labels, val_end_sec, val_edf_ids = score_partition(
        model, val_ds, device, batch_size=batch_size, num_workers=num_workers
    )
    end_sec_by_edf, scores_by_edf, events_by_edf, exposure_by_edf = group_by_edf(
        val_end_sec, val_scores, val_edf_ids, records, fold.val_edf_ids
    )
    frozen = fit_threshold_on_val(
        end_sec_by_edf, scores_by_edf, events_by_edf, exposure_by_edf,
        method=postprocess_method, ema_alpha=postprocess_ema_alpha, run_length=postprocess_run_length,
        event_merge_gap_s=postprocess_event_merge_gap_s, threshold_on_grid=threshold_on_grid,
        threshold_off_grid=threshold_off_grid, fold_id=fold.fold_id,
        far_weight=far_weight, far_cap_per_hour=far_cap_per_hour,
        objective=objective, sensitivity_floor=sensitivity_floor,
        window_s=window_s, alarm_timestamp=postprocess_alarm_timestamp,
    )

    test_scores, test_labels, test_end_sec, test_edf_ids = score_partition(
        model, test_ds, device, batch_size=batch_size, num_workers=num_workers
    )
    test_end_sec_by_edf, test_scores_by_edf, test_events_by_edf, test_exposure_by_edf = group_by_edf(
        test_end_sec, test_scores, test_edf_ids, records, fold.test_edf_ids
    )

    all_events, delays = [], []
    total_matched = total_false_alarms = 0
    total_exposure = 0.0
    for edf_id in fold.test_edf_ids:
        alarms = run_postprocess(test_end_sec_by_edf[edf_id], test_scores_by_edf[edf_id], frozen.params)
        edf_events = test_events_by_edf[edf_id]
        m = compute_event_metrics(edf_events, alarms, test_exposure_by_edf[edf_id])
        total_matched += m.n_matched
        total_false_alarms += m.n_false_alarms
        total_exposure += m.exposure_hours
        delays.extend(m.delays_s)
        all_events.extend(edf_events)

    n_events = len(all_events)
    test_event_metrics = EventMetrics(
        n_events=n_events,
        n_matched=total_matched,
        n_missed=n_events - total_matched,
        n_false_alarms=total_false_alarms,
        sensitivity=(total_matched / n_events if n_events else float("nan")),
        far_per_hour=(total_false_alarms / total_exposure if total_exposure else float("nan")),
        delays_s=delays,
        exposure_hours=total_exposure,
    )
    test_segment_metrics = compute_segment_metrics(test_labels, test_scores)

    return FoldResult(
        fold_id=fold.fold_id,
        model=model,
        frozen_postprocess=frozen,
        test_event_metrics=test_event_metrics,
        test_segment_metrics=test_segment_metrics,
    )


def run_fold(
    model: torch.nn.Module,
    records: dict[str, EEGRecord],
    fold: Fold,
    window_s: float,
    stride_s: float,
    postprocess_method: str,
    postprocess_ema_alpha: float,
    postprocess_run_length: int,
    postprocess_event_merge_gap_s: float,
    threshold_on_grid: list[float],
    threshold_off_grid: list[float],
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str = "cpu",
    class_balanced_sampling: bool = True,
    early_stopping_patience: int = 8,
    num_workers: int = 0,
    far_weight: float = 0.1,
    far_cap_per_hour: float | None = None,
    objective: str = "max_sensitivity",
    sensitivity_floor: float | None = None,
    postprocess_alarm_timestamp: str = "window_end",
    compile_mode: str | None = None,
    model_selection: str = "val_loss",
    teacher_logits: np.ndarray | None = None,
    distill_alpha: float = 0.0,
    distill_temperature: float = 2.0,
) -> FoldResult:
    datasets, _band, _normalizer = build_fold_datasets(
        records, fold, window_s, stride_s, teacher_logits=teacher_logits
    )
    train_ds, val_ds = datasets["train"], datasets["val"]
    dl_kwargs = _dataloader_kwargs(device, num_workers)

    if class_balanced_sampling:
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=make_class_balanced_sampler(train_ds), **dl_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **dl_kwargs)

    train_result = train_classifier(
        model, train_loader, val_loader, epochs=epochs, lr=lr, weight_decay=weight_decay,
        device=device, early_stopping_patience=early_stopping_patience,
        compile_mode=compile_mode, model_selection=model_selection,
        distill_alpha=distill_alpha, distill_temperature=distill_temperature,
    )

    return evaluate_fold(
        model=train_result.model, records=records, fold=fold, window_s=window_s, stride_s=stride_s,
        postprocess_method=postprocess_method, postprocess_ema_alpha=postprocess_ema_alpha,
        postprocess_run_length=postprocess_run_length, postprocess_event_merge_gap_s=postprocess_event_merge_gap_s,
        threshold_on_grid=threshold_on_grid, threshold_off_grid=threshold_off_grid,
        batch_size=batch_size, device=device, num_workers=num_workers,
        far_weight=far_weight, far_cap_per_hour=far_cap_per_hour,
        objective=objective, sensitivity_floor=sensitivity_floor,
        postprocess_alarm_timestamp=postprocess_alarm_timestamp,
    )
