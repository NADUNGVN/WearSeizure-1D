"""The published evaluation protocol, reproduced deliberately (item A7).

Why this exists
---------------
`baseline_frontiers2d` reproduces the Chung 2024 architecture and reaches 0.8811
under this project's leakage-safe protocol. The paper reports 0.9962. The
project wants to attribute that gap to the PROTOCOL -- but so far that is an
inference, not a measurement: a reviewer can reasonably answer "or your
reproduction is simply worse". Only running the same code on the same data under
their protocol, and arriving near their number, rules that out.

Four factors, not one
---------------------
Reporting a single "leaky" number would repeat the mistake it exists to expose.
The published figure differs from ours in at least four independent ways, and
each one is switchable here so the gap can be decomposed:

1. `random_window_split` -- windows pooled and split at RANDOM instead of by
   recording. With 4s windows at 1s stride, adjacent windows share 75% of their
   samples, so a test window's near-duplicate is almost always in train. This is
   the leak proper.
2. `global_normalisation` -- the affine normaliser fitted on ALL data before
   splitting, so test statistics inform the training inputs.
3. `threshold_on_test` -- the decision threshold chosen on the evaluation data
   rather than frozen on a held-out validation partition.
4. `segment_metric` -- sensitivity counted per WINDOW rather than per seizure
   EVENT.

The fourth is not a leak at all, and it is the one that matters most for
honesty. Under a random window split an event's windows are distributed across
train and test, so event-level detection is not even DEFINABLE -- which means
the published 99.62% is necessarily a window-level number and is not comparable
to an event sensitivity by construction. Putting the two side by side without
saying so would be the same category error this project exists to document.

The ladder, one factor per rung:

    A  random split + global norm + threshold on test + segment   (as published)
    B  recording split + global norm + threshold on test + segment
    C  recording split + train-only norm + threshold on val + segment
    D  recording split + train-only norm + threshold on val + event   (ours)

A->B is the window-split leak, B->C is the fitting leak, C->D is the metric
definition. Only A->B is dishonesty; C->D is a difference in what is being
counted, and it belongs in the figure with that label.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from wearseizure.data.dataset import WearSeizureWindowDataset, _filter_records
from wearseizure.data.records import EEGRecord
from wearseizure.signal.filters import CausalBandpass
from wearseizure.signal.normalize import fit_affine_normalizer


@dataclass(frozen=True)
class ProtocolConfig:
    """One rung of the ladder. The defaults are this project's own protocol."""

    name: str
    random_window_split: bool = False
    global_normalisation: bool = False
    threshold_on_test: bool = False
    segment_metric: bool = False

    def describe(self) -> str:
        parts = [
            "random-window split" if self.random_window_split else "split by recording",
            "normalised on everything" if self.global_normalisation else "normalised on train only",
            "threshold on test" if self.threshold_on_test else "threshold frozen on val",
            "per-window sensitivity" if self.segment_metric else "per-event sensitivity",
        ]
        return f"{self.name}: " + ", ".join(parts)


LADDER = (
    ProtocolConfig("A_as_published", True, True, True, True),
    ProtocolConfig("B_split_by_recording", False, True, True, True),
    ProtocolConfig("C_no_fitting_leak", False, False, False, True),
    ProtocolConfig("D_ours_event_level", False, False, False, False),
)


def prepare_records(
    records: dict[str, EEGRecord],
    edf_ids: frozenset[str],
    train_edf_ids: frozenset[str],
    *,
    global_normalisation: bool,
    band_low_hz: float = 1.0,
    band_high_hz: float = 30.0,
) -> dict[str, EEGRecord]:
    """Filter, then normalise on either everything or the train split alone.

    The bandpass stays causal and per-recording state-reset in both arms. That
    is a property of the filter rather than of the protocol, and relaxing it
    here would confound this comparison with a different one.
    """
    band = CausalBandpass(low_hz=band_low_hz, high_hz=band_high_hz)
    filtered = _filter_records(records, sorted(edf_ids), band)
    fit_on = sorted(edf_ids) if global_normalisation else sorted(train_edf_ids)
    if not fit_on:
        raise ValueError("nothing to fit the normaliser on")
    normalizer = fit_affine_normalizer([filtered[eid].signal for eid in fit_on])
    return {
        eid: replace(r, signal=normalizer.apply(r.signal).astype(np.float32, copy=False))
        for eid, r in filtered.items()
    }


def split_window_indices(
    dataset: WearSeizureWindowDataset,
    *,
    random_window_split: bool,
    test_edf_ids: frozenset[str],
    val_fraction: float,
    seed: int,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train / val / test positions into `dataset.windows`.

    Under the random split every recording contributes to every partition --
    that is the leak. Under the recording split the partitions are disjoint by
    recording, which is this project's protocol, expressed over the SAME window
    list so that nothing but the partitioning differs between the two arms.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset.windows)
    if n == 0:
        raise ValueError("no windows to split")

    if random_window_split:
        order = rng.permutation(n)
        n_test = round(n * test_fraction)
        n_val = round(n * val_fraction)
        if n_test + n_val >= n:
            raise ValueError(f"test+val fractions leave no training windows ({n} windows)")
        return order[n_test + n_val:], order[n_test:n_test + n_val], order[:n_test]

    is_test = np.fromiter((w.edf_id in test_edf_ids for w in dataset.windows), bool, n)
    test_idx = np.flatnonzero(is_test)
    rest = np.flatnonzero(~is_test)
    if test_idx.size == 0 or rest.size == 0:
        raise ValueError("the recording split left a partition empty")

    # Validation is carved out BY RECORDING too. Taking it at random here would
    # reintroduce through the val split exactly the leak this rung removes.
    rest_edfs = sorted({dataset.windows[i].edf_id for i in rest})
    n_val_edfs = min(len(rest_edfs) - 1, max(1, round(len(rest_edfs) * val_fraction)))
    val_edfs = set(np.array(rest_edfs)[rng.permutation(len(rest_edfs))[:n_val_edfs]].tolist())
    in_val = np.fromiter((dataset.windows[i].edf_id in val_edfs for i in rest), bool, rest.size)
    return rest[~in_val], rest[in_val], test_idx


def near_duplicate_fraction(
    dataset: WearSeizureWindowDataset,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> float:
    """Share of test windows that share samples with some training window.

    This is what makes the leak concrete rather than rhetorical. At 4s windows
    and 1s stride a window overlaps its neighbour by 75%, so under a random
    split nearly every test window has a training window covering most of it.
    Report it beside the accuracy: a 99% accuracy next to a 99% overlap fraction
    explains itself.
    """
    if len(test_idx) == 0:
        return float("nan")

    spans_by_edf: dict[str, list[tuple[int, int]]] = {}
    for i in train_idx:
        w = dataset.windows[i]
        spans_by_edf.setdefault(w.edf_id, []).append((w.start_idx, w.end_idx))
    for spans in spans_by_edf.values():
        spans.sort()

    overlapping = 0
    for i in test_idx:
        w = dataset.windows[i]
        spans = spans_by_edf.get(w.edf_id)
        if not spans:
            continue
        # Rightmost training span that starts before this window ends; if it
        # also ends after this window starts, the two share samples.
        lo, hi = 0, len(spans)
        while lo < hi:
            mid = (lo + hi) // 2
            if spans[mid][0] < w.end_idx:
                lo = mid + 1
            else:
                hi = mid
        if lo and spans[lo - 1][1] > w.start_idx:
            overlapping += 1
    return overlapping / len(test_idx)
