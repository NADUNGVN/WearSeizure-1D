# Data protocol

Source: Research Decision Memo, sections 2.2 and 5. This document is the
implementation-facing summary; the memo is the source of truth.

## Four risks the original (Frontiers 2024) protocol had, and how this
## codebase avoids them

1. **Overlap leakage.** The original paper windows every 1/256s, so adjacent
   windows are nearly identical. Fix: split by **EDF file / seizure event**,
   never by window. `data/splits.py` operates only on `edf_id`s;
   `data/windowing.py` refuses (`ValueError`) to window an EDF that is not in
   the fold's allow-list for the requested partition.
2. **Test-tuned postprocessing.** Threshold `Th` and run-length `L` must
   never be chosen using test data. Fix: `training/threshold_selection.py`
   fits only on the validation partition and returns a `FrozenPostprocessParams`
   object that `evaluate.py`/`engine_baseline.py` load and apply as-is.
3. **FAR without exposure.** Accuracy on a balanced set says nothing about
   false alarms per hour of real, continuous, mostly-interictal recording.
   Fix: `eval/metrics_event.py` always reports `exposure_hours` (from actual
   EDF duration) alongside `far_per_hour`, and `eval/report.py` never reports
   FAR without it.
4. **Causal mismatch.** Offline zero-phase filtering / whole-file
   normalization cannot run in a streaming deployment. Fix:
   `signal/filters.py`'s `CausalBandpass` is one-pass IIR (`lfilter`, never
   `filtfilt`), state resets per EDF, and is the *only* filter used anywhere
   in training, evaluation, or the integer reference -- there is no separate
   "training filter" and "deployment filter".

## Split unit and order (memo 5.1)

1. Build the manifest (`data/manifest.py`), hash it.
2. Split by EDF/subject (`data/splits.py`) -- **before** any filtering,
   normalization, or windowing.
3. Fit filter state policy, normalization, and postprocess thresholds only on
   train/validation.
4. Personalized mode: leave-one-seizure-EDF-out per subject, each fold's test
   partition also gains one never-before-reused interictal EDF (increases
   exposure hours). Zero-shot mode: leave-one-subject-out, entire subject
   held out.
5. Continuous test runs causally: a window's decision is made at its
   `end_sec`, using only samples up to that point.

## Metrics (memo 5.2)

- **Event-level (primary)**: sensitivity, FAR/hour, detection delay. Always
  reported per-patient, macro-averaged, and micro-pooled, with 95% CIs
  (`eval/bootstrap.py`: exact binomial for sensitivity, Poisson for FAR,
  cluster bootstrap by patient for anything averaged).
- **Segment-level (secondary)**: AUPRC, AUROC, F1, balanced accuracy, always
  alongside class prevalence -- never accuracy alone.
- **Worst-patient gate**: no patient may fall below the Table 6 minimum in
  personalized mode without triggering failure analysis, not averaging it away.

## What "leakage-safe" is verified by

`tests/unit/test_splits_no_leakage.py` and `test_splits_loso.py` check, on a
synthetic cohort: no EDF appears in two partitions of the same fold; a
held-out zero-shot subject's EDFs never appear in that fold's train/val;
splits are deterministic given a seed; and `data/splits.py` raises on any
detected overlap (`validate_fold`).
