"""Leakage-safe splitting -- the load-bearing module of the whole protocol.

Memo 5.1: split unit is the **EDF file** (never a window), decided *before*
any filtering/normalization/windowing happens. Two strategies:

- `make_patient_specific_loso_edf`: personalized / primary mode. For each
  subject, one fold per ictal EDF: that EDF plus one never-before-reused
  interictal EDF from the same subject go to test; the rest of that subject's
  EDFs split into train/val.
- `make_zero_shot_loso_subject`: Q1 zero-shot mode. One fold per subject: the
  *entire* subject (every EDF) is held out for test; remaining subjects split
  into train/val at the subject level.

Both return `Fold` objects carrying `manifest_hash`, so a fold generated
against an old manifest can never silently be reused against a new one.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd

from wearseizure.data.manifest import events_for_row, hash_manifest
from wearseizure.utils.seeding import rng_for


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_edf_ids: frozenset[str]
    val_edf_ids: frozenset[str]
    test_edf_ids: frozenset[str]
    held_out_key: str
    manifest_hash: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_edf_ids"] = sorted(self.train_edf_ids)
        d["val_edf_ids"] = sorted(self.val_edf_ids)
        d["test_edf_ids"] = sorted(self.test_edf_ids)
        return d

    @staticmethod
    def from_dict(d: dict) -> Fold:
        return Fold(
            fold_id=d["fold_id"],
            train_edf_ids=frozenset(d["train_edf_ids"]),
            val_edf_ids=frozenset(d["val_edf_ids"]),
            test_edf_ids=frozenset(d["test_edf_ids"]),
            held_out_key=d["held_out_key"],
            manifest_hash=d["manifest_hash"],
        )


def validate_fold(fold: Fold) -> None:
    """Raise if any EDF appears in more than one partition of this fold."""
    pairs = [
        ("train", "val", fold.train_edf_ids & fold.val_edf_ids),
        ("train", "test", fold.train_edf_ids & fold.test_edf_ids),
        ("val", "test", fold.val_edf_ids & fold.test_edf_ids),
    ]
    for a, b, overlap in pairs:
        if overlap:
            raise ValueError(f"fold {fold.fold_id}: {a}/{b} overlap on edf_id(s) {sorted(overlap)}")


def _has_events(row: pd.Series) -> bool:
    return len(events_for_row(row)) > 0


def make_patient_specific_loso_edf(
    manifest_df: pd.DataFrame,
    seed: int,
    val_fraction: float = 0.2,
) -> list[Fold]:
    manifest_hash = hash_manifest(manifest_df)
    folds: list[Fold] = []

    for subject_id, group in manifest_df.groupby("subject_id"):
        group = group.reset_index(drop=True)
        edf_ids = sorted(group["edf_id"])
        ictal_edf_ids = sorted(row["edf_id"] for _, row in group.iterrows() if _has_events(row))
        interictal_edf_ids = sorted(e for e in edf_ids if e not in ictal_edf_ids)
        if not ictal_edf_ids:
            continue  # nothing to leave out for personalized evaluation

        used_interictal: set[str] = set()
        for ictal_edf in ictal_edf_ids:
            candidates = [e for e in interictal_edf_ids if e not in used_interictal] or interictal_edf_ids
            test_ids = {ictal_edf}
            held_out_interictal = None
            if candidates:
                pick_rng = rng_for(subject_id, ictal_edf, "held_out_interictal", base_seed=seed)
                held_out_interictal = candidates[int(pick_rng.integers(0, len(candidates)))]
                used_interictal.add(held_out_interictal)
                test_ids.add(held_out_interictal)

            remaining = sorted(e for e in edf_ids if e not in test_ids)
            val_rng = rng_for(subject_id, ictal_edf, "val_split", base_seed=seed)
            n_val = min(len(remaining), max(1, round(len(remaining) * val_fraction))) if len(remaining) > 1 else 0
            val_idx = (
                val_rng.choice(len(remaining), size=n_val, replace=False) if n_val > 0 else []
            )
            val_ids = {remaining[i] for i in val_idx}
            train_ids = {e for e in remaining if e not in val_ids}

            fold = Fold(
                fold_id=f"{subject_id}__{ictal_edf}",
                train_edf_ids=frozenset(train_ids),
                val_edf_ids=frozenset(val_ids),
                test_edf_ids=frozenset(test_ids),
                held_out_key=ictal_edf,
                manifest_hash=manifest_hash,
            )
            validate_fold(fold)
            folds.append(fold)

    return folds


def make_zero_shot_loso_subject(
    manifest_df: pd.DataFrame,
    seed: int,
    val_fraction: float = 0.2,
) -> list[Fold]:
    manifest_hash = hash_manifest(manifest_df)
    subjects = sorted(manifest_df["subject_id"].unique())
    if len(subjects) < 3:
        raise ValueError("make_zero_shot_loso_subject requires at least 3 subjects (1 test + train/val)")
    folds: list[Fold] = []

    for held_out_subject in subjects:
        test_ids = frozenset(manifest_df.loc[manifest_df["subject_id"] == held_out_subject, "edf_id"])
        remaining_subjects = [s for s in subjects if s != held_out_subject]

        val_rng = rng_for(held_out_subject, "val_subjects", base_seed=seed)
        n_val_subjects = max(1, round(len(remaining_subjects) * val_fraction))
        n_val_subjects = min(n_val_subjects, len(remaining_subjects) - 1) if len(remaining_subjects) > 1 else 0
        val_idx = val_rng.choice(len(remaining_subjects), size=n_val_subjects, replace=False) if n_val_subjects > 0 else []
        val_subjects = {remaining_subjects[i] for i in val_idx}
        train_subjects = [s for s in remaining_subjects if s not in val_subjects]

        val_ids = frozenset(manifest_df.loc[manifest_df["subject_id"].isin(val_subjects), "edf_id"])
        train_ids = frozenset(manifest_df.loc[manifest_df["subject_id"].isin(train_subjects), "edf_id"])

        fold = Fold(
            fold_id=f"loso__{held_out_subject}",
            train_edf_ids=train_ids,
            val_edf_ids=val_ids,
            test_edf_ids=test_ids,
            held_out_key=held_out_subject,
            manifest_hash=manifest_hash,
        )
        validate_fold(fold)
        folds.append(fold)

    return folds


def save_folds(folds: list[Fold], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([fold.to_dict() for fold in folds], f, indent=2, sort_keys=True)


def load_folds(path: str, expected_manifest_hash: str | None = None) -> list[Fold]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    folds = [Fold.from_dict(d) for d in raw]
    if expected_manifest_hash is not None:
        for fold in folds:
            if fold.manifest_hash != expected_manifest_hash:
                raise ValueError(
                    f"fold {fold.fold_id} was generated against manifest_hash "
                    f"{fold.manifest_hash}, but the current manifest hashes to "
                    f"{expected_manifest_hash}. Regenerate splits with make_splits.py."
                )
    return folds


def subject_from_fold_id(fold_id: str, strategy: str) -> str:
    """Recover the patient/subject id a fold belongs to from its `fold_id`
    (`"{subject}__{held_out_key}"` for patient-specific, `"loso__{subject}"`
    for zero-shot) -- used to group folds by patient for aggregation
    (scripts/evaluate.py) and for pooled threshold selection
    (scripts/rethreshold_pooled.py).
    """
    prefix, _, rest = fold_id.partition("__")
    return rest if strategy == "zero_shot_loso_subject" else prefix
