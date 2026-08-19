"""Path resolution helpers so no absolute machine-specific path is hardcoded.

Configs reference paths via Hydra profile groups (``configs/profile/*.yaml``),
which in turn interpolate environment variables for the server profile. This
module only centralizes the small amount of Python-side path logic (e.g.
resolving relative to the repo root) that configs cannot express on their own.
"""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


LEGACY_ARTIFACT_GLOBS = ("*.metrics.json", "*.pt")


def seeds_from_cfg(cfg) -> list[int]:
    """Which seeds a run covers.

    `train.seeds` was declared in `configs/train/default.yaml` from the start
    (memo 5.3 asks for three random seeds) but nothing ever read it --
    `scripts/train.py` only ever used `cfg.seed`. Every number in
    `docs/EXPERIMENT_LOG_G1a.md` is therefore a single-seed point estimate with
    no error bar, which is why the top three configurations (0.9218 / 0.9256 /
    0.9359, i.e. about one seizure out of 77) cannot be ranked against each
    other.

    Default stays `null` = the single `cfg.seed`, so existing runs reproduce
    exactly. Ask for error bars explicitly with `train.seeds=[0,1,2]`.
    """
    seeds = cfg.train.get("seeds")
    if seeds is None:
        return [int(cfg.seed)]
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("train.seeds is an empty list; use null for a single-seed run")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"train.seeds has duplicates: {seeds}")
    return seeds


def fold_run_dir(artifacts_dir: str | Path, model_name: str, split_name: str,
                 window_name: str, seed: int) -> Path:
    """Where one (model, split, window, seed) run keeps its checkpoints,
    per-fold metrics and report.

    The `seed<N>` level is what makes multi-seed runs possible without folds
    from different seeds overwriting each other's `<fold_id>.metrics.json` --
    the same reason `window.name` is already part of the path (commit 5564f9f).
    Artifacts produced before this level existed sit one directory up; see
    `warn_if_legacy_artifacts`.
    """
    return Path(artifacts_dir) / model_name / split_name / window_name / f"seed{seed}"


def pretrain_cache_dir(artifacts_dir: str | Path, model_name: str,
                       window_name: str, seed: int) -> Path:
    """Cohort pre-training inits are keyed by seed too: `cohort_pretrain_fold`
    draws the subject-level validation split from `rng_for(..., base_seed=seed)`,
    so two seeds genuinely produce different initialisations and must not share
    a cache entry.
    """
    return Path(artifacts_dir) / "pretrain" / model_name / window_name / f"seed{seed}"


def warn_if_legacy_artifacts(run_dir: Path, log) -> None:
    """Say so loudly when a pre-`seed<N>` run is sitting one level up.

    Without this the symptom is silent and expensive: `train.py` finds no
    `<fold_id>.metrics.json` in the new directory and cheerfully re-trains all
    66 folds, while the checkpoints that produced rows 21-26 sit untouched in
    the parent directory.
    """
    # `any(dir.glob(g) for g in ...)` would be a no-op: a generator object is
    # always truthy, so the guard would pass whether or not anything matched.
    if any(next(run_dir.glob(g), None) is not None for g in LEGACY_ARTIFACT_GLOBS):
        return
    legacy = run_dir.parent
    stale = [p.name for g in LEGACY_ARTIFACT_GLOBS for p in legacy.glob(g)]
    if not stale:
        return
    log.warning(
        f"{len(stale)} artifact(s) from before the seed<N> directory level are in {legacy} "
        f"while this run reads {run_dir}. They will be ignored and every fold re-trained. "
        f"To adopt them as seed {run_dir.name.removeprefix('seed')}, run once:\n"
        f"  mkdir -p '{run_dir}' && mv '{legacy}'/*.pt '{legacy}'/*.json '{run_dir}'/"
    )
