"""Confidence intervals (memo 5.3): exact binomial for pooled sensitivity,
Poisson for FAR, cluster bootstrap by patient for anything averaged across
patients (so that a high-seizure-count patient like chb15 cannot dominate a
resample the way it would dominate a plain pooled average).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.stats import beta, chi2


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lower, upper


def poisson_rate_ci(count: int, exposure_hours: float, alpha: float = 0.05) -> tuple[float, float]:
    if exposure_hours <= 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if count == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * count) / exposure_hours
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * count + 2) / exposure_hours
    return float(lower), float(upper)


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, Sequence[float]],
    statistic: Callable[[Sequence[float]], float] = np.mean,
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Resample clusters (patients) with replacement, then resample within
    each sampled cluster, pool, and compute the statistic -- repeated n_boot
    times to get a percentile CI.
    """
    rng = rng or np.random.default_rng(0)
    clusters = [c for c, vals in values_by_cluster.items() if len(vals) > 0]
    if not clusters:
        return (float("nan"), float("nan"))

    boot_stats = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        pooled: list[float] = []
        for c in sampled_clusters:
            vals = values_by_cluster[c]
            idx = rng.integers(0, len(vals), size=len(vals))
            pooled.extend(vals[i] for i in idx)
        if pooled:
            boot_stats.append(statistic(pooled))

    if not boot_stats:
        return (float("nan"), float("nan"))
    lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lower, upper


def paired_cluster_bootstrap(
    cluster_ids: Sequence[str],
    delta_statistic: Callable[[Sequence[str]], float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Percentile CI for a PAIRED difference between two configurations,
    resampling whole clusters (patients) with replacement.

    Why paired rather than two independent CIs: the two configurations are
    evaluated on exactly the same patients, folds and seizures, so most of the
    spread in either one is shared between them. Overlapping marginal CIs would
    therefore say almost nothing about whether the difference is real. What the
    paper's claim needs is an interval on the DIFFERENCE.

    Why the cluster is the patient and not the fold or the seizure: seizures
    from one patient are not independent draws -- CHB-MIT patients range from 3
    to 20+ seizures, so resampling seizures directly lets a single high-count
    patient dominate a replicate. `delta_statistic` receives the resampled list
    of cluster ids (with repeats) and must recompute A minus B from scratch on
    exactly those, which is what keeps the two sides paired within every
    replicate.

    `p_delta_le_0` is the bootstrap fraction of replicates in which the
    difference did not favour A. It is reported for orientation only -- the
    decision rule stated in docs/RESEARCH_REALITY_CHECK.md section 11.3 is
    whether the interval contains 0, not a p-value.
    """
    rng = rng or np.random.default_rng(0)
    clusters = list(cluster_ids)
    if not clusters:
        return {
            "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
            "p_delta_le_0": float("nan"), "n_clusters": 0, "n_boot": 0,
        }

    point = delta_statistic(clusters)
    deltas = []
    for _ in range(n_boot):
        sampled = [clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))]
        value = delta_statistic(sampled)
        if not np.isnan(value):
            deltas.append(value)

    if not deltas:
        return {
            "delta": point, "ci_low": float("nan"), "ci_high": float("nan"),
            "p_delta_le_0": float("nan"), "n_clusters": len(clusters), "n_boot": 0,
        }
    arr = np.asarray(deltas, dtype=float)
    return {
        "delta": float(point),
        "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        "p_delta_le_0": float((arr <= 0).mean()),
        "n_clusters": len(clusters),
        "n_boot": int(arr.size),
    }
