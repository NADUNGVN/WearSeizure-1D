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
