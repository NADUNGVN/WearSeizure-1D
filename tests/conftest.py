from __future__ import annotations

import numpy as np
import pytest

from wearseizure.data.synthetic import generate_synthetic_cohort


@pytest.fixture(scope="session")
def synthetic_cohort():
    """Small, fast, seeded synthetic cohort shared by unit/integration tests.

    5 subjects x 4 EDFs x 120s is enough to exercise both split strategies
    (needs >=3 subjects for zero-shot LOSO, >=1 ictal EDF per subject for
    patient-specific) while keeping the whole test suite fast on CPU.
    """
    manifest_df, records = generate_synthetic_cohort(
        n_subjects=5,
        edfs_per_subject=4,
        seed=0,
        edf_duration_s=120.0,
    )
    return manifest_df, records


@pytest.fixture()
def rng():
    return np.random.default_rng(0)
