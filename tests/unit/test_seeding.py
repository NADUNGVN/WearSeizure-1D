from __future__ import annotations

from wearseizure.utils.seeding import rng_for


def test_rng_for_is_deterministic_given_same_inputs():
    a = rng_for("subject_x", "edf_y", base_seed=42).standard_normal(10)
    b = rng_for("subject_x", "edf_y", base_seed=42).standard_normal(10)
    assert (a == b).all()


def test_rng_for_differs_across_base_seed():
    a = rng_for("subject_x", "edf_y", base_seed=0).integers(0, 1_000_000, size=20)
    b = rng_for("subject_x", "edf_y", base_seed=1).integers(0, 1_000_000, size=20)
    assert list(a) != list(b)


def test_rng_for_differs_across_identifying_parts():
    a = rng_for("subject_x", base_seed=0).integers(0, 1_000_000, size=20)
    b = rng_for("subject_y", base_seed=0).integers(0, 1_000_000, size=20)
    assert list(a) != list(b)
