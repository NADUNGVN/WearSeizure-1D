"""The profile/data pairing guard.

`configs/config.yaml` defaults to `data: synthetic` and no profile group
changes it, so `profile=server` on its own silently keeps the synthetic
dataset config while resolving its paths against the clinical directories.
This is the failure the guard exists to make loud.
"""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from wearseizure.utils.profile_guard import check_profile_data_pairing


def _cfg(profile_name: str, data_name: str):
    return OmegaConf.create(
        {
            "profile": {"name": profile_name},
            "data": {"name": data_name, "manifest_path": "/artifacts/manifest/x.csv"},
        }
    )


def test_server_profile_with_synthetic_data_is_refused():
    with pytest.raises(ValueError, match="data=chbmit"):
        check_profile_data_pairing(_cfg("server", "synthetic"))


def test_local_profile_with_clinical_data_is_refused():
    with pytest.raises(ValueError, match="local_synthetic"):
        check_profile_data_pairing(_cfg("local_synthetic", "chbmit"))


def test_the_two_valid_pairings_pass():
    check_profile_data_pairing(_cfg("server", "chbmit"))
    check_profile_data_pairing(_cfg("local_synthetic", "synthetic"))


def test_error_names_the_manifest_it_would_have_looked_for():
    # The message has to be actionable at 2am on a server, not just correct.
    with pytest.raises(ValueError) as excinfo:
        check_profile_data_pairing(_cfg("server", "synthetic"))
    assert "/artifacts/manifest/x.csv" in str(excinfo.value)


def test_unknown_profile_names_are_left_alone():
    # The guard should not become a whitelist of profile names; a new profile
    # group must not start failing just because this file has not heard of it.
    check_profile_data_pairing(_cfg("some_future_profile", "chbmit"))
    check_profile_data_pairing(_cfg("some_future_profile", "synthetic"))
