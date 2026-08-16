"""Refuse profile/data combinations that are always a mistake.

Hydra's `profile` and `data` groups are independent, and `configs/config.yaml`
defaults to `data: synthetic`. So `profile=server` on its own does **not**
select the clinical dataset -- `scripts/server_bootstrap.sh` has always had to
pass `data=chbmit` explicitly, while README step 4 says only "all with
`profile=server`".

Getting that wrong is not a loud failure in general. With
`profile=server data=synthetic`, `configs/data/synthetic.yaml` resolves
`generated_dir: ${profile.data_root}` to `$CHBMIT_RAW_DIR`, i.e. the synthetic
loader is pointed straight at the clinical recordings, and the manifest path
becomes `$WEARSEIZURE_ARTIFACTS_DIR/manifest/synthetic_manifest.csv`. Depending
on what happens to exist on disk that either dies with a confusing
FileNotFoundError or, worse, quietly reads real data through the synthetic
path.

Neither direction of the mismatch is ever intended, so both are refused here
with a message that names the fix.
"""
from __future__ import annotations


def check_profile_data_pairing(cfg) -> None:
    """Raise if the profile and data groups disagree about real vs synthetic."""
    profile_name = cfg.profile.get("name")
    data_name = cfg.data.get("name")

    if profile_name == "server" and data_name == "synthetic":
        raise ValueError(
            "profile=server selects the GPU/clinical-path profile but data=synthetic is "
            "still the default dataset group, so this run would look for "
            f"{cfg.data.manifest_path!r} and point the synthetic loader at the real "
            "recordings.\n"
            "Add the data group explicitly:  python <script>.py profile=server data=chbmit ..."
        )

    if profile_name == "local_synthetic" and data_name != "synthetic":
        raise ValueError(
            f"profile=local_synthetic has no clinical data available, but data={data_name!r} "
            "was requested. Use profile=server data=chbmit on the training server, or "
            "profile=local_synthetic data=synthetic here."
        )
