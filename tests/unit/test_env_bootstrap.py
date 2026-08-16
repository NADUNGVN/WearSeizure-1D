"""`.env` loading and the early failure for `profile=server`.

`.env.example` told users to copy it to `.env`, and the README pointed at
`.env` as the way to set the server's paths -- but nothing ever read that file.
When the variables are unset, `configs/profile/server.yaml` fails inside
`${oc.env:...}` while Hydra resolves `hydra.run.dir`, i.e. before any script
code runs, producing ~200 lines of ANTLR traceback per process.
"""
from __future__ import annotations

import os

import pytest

from wearseizure.utils.env import load_env_file, require_server_env


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("CHBMIT_RAW_DIR", "WEARSEIZURE_ARTIFACTS_DIR", "SOME_OTHER_VAR"):
        monkeypatch.delenv(name, raising=False)


def _write(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_simple_key_values(tmp_path):
    path = _write(tmp_path, "CHBMIT_RAW_DIR=/data/chbmit\nWEARSEIZURE_ARTIFACTS_DIR=/data/art\n")
    assert set(load_env_file(path)) == {"CHBMIT_RAW_DIR", "WEARSEIZURE_ARTIFACTS_DIR"}
    assert os.environ["CHBMIT_RAW_DIR"] == "/data/chbmit"


def test_an_existing_environment_variable_always_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("CHBMIT_RAW_DIR", "/exported/by/hand")
    path = _write(tmp_path, "CHBMIT_RAW_DIR=/from/dotenv\n")
    assert load_env_file(path) == []
    assert os.environ["CHBMIT_RAW_DIR"] == "/exported/by/hand"


def test_expands_tilde(tmp_path):
    # server_bootstrap.sh documents paths as ~/Manh/...; a shell expands that on
    # export, a .env file does not.
    path = _write(tmp_path, "CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0\n")
    load_env_file(path)
    assert "~" not in os.environ["CHBMIT_RAW_DIR"]
    assert os.environ["CHBMIT_RAW_DIR"].endswith("Manh/datasets/CHB-MIT/1.0.0")


def test_ignores_comments_blanks_and_strips_quotes_and_export(tmp_path):
    path = _write(
        tmp_path,
        "# a comment\n\n"
        'CHBMIT_RAW_DIR="/quoted/path"\n'
        "export WEARSEIZURE_ARTIFACTS_DIR='/single/quoted'\n"
        "not a key value line\n",
    )
    load_env_file(path)
    assert os.environ["CHBMIT_RAW_DIR"] == "/quoted/path"
    assert os.environ["WEARSEIZURE_ARTIFACTS_DIR"] == "/single/quoted"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_server_run_without_the_vars_exits_with_one_readable_message():
    with pytest.raises(SystemExit) as excinfo:
        require_server_env(["train.py", "profile=server", "data=chbmit"])
    message = str(excinfo.value)
    assert "CHBMIT_RAW_DIR" in message
    assert "WEARSEIZURE_ARTIFACTS_DIR" in message
    assert "export" in message


def test_server_run_with_the_vars_set_passes(monkeypatch):
    monkeypatch.setenv("CHBMIT_RAW_DIR", "/data/chbmit")
    monkeypatch.setenv("WEARSEIZURE_ARTIFACTS_DIR", "/data/art")
    require_server_env(["train.py", "profile=server", "data=chbmit"])


def test_only_the_missing_variable_is_named(monkeypatch):
    monkeypatch.setenv("CHBMIT_RAW_DIR", "/data/chbmit")
    with pytest.raises(SystemExit) as excinfo:
        require_server_env(["train.py", "profile=server"])
    lines = [ln for ln in str(excinfo.value).splitlines() if ln.startswith("  ")]
    assert any("WEARSEIZURE_ARTIFACTS_DIR" in ln for ln in lines)
    assert not any(ln.strip() == "CHBMIT_RAW_DIR" for ln in lines)


def test_local_runs_are_untouched():
    # profile=local_synthetic needs no environment variables at all.
    require_server_env(["train.py", "profile=local_synthetic"])
    require_server_env(["train.py"])
