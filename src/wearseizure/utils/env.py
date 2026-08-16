"""Load `.env` and fail legibly when the server profile's paths are missing.

Two things were broken before this module existed:

1. `.env.example` says "Copy to .env and fill in on the machine that runs
   profile=server", and README pointed at `.env` as the way to set paths -- but
   nothing in the codebase ever read that file, and `python-dotenv` was not a
   dependency. A `.env` file was decorative; only a manual `export` worked.

2. When the variables really are unset, `configs/profile/server.yaml` fails at
   `${oc.env:WEARSEIZURE_ARTIFACTS_DIR}` while Hydra is resolving
   `hydra.run.dir` -- i.e. *before* the decorated `main()` runs, so no guard
   inside the script can catch it. The result is ~200 lines of interleaved
   ANTLR/OmegaConf traceback (three times over, when shards run concurrently)
   ending in a single meaningful line.

So both steps have to happen at **import time**, before Hydra resolves
anything: read `.env` if present, then refuse early with a one-line message if
a `profile=server` run still has nothing to interpolate.

Deliberately dependency-free. The training server's conda env is pinned and
shared; a new third-party import is a worse trade than twenty lines of parsing.
"""
from __future__ import annotations

import os
from pathlib import Path

from wearseizure.utils.paths import repo_root

#: Variables `configs/profile/server.yaml` interpolates with `${oc.env:...}`.
SERVER_ENV_VARS = ("CHBMIT_RAW_DIR", "WEARSEIZURE_ARTIFACTS_DIR")


def _candidate_env_files() -> list[Path]:
    # Hydra runs with `job.chdir: false`, so the working directory is wherever
    # the command was launched -- usually but not always the repo root.
    return [repo_root() / ".env", Path.cwd() / ".env"]


def load_env_file(path: str | Path | None = None) -> list[str]:
    """Set variables from a `.env` file. Returns the names actually set.

    An existing environment variable always wins, so an explicit `export` (or
    a value inherited from the parent shell) is never silently overridden.
    `~` is expanded, because `scripts/server_bootstrap.sh` documents paths as
    `~/Manh/...` and a shell expands that on `export` while a `.env` file does
    not.
    """
    paths = [Path(path)] if path is not None else _candidate_env_files()
    set_names: list[str] = []

    for env_path in paths:
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            value = value.strip().strip('"').strip("'")
            if not key or key in os.environ:
                continue
            os.environ[key] = os.path.expanduser(value)
            set_names.append(key)
        break  # first file found wins

    return set_names


def require_server_env(argv: list[str]) -> None:
    """Exit with one clear line if a `profile=server` run has unset paths.

    Raises `SystemExit` rather than an exception so the user sees the message
    and nothing else -- the traceback this replaces is pure noise.
    """
    if not any(arg.strip() == "profile=server" for arg in argv):
        return

    missing = [name for name in SERVER_ENV_VARS if not os.environ.get(name)]
    if not missing:
        return

    raise SystemExit(
        "profile=server needs these environment variables, and they are unset:\n"
        + "".join(f"  {name}\n" for name in missing)
        + "\nSet them in this shell:\n"
        "  export CHBMIT_RAW_DIR=~/Manh/datasets/CHB-MIT/1.0.0\n"
        "  export WEARSEIZURE_ARTIFACTS_DIR=~/Manh/WearSeizure-1D-artifacts\n"
        "\nor put them in a .env file at the repo root (now actually read; see\n"
        ".env.example). Without them Hydra fails while resolving hydra.run.dir,\n"
        "before any script code runs, which is why the traceback is unreadable."
    )


def bootstrap_env(argv: list[str]) -> None:
    """`load_env_file()` then `require_server_env()` -- the entry-point call."""
    load_env_file()
    require_server_env(argv)
