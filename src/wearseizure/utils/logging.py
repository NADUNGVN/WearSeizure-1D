from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _ensure_root_stdout_handler(level: int) -> None:
    """Give the ROOT logger a stdout handler if nothing has configured one yet.

    Deliberately on root rather than on the module logger. Hydra configures
    logging with `dictConfig`, whose `root:` section removes root's existing
    handlers before installing its own console + file handlers -- so this
    fallback is replaced rather than duplicated once a Hydra job starts, while
    still giving non-Hydra entry points (e.g. scripts/measure_model_size.py,
    plain `pytest`) visible output.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """A module logger that propagates to root.

    `propagate` must stay True. Hydra writes `${hydra.runtime.output_dir}/
    <job>.log` by attaching a FileHandler to the ROOT logger; a module logger
    with `propagate=False` and its own stdout handler never reaches it, which
    is why every file under `artifacts/runs/*/*.log` was 0 bytes despite the
    runs themselves logging normally to the console.
    """
    _ensure_root_stdout_handler(level)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    return logger
