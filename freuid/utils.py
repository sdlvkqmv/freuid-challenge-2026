"""Reproducibility, logging, paths, git provenance."""
from __future__ import annotations

import logging
import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(exp_root: str, exp_name: str) -> Path:
    run = Path(exp_root) / f"{exp_name}_{timestamp()}"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def device_from_cfg(cfg) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{cfg.get('gpu', 0)}")
    return torch.device("cpu")
