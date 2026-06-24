"""Minimal YAML config with dotted-key CLI overrides and resolved-config dump."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access (cfg.train.lr) and dotted get/set."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_dotted(self, dotted: str) -> Any:
        node: Any = self
        for part in dotted.split("."):
            node = node[part]
        return node

    def set_dotted(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _coerce(s: str) -> Any:
    """Parse a CLI override value with YAML rules (int/float/bool/list/str)."""
    try:
        return yaml.safe_load(s)
    except yaml.YAMLError:
        return s


def load_config(path: str, overrides: list[str] | None = None) -> Cfg:
    cfg = Cfg(yaml.safe_load(Path(path).read_text()))
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"--set expects key=value, got: {ov}")
        key, raw = ov.split("=", 1)
        cfg.set_dotted(key.strip(), _coerce(raw.strip()))
    return cfg


def dump_config(cfg: Cfg, path: str | Path) -> None:
    plain = copy.deepcopy(dict(cfg))
    Path(path).write_text(yaml.safe_dump(plain, sort_keys=False, allow_unicode=True))


def parse_args_and_config() -> Cfg:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4")
    args = p.parse_args()
    return load_config(args.config, args.set)
