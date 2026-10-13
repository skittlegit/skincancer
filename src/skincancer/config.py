"""YAML configs with `extends`, dotted `--set` overrides and `{key}` name templating."""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(_load(path.parent / parent), cfg)
    return cfg


def get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def set_dotted(cfg: dict, dotted: str, value) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def _fill(template: str, cfg: dict) -> str:
    return re.sub(r"\{([\w.]+)\}", lambda m: str(get(cfg, m.group(1))), template)


def load_config(path: str | Path, overrides: list[str] | tuple = ()) -> dict:
    """Load a config, apply `key.sub=value` overrides, then fill `{key}` templates
    in `name` and `init_from` (e.g. name: e5_f{data.fraction})."""
    cfg = _load(Path(path))
    for item in overrides:
        key, _, raw = item.partition("=")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    for key in ("name", "init_from"):
        if isinstance(cfg.get(key), str):
            cfg[key] = _fill(cfg[key], cfg)
    return cfg


def load_data_config(path: str | Path = "configs/data.yaml") -> dict:
    return _load(Path(path))
