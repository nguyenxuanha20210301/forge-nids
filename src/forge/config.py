"""yaml configs with inheritance and command line overrides.

an experiment config may name parent files under the key inherit. parents are merged first, in order, then the file itself, then overrides of the form a.b.c=value.
"""

from copy import deepcopy
from pathlib import Path

import yaml


def _merge(base: dict, new: dict) -> dict:
    out = deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path, overrides=()) -> dict:
    path = Path(path).resolve()
    raw = _read(path)
    cfg = {}
    for parent in raw.pop("inherit", []):
        cfg = _merge(cfg, load_config(path.parent / parent))
    cfg = _merge(cfg, raw)
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def save_config(cfg: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
