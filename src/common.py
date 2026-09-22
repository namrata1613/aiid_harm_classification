"""Shared helpers: config + path resolution."""

import os
import pathlib
import yaml
from typing import Optional

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_config(path: Optional[str] = None) -> dict:
    if path:
        p = pathlib.Path(path)
    # elif os.environ.get("AIID_CONFIG"):
    #     p = pathlib.Path(os.environ["AIID_CONFIG"])
    else:
        p = _ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_path(*parts: str) -> pathlib.Path:
    cfg = load_config()
    p = _ROOT / cfg["run"]["data_dir"]
    p = p.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def norm(s: str) -> str:
    """Normalize a column name for tolerant matching."""
    return "".join(str(s).lower().split())