from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Return absolute path to the repository root."""
    # helper lives under <repo>/src/helper/, so parents[2] == repo root
    return Path(__file__).resolve().parents[2]


def src_dir() -> Path:
    """Return the <repo>/src directory."""
    return repo_root() / "src"


def modeling_dir(subdir: str = "yapo") -> Optional[Path]:
    """Return the modeling/<subdir> folder if it exists."""
    candidate = src_dir() / "modeling" / subdir
    return candidate if candidate.exists() else None


def add_to_sys_path(path: Path) -> None:
    """Insert a path at the front of sys.path if it is not already present."""
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def ensure_src_on_sys_path() -> Path:
    """Ensure <repo>/src is on sys.path."""
    src = src_dir()
    add_to_sys_path(src)
    return src


def ensure_modeling_on_sys_path(subdir: str = "yapo") -> Optional[Path]:
    """Ensure modeling/<subdir> is on sys.path if it exists."""
    modeling = modeling_dir(subdir=subdir)
    if modeling:
        add_to_sys_path(modeling)
    return modeling
