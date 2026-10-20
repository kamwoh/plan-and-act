"""Finds the Panda model that scripts/fetch_assets.sh downloads."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class AssetsMissing(RuntimeError):
    pass


def panda_dir() -> Path:
    base = Path(os.environ.get("PLAN_AND_ACT_ASSETS", REPO_ROOT / "assets" / "mujoco_menagerie"))
    folder = base / "franka_emika_panda"
    if not (folder / "panda.xml").is_file():
        raise AssetsMissing(f"Panda model not found in {folder}. Run scripts/fetch_assets.sh")
    return folder
