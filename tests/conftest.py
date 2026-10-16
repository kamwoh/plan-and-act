import pytest

from armmind.sim.assets import AssetsMissing, panda_dir


def _have_assets() -> bool:
    try:
        panda_dir()
    except AssetsMissing:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    if _have_assets():
        return
    skip = pytest.mark.skip(reason="Panda model missing: run scripts/fetch_assets.sh")
    for item in items:
        if "assets" in item.keywords:
            item.add_marker(skip)
