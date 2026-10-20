import pytest

from plan_and_act.scenes import SCENES
from plan_and_act.sim.assets import AssetsMissing, panda_dir
from plan_and_act.sim.meshes import have_meshes

# Scenes with scanned objects. A test with one of these names as a parameter needs the objects.
MESH_SCENES = {name for name, s in SCENES.items() if s.meshes}


def _have_assets() -> bool:
    try:
        panda_dir()
    except AssetsMissing:
        return False
    return True


def _needs_objects(item) -> bool:
    if "objects" in item.keywords:
        return True
    params = getattr(getattr(item, "callspec", None), "params", {})
    return any(isinstance(v, str) and v in MESH_SCENES for v in params.values())


def pytest_collection_modifyitems(config, items):
    panda = _have_assets()
    objects = have_meshes({m for s in SCENES.values() for m in s.meshes})
    no_panda = pytest.mark.skip(reason="Panda model missing: run scripts/fetch_assets.sh")
    no_objects = pytest.mark.skip(reason="scanned objects missing: run scripts/fetch_objects.sh")
    for item in items:
        if not panda and "assets" in item.keywords:
            item.add_marker(no_panda)
        elif not objects and _needs_objects(item):
            item.add_marker(no_objects)
