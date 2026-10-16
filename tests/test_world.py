import numpy as np
import pytest

from armmind.scenes import SCENES
from armmind.sim.assets import AssetsMissing, panda_dir
from armmind.sim.world import World

CUBES = SCENES["cubes"]

LAYOUT = {"red_cube": (0.5, -0.1), "blue_cube": (0.5, 0.1), "green_cube": (0.6, 0.0)}


def test_panda_dir_missing_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ARMMIND_ASSETS", str(tmp_path))
    with pytest.raises(AssetsMissing, match="fetch_assets.sh"):
        panda_dir()


@pytest.mark.assets
def test_reset_places_cubes_on_table():
    w = World(CUBES)
    w.reset(LAYOUT)
    for obj in w.objects:
        x, y = LAYOUT[obj.name]
        np.testing.assert_allclose(w.object_pos(obj.name), [x, y, 0.02], atol=1e-6)


@pytest.mark.assets
def test_reset_rejects_incomplete_layout():
    w = World(CUBES)
    with pytest.raises(ValueError, match="green_cube"):
        w.reset({"red_cube": (0.5, 0.0), "blue_cube": (0.5, 0.2)})


@pytest.mark.assets
def test_home_pose_gripper_open_and_points_down():
    w = World(CUBES)
    w.reset(LAYOUT)
    assert w.gripper_width() == pytest.approx(0.08, abs=1e-3)
    z_axis = w.data.xmat[w.hand_id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(z_axis, [0, 0, -1], atol=1e-3)
    assert w.tcp()[2] > 0.3


@pytest.mark.assets
def test_no_gripper_contacts_at_home():
    w = World(CUBES)
    w.reset(LAYOUT)
    w.step(50)
    assert w.gripper_contacts() == []


@pytest.mark.assets
def test_scene_file_is_named_after_the_scene():
    w = World(CUBES)
    assert (w.scene_file.name, w.scene_file.parent.joinpath("panda.xml").is_file()) == ("armmind_scene_cubes.xml", True)
