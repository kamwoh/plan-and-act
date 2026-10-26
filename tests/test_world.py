import numpy as np
import pytest

from plan_and_act.scenes import SCENES
from plan_and_act.sim.assets import AssetsMissing, panda_dir
from plan_and_act.sim.world import World

CUBES = SCENES["cubes"]

LAYOUT = {"red_cube": (0.5, -0.1), "blue_cube": (0.5, 0.1), "green_cube": (0.6, 0.0)}


def test_panda_dir_missing_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAN_AND_ACT_ASSETS", str(tmp_path))
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
    z_axis = w.data.site_xmat[w.tcp_id].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(z_axis, [0, 0, -1], atol=1e-3)
    assert w.tcp()[2] > 0.3


@pytest.mark.assets
def test_no_gripper_contacts_at_home():
    w = World(CUBES)
    w.reset(LAYOUT)
    w.step(50)
    assert w.gripper_contacts() == []


@pytest.mark.assets
def test_the_robot_comes_before_the_scene_in_the_model():
    """The arm joints come first in qpos, as with the old include of panda.xml."""
    w = World(CUBES)
    assert [w.model.body(b).name for b in range(1, 4)] == ["link0", "link1", "link2"]
    assert list(w.parts.arm_qpos) == list(range(7)) and w.model.joint(7).name == "finger_joint1"


@pytest.mark.assets
def test_gripper_yaw_is_zero_at_home():
    w = World(CUBES)
    w.reset(LAYOUT)
    assert w.gripper_yaw() == pytest.approx(0.0, abs=0.1)


# A test scene: a block on a slide joint along x, with a monitor that fires when it is out more than 5 cm.
SLIDER_XML = """    <body name="slider" pos="0.6 -0.3 0.03">
      <joint name="slider" type="slide" axis="1 0 0" range="0 0.1" damping="0"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>
    </body>"""


def _probe_scene():
    from plan_and_act.scenes.base import Scene, threshold_monitor
    return Scene(name="probe", title="Probe", objects=(), fixtures_xml=SLIDER_XML, fixtures_text="",
                 region=(0.4, 0.5, -0.1, 0.1), min_gap=0.1, tasks={}, joints={"slider": "slider out {:.3f} m"},
                 monitors=(threshold_monitor("slider", 0.05, 0.02, "slider out"),), initial_qpos={"slider": 0.01})


def _push(w, speed: float, steps: int) -> None:
    w.data.qvel[w.model.jnt_dofadr[w.model.joint("slider").id]] = speed
    w.step(steps)


@pytest.mark.assets
def test_reset_sets_scene_joints_and_clears_events():
    w = World(_probe_scene())
    w.reset({})
    assert w.joint_pos("slider") == pytest.approx(0.01)
    w.events.append("old")
    w.reset({}, {"slider": 0.07})
    assert w.joint_pos("slider") == pytest.approx(0.07)
    assert w.events == []
    assert w.joints() == {"slider": pytest.approx(0.07)}


@pytest.mark.assets
def test_monitor_fires_once_per_rising_edge():
    w = World(_probe_scene())
    w.reset({})
    _push(w, 1.0, 50)  # out to about 0.1, past 0.05 once
    assert w.events == ["slider out"]
    w.step(100)  # stays out: no new event
    assert w.events == ["slider out"]
    _push(w, -1.0, 30)  # back to about 0.04: not re-armed yet
    _push(w, 1.0, 30)
    assert w.events == ["slider out"]
    _push(w, -1.0, 100)  # back to 0: re-armed
    _push(w, 1.0, 100)
    assert w.events == ["slider out", "slider out"]
    w.reset({})
    _push(w, 1.0, 50)
    assert w.events == ["slider out"]  # a reset re-arms the monitor too
