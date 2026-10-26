import numpy as np
import pytest

from plan_and_act.agent.prompts import CHAIN_SYSTEM, PLAN_SYSTEM, SYSTEM, chain_system, plan_system, system
from plan_and_act.scenes import SCENES, find_scene
from plan_and_act.sim.robots import ROBOTS
from plan_and_act.types import PANDA_FACTS, ArmFacts

# The arms whose scripted cubes pick works for seeds 0 to 2. Each other arm loads and starts at its ready point.
PICKS = ("panda", "fr3", "ur5e", "ur10e", "gen3", "iiwa14", "sawyer", "rizon4", "xarm7", "vx300s", "yam", "arx_l5",
         "yam_ultra")


def test_the_panda_is_first_and_names_are_unique():
    assert next(iter(ROBOTS)) == "panda"
    assert all(name == r.name and "@" not in name for name, r in ROBOTS.items())


def test_a_scene_key_picks_the_robot():
    assert find_scene("cubes") is SCENES["cubes"] and find_scene("cubes@panda") is SCENES["cubes"]
    ur5e = find_scene("cubes@ur5e")
    assert (ur5e.name, ur5e.robot, ur5e.tasks) == ("cubes@ur5e", "ur5e", SCENES["cubes"].tasks)
    assert find_scene("cubes@ur5e") is ur5e  # the same object each time
    assert find_scene("cubes@nothing") is None and find_scene("nothing@ur5e") is None


def test_the_panda_prompts_are_the_old_text():
    assert (system(), plan_system(), chain_system()) == (SYSTEM, PLAN_SYSTEM, CHAIN_SYSTEM)
    assert "Franka Panda robot arm" in SYSTEM and "x=0, y=0" in SYSTEM and "0.08 m wide" in SYSTEM
    assert "x 0.30, y 0.00, z 0.40" in PLAN_SYSTEM


def test_the_prompts_name_another_arm():
    arm = ArmFacts("Universal Robots UR5e", (0.1, 0.0), 0.085, (0.35, 0.0, 0.3))
    text = chain_system(arm)
    assert "Universal Robots UR5e robot arm" in text and "Franka" not in text
    assert "x=0.1, y=0" in text and "0.085 m wide" in text and "x 0.35, y 0.00, z 0.30" in text
    assert "{" + "width}" not in system(arm)
    assert "cannot reach every point" in text and "cannot reach every point" not in CHAIN_SYSTEM
    assert "tilts" not in text and "tilts the gripper up to 45 degrees" in chain_system(ArmFacts(max_tilt=45.0))


def test_the_panda_needs_no_model_for_its_facts():
    from plan_and_act.sim.world import arm_facts
    assert arm_facts(SCENES["cubes"]) is PANDA_FACTS


@pytest.mark.assets
def test_a_robot_changes_the_state_version():
    from plan_and_act.sim.state import scene_version
    assert scene_version(SCENES["cubes"]) != scene_version(find_scene("cubes@ur5e"))


@pytest.mark.assets
@pytest.mark.parametrize("name", list(ROBOTS))
def test_every_robot_starts_ready_with_the_gripper_down_and_open(name):
    from plan_and_act.sim.arm import Arm
    from plan_and_act.sim.world import World, arm_facts
    scene = find_scene(f"cubes@{name}")
    w = World(scene)
    w.reset(scene.layout(0))
    arm = Arm(w)
    arm.reset_ready()
    robot = ROBOTS[name]
    np.testing.assert_allclose(w.tcp(), robot.ready, atol=0.02)
    tilt = np.degrees(np.arccos(-w.data.site_xmat[w.tcp_id].reshape(3, 3)[2, 2]))
    assert tilt <= max(robot.max_tilt + 1.0, 3.0)  # down, or tilted within the limit of the arm
    assert abs(w.gripper_yaw()) < 3.0
    assert 0.015 < w.parts.open_width < 0.12
    assert w.gripper_width() == pytest.approx(w.parts.open_width, abs=0.008)  # a weak finger servo sags a little
    assert w.gripper_contacts() == [] and not w.gripper_closed()
    assert arm_facts(scene).title == robot.title
    agent = w.parts.agent
    assert [p["name"] for p in agent["parts"]][-2:] == ["finger_left", "finger_right"]


@pytest.mark.assets
@pytest.mark.parametrize("name", list(ROBOTS))
def test_every_gripper_closes_on_nothing(name):
    from plan_and_act.sim.arm import Arm
    from plan_and_act.sim.world import World
    from plan_and_act.types import CLOSED_ON_NOTHING_M, Action
    scene = find_scene(f"cubes@{name}")
    w = World(scene)
    w.reset(scene.layout(0))
    arm = Arm(w)
    arm.reset_ready()
    arm.act(Action("close"))
    assert w.gripper_closed() and w.gripper_width() < CLOSED_ON_NOTHING_M and arm.held() == ()


@pytest.mark.assets
@pytest.mark.parametrize("name,seed", [(n, s) for n in PICKS for s in range(3)])
def test_the_scripted_pick_works_on_these_arms(name, seed):
    from plan_and_act.sim.arm import Arm
    from plan_and_act.sim.world import World
    scene = find_scene(f"cubes@{name}")
    task = scene.tasks["pick"]
    w = World(scene)
    w.reset(scene.layout(seed))
    arm = Arm(w)
    arm.reset_ready()
    for action in task.solution(w):
        arm.act(action)
    assert task.success(w)


@pytest.mark.assets
def test_robot_parts_do_not_touch_each_other_but_touch_the_scene():
    from plan_and_act.sim.robots import ROBOT_CONAFFINITY, ROBOT_CONTYPE
    from plan_and_act.sim.world import World
    w = World(find_scene("cubes@ur5e"))
    m = w.model
    robot = [g for g in range(m.ngeom) if m.geom_contype[g] == ROBOT_CONTYPE]
    assert robot and all(m.geom_conaffinity[g] == ROBOT_CONAFFINITY for g in robot)
    cube = m.geom("red_cube").id
    assert m.geom_contype[cube] & ROBOT_CONAFFINITY and not ROBOT_CONTYPE & ROBOT_CONAFFINITY
