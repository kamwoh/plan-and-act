import threading

import pytest

from armmind.env import MujocoEnv, SimThread
from armmind.scenes import SCENES
from armmind.types import Action

CUBES = SCENES["cubes"]
TASKS = CUBES.tasks


async def test_sim_thread_uses_one_thread():
    sim = SimThread()
    names = {await sim.call(lambda: threading.current_thread().name) for _ in range(5)}
    sim.shutdown()
    assert len(names) == 1
    assert names != {threading.current_thread().name}


@pytest.mark.assets
def test_observe_has_two_images_and_no_positions():
    env = MujocoEnv(CUBES, TASKS["reach"], seed=0, instruction="Touch it.")
    try:
        obs = env.observe()
        assert obs.front_png[:4] == b"\x89PNG" and obs.top_png[:4] == b"\x89PNG"
        assert obs.objects is None
        assert obs.gripper_yaw == pytest.approx(0.0, abs=1.0)
        assert obs.gripper_width == pytest.approx(0.08, abs=0.005)
        assert env.instruction == "Touch it." and env.has_check and env.check_name == "reach"
        assert env.scene_name == "cubes" and env.scene_text == CUBES.describe()
        assert env.image_text is None
    finally:
        env.close()


@pytest.mark.assets
def test_debug_switch_shows_positions():
    env = MujocoEnv(CUBES, TASKS["reach"], seed=0, instruction="Touch it.", show_positions=True)
    try:
        objects = env.observe().objects
        assert set(objects) == {"red_cube", "blue_cube", "green_cube"}
        assert objects["red_cube"][2] == pytest.approx(0.02, abs=0.002)
    finally:
        env.close()


@pytest.mark.assets
def test_act_sends_front_and_top_jpeg_frames():
    frames: list[tuple[str, bytes]] = []
    env = MujocoEnv(CUBES, TASKS["reach"], seed=0, instruction="Touch it.",
                    on_frame=lambda c, j: frames.append((c, j)))
    try:
        frames.clear()
        res = env.act(Action("wait", seconds=1.0))
        assert res.status == "reached"
        front = [j for c, j in frames if c == "front"]
        top = [j for c, j in frames if c == "top"]
        assert 28 <= len(front) <= 32
        assert 8 <= len(top) <= 12
        assert len(front) + len(top) == len(frames)
        assert all(j[:2] == b"\xff\xd8" for _, j in frames)
        assert env.success() is False
    finally:
        env.close()


@pytest.mark.assets
def test_observe_renders_with_shadows(monkeypatch):
    env = MujocoEnv(CUBES, TASKS["reach"], seed=0, instruction="Touch it.")
    calls = []
    real = env.cams.render
    monkeypatch.setattr(env.cams, "render", lambda cam, shadows=True: calls.append(shadows) or real(cam, shadows))
    try:
        env.observe()
        assert calls == [True, True]
    finally:
        env.close()


@pytest.mark.assets
async def test_env_works_through_sim_thread():
    sim = SimThread()
    env = await sim.call(MujocoEnv, CUBES, TASKS["reach"], 0, "Touch it.")
    for _ in range(3):
        obs = await sim.call(env.observe)
        assert len(obs.front_png) > 1000
    await sim.call(env.close)
    sim.shutdown()


@pytest.mark.assets
def test_custom_env_has_no_check():
    env = MujocoEnv(CUBES, None, seed=0, instruction="Wave at the blue cube.")
    try:
        assert env.has_check is False and env.check_name is None
        assert env.success() is False
        assert env.instruction == "Wave at the blue cube."
    finally:
        env.close()


@pytest.mark.assets
def test_observe_reports_the_yaw():
    env = MujocoEnv(CUBES, TASKS["reach"], seed=0, instruction="Touch it.")
    try:
        env.act(Action("move_to", (0.5, 0.0, 0.2), yaw=-30.0))
        assert env.observe().gripper_yaw == pytest.approx(-30.0, abs=3.0)
    finally:
        env.close()


@pytest.mark.assets
def test_observe_gives_new_events_and_debug_joints():
    from tests.test_world import _probe_scene, _push
    scene = _probe_scene()
    plain = MujocoEnv(scene, None, seed=0, instruction="x")
    env = MujocoEnv(scene, None, seed=0, instruction="x", show_positions=True)
    try:
        assert plain.observe().joints is None
        assert env.observe().joints == {"slider": pytest.approx(0.01, abs=1e-3)}
        _push(env.world, 1.0, 50)
        assert env.observe().events == ("slider out",)
        assert env.observe().events == ()  # each event comes once
    finally:
        plain.close()
        env.close()
