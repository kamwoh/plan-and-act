"""The camera data of a normal run, and the projection and the path overlay of the page."""
import json
import subprocess
from pathlib import Path

import pytest

from armmind.agent.loop import Agent, AgentConfig
from armmind.llm.fake import FakeBackend
from armmind.scenes import SCENES
from tests.fakes import Recorder, direct

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "tests" / "js"


def _env(scene="cubes", seed=0):
    from armmind.env import MujocoEnv
    return MujocoEnv(SCENES[scene], None, seed, "")


@pytest.mark.assets
@pytest.mark.parametrize("scene", ["cubes", "drawer"])
async def test_run_start_has_the_cameras_of_the_scene(scene):
    from armmind.sim.planview import cameras
    env, rec = _env(scene), Recorder()
    try:
        await Agent(FakeBackend([]), AgentConfig(max_steps=1)).run(env, direct, rec)
        want = cameras(env.world, env.cams.w, env.cams.h)
    finally:
        env.close()
    got = dict(rec.events)["run_start"]["cameras"]
    assert got == want
    assert {c["name"] for c in got} == {"front", "top"} and all((c["width"], c["height"]) == (640, 480) for c in got)
    front = next(c for c in got if c["name"] == "front")
    assert front["label"] == SCENES[scene].camera_labels["front"]
    assert front["grid"] == (scene != "drawer")  # the drawer scene has its own side camera


async def test_a_run_without_camera_data_has_none_in_run_start():
    from tests.fakes import FakeEnv
    rec = Recorder()
    await Agent(FakeBackend([]), AgentConfig(max_steps=1)).run(FakeEnv(), direct, rec)
    assert "cameras" not in dict(rec.events)["run_start"]


@pytest.mark.assets
@pytest.mark.parametrize("scene", ["cubes", "drawer"])
def test_the_page_projection_matches_mujoco(scene, tmp_path):
    from armmind.sim.planview import cameras
    env = _env(scene)
    try:
        cube = env.world.objects[0].name
        points = [[x, y, 0.0] for x in (-1.0, 1.0) for y in (-1.0, 1.0)]  # the corners of the table plane
        points += [[0.2, -0.4, 0.0], [0.8, 0.4, 0.0], [float(v) for v in env.world.object_pos(cube)]]
        cases = [{"camera": c, "points": points, "pixels": env.cams.project(c["name"], points).tolist()}
                 for c in cameras(env.world, env.cams.w, env.cams.h)]
    finally:
        env.close()
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    r = subprocess.run(["node", str(JS / "camproj_check.js"), str(tmp_path / "cases.json")],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
