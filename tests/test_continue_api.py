"""Re-plan modes: "continue" goes on from the end state of the last execution, "start" resets the scene."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from armmind.llm.fake import FakeBackend
from armmind.scenes import SCENES
from armmind.server.app import create_app
from armmind.sim.state import END_STATE, START_STATE, scene_version
from tests.fakes import judged
from tests.test_agent_plan import good_plan
from tests.test_plan_api import PICK
from tests.test_replan_api import TrueStateEnv, run_events
from tests.test_server import make_env, wait_idle

ENVS: list = []


def write_state(path, scene="cubes", seed=5, marker=0.0):
    """A state file in the format of armmind.sim.state, with a marker value in qpos."""
    np.savez(path, scene=np.array(scene), seed=np.array(seed), version=np.array(scene_version(SCENES[scene])),
             time=np.array(1.5), yaw=np.array(0.0), events=np.array([], dtype=str), qpos=np.array([marker]),
             qvel=np.zeros(1), act=np.zeros(0), ctrl=np.zeros(1), qacc_warmstart=np.zeros(1))


class StateEnv(TrueStateEnv):
    """Saves a state file whose qpos counts the executions, so a test can tell the states apart."""
    executions = 0

    def execute_plan(self, keyframes, should_stop, on_action=None):
        StateEnv.executions += 1
        return super().execute_plan(keyframes, should_stop, on_action)

    def check_plan(self, belief, keyframes):
        self.checked = (belief, keyframes)
        return super().check_plan(belief, keyframes)

    def save_state(self, path):
        write_state(path, self.scene_name, self.seed, float(StateEnv.executions))


def factory(scene, task, instruction, seed, show_positions, on_frame, start=None):
    env = make_env(scene, task, instruction, cls=StateEnv)
    env.seed, env.start = seed, start
    env.start_marker = float(np.load(start)["qpos"][0]) if start is not None else None
    ENVS.append(env)
    return env


def client(tmp_path, backend):
    return TestClient(create_app(backend, factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons"))


def plan_and_execute(c):
    run_id = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
    wait_idle(c)
    assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
    wait_idle(c)
    return run_id


@pytest.fixture(autouse=True)
def _clear():
    ENVS.clear()
    StateEnv.executions = 0


def test_execute_saves_the_end_state(tmp_path):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        run_id = plan_and_execute(c)
    assert float(np.load(tmp_path / "runs" / run_id / END_STATE)["qpos"][0]) == 1.0
    assert run_events(tmp_path, run_id)[0]["round_mode"] is None  # a round 1 plan has no mode


def test_a_continue_replan_starts_from_the_end_state(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan()])
    with client(tmp_path, backend) as c:
        first = plan_and_execute(c)
        r = c.post(f"/api/plan/{first}/replan")  # continue is the default
        assert r.status_code == 200, r.text
        wait_idle(c)
        second = r.json()["run_id"]
    folder = tmp_path / "runs" / second
    assert ENVS[-1].start == folder / START_STATE and ENVS[-1].start_marker == 1.0
    assert ENVS[-1].checked is not None  # the plan check ran on the env of the end state
    start = run_events(tmp_path, second)[0]
    assert start["round_mode"] == "continue" and start["round"] == 2 and start["parent"] == first
    msg = backend.sent[-1][0]
    assert msg.images == (b"front", b"top")  # the current images are the end images: no extra ones
    assert "did not reset" in msg.text and "only what is left" in msg.text
    assert "Images 3 and 4" not in msg.text and "back at its start" not in msg.text
    assert "Plan what is left of the task now" in msg.text
    for secret in ("0.587", "0.5871", "-0.123", "0.0213", "33.3", "7.31", "missed by", "blue_cube", "debug"):
        assert secret not in msg.text, secret


def test_a_start_replan_is_as_in_v1_12(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan()])
    with client(tmp_path, backend) as c:
        first = plan_and_execute(c)
        r = c.post(f"/api/plan/{first}/replan", json={"mode": "start"})
        assert r.status_code == 200, r.text
        wait_idle(c)
        second = r.json()["run_id"]
    assert ENVS[-1].start is None and not (tmp_path / "runs" / second / START_STATE).exists()
    assert run_events(tmp_path, second)[0]["round_mode"] == "start"
    msg = backend.sent[-1][0]
    assert msg.images == (b"front", b"top", b"front", b"top")
    assert "Images 3 and 4 show the end" in msg.text and "back at its start" in msg.text
    assert "did not reset" not in msg.text and "Plan the whole task now" in msg.text


def test_execute_of_a_continue_plan_starts_from_the_same_state_each_time(tmp_path):
    with client(tmp_path, FakeBackend([good_plan(), judged(), good_plan(), *[judged()] * 4])) as c:
        first = plan_and_execute(c)
        second = c.post(f"/api/plan/{first}/replan").json()["run_id"]
        wait_idle(c)
        for _ in range(2):
            assert c.post(f"/api/plan/{second}/execute").status_code == 200
            wait_idle(c)
        # The parent runs again: its end state changes, but the continue round keeps its own start.
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        assert c.post(f"/api/plan/{second}/execute").status_code == 200
        wait_idle(c)
    runs = [e for e in ENVS if e.start is not None]
    assert len(runs) == 4  # the plan and three executions of round 2
    assert {e.start for e in runs} == {tmp_path / "runs" / second / START_STATE}
    assert [e.start_marker for e in runs] == [1.0, 1.0, 1.0, 1.0]
    ends = [e for e in run_events(tmp_path, second) if e["kind"] == "exec_end"]
    assert len(ends) == 3 and all(e["status"] == "stopped" for e in ends)


def test_continue_without_an_end_state_is_409(tmp_path):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        first = plan_and_execute(c)
        (tmp_path / "runs" / first / END_STATE).unlink()
        r = c.post(f"/api/plan/{first}/replan", json={"mode": "continue"})
        assert r.status_code == 409 and "end state" in r.json()["detail"]
        assert c.post(f"/api/plan/{first}/replan", json={"mode": "sideways"}).status_code == 422


def test_continue_from_a_state_of_another_scene_is_409(tmp_path):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        first = plan_and_execute(c)
        write_state(tmp_path / "runs" / first / END_STATE, scene="tray")
        r = c.post(f"/api/plan/{first}/replan")
        assert r.status_code == 409 and "tray" in r.json()["detail"]


def test_execute_of_a_continue_plan_without_its_start_state_is_409(tmp_path):
    with client(tmp_path, FakeBackend([good_plan(), judged(), good_plan(), judged()])) as c:
        first = plan_and_execute(c)
        second = c.post(f"/api/plan/{first}/replan").json()["run_id"]
        wait_idle(c)
        (tmp_path / "runs" / second / START_STATE).unlink()
        r = c.post(f"/api/plan/{second}/execute")
        assert r.status_code == 409 and "state" in r.json()["detail"]


def test_a_new_execution_removes_the_old_end_state_first(tmp_path, monkeypatch):
    def full(self, path):
        raise OSError("disk full")

    with client(tmp_path, FakeBackend([good_plan()])) as c:
        first = plan_and_execute(c)
        monkeypatch.setattr(StateEnv, "save_state", full)
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        assert [e["kind"] for e in run_events(tmp_path, first)][-2:] == ["exec_end", "verdict"]  # the result stays
        r = c.post(f"/api/plan/{first}/replan")
        assert r.status_code == 409 and "end state" in r.json()["detail"]


def test_the_plan_mode_must_be_continue_or_start(tmp_path):
    with client(tmp_path, FakeBackend([])) as c:
        assert c.post("/api/plan", json={"instruction": PICK, "auto": True, "mode": "x"}).status_code == 422


def test_the_plan_prompt_speaks_of_a_round_that_continues():
    from armmind.agent.prompts import PLAN_SYSTEM, Feedback, build_plan_text
    from armmind.types import Observation
    assert "re-plan that continues" in PLAN_SYSTEM and "did not reset" in PLAN_SYSTEM
    fb = Feedback((), (), "failed", True, False, None, None, (), "20261018-100000", 2, 2, mode="continue")
    obs = Observation(b"f", b"t", (0.5, -0.1, 0.08), 0.04, ("red_cube",))
    text = build_plan_text("Scene: s", PICK, obs, feedback=fb, gripper="closed")
    assert "The gripper is closed now." in text and "There are no images" not in text
    assert "The gripper is" not in build_plan_text("Scene: s", PICK, obs, gripper="closed")  # round 1
