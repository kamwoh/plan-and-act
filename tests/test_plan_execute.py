"""Run a saved plan once in the physics: POST /api/plan/{id}/execute and MujocoEnv.execute_plan."""
import pytest
from fastapi.testclient import TestClient

from armmind.llm.fake import FakeBackend
from armmind.plan.formats import Keyframe
from armmind.scenes import SCENES
from armmind.server.app import create_app
from armmind.types import Action
from tests.fakes import HangingBackend
from tests.test_agent_plan import good_plan
from tests.test_plan_api import ENVS, PICK, client, events, plan_factory
from tests.test_plan_scores import scripted_keyframes, with_holding
from tests.test_server import wait_idle


def test_keyframes_become_moves_and_gripper_changes():
    from armmind.env import keyframe_actions
    kfs = [Keyframe((0.5, 0.0, 0.15), 10.0, "open", None), Keyframe((0.5, 0.0, 0.02), 10.0, "closed", "red cube"),
           Keyframe((0.5, 0.2, 0.15), 0.0, "closed", "red cube"), Keyframe((0.5, 0.2, 0.06), 0.0, "open", None),
           Keyframe((0.5, 0.2, 0.06), 0.0, "closed", None)]
    assert keyframe_actions(kfs) == [
        Action("move_to", (0.5, 0.0, 0.15), yaw=10.0), Action("move_to", (0.5, 0.0, 0.02), yaw=10.0),
        Action("close", expect="red cube"), Action("move_to", (0.5, 0.2, 0.15), yaw=0.0),
        Action("move_to", (0.5, 0.2, 0.06), yaw=0.0), Action("open", expect="nothing"),
        Action("move_to", (0.5, 0.2, 0.06), yaw=0.0), Action("close", expect="nothing")]


def _execute(scene, task, seed, kfs_from=None, stop=lambda: False):
    from armmind.env import MujocoEnv
    sc = SCENES[scene]
    env = MujocoEnv(sc, sc.tasks[task], seed, sc.tasks[task].instruction)
    try:
        kfs = kfs_from(env) if kfs_from else with_holding(scripted_keyframes(sc.tasks[task].solution(env.world)),
                                                         env.world)
        return env.execute_plan(tuple(kfs), stop)
    finally:
        env.close()


@pytest.mark.assets
@pytest.mark.parametrize("scene, task", [("cubes", "pick"), ("tray", "red_in_tray")])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_scripted_plan_from_the_true_state_succeeds(scene, task, seed):
    out = _execute(scene, task, seed)
    assert out["status"] == "success" and out["success"] is True and out["stopped"] is None
    assert len(out["results"]) == len(out["actions"]) and all(": reached" in r for r in out["results"]
                                                              if r.startswith("move_to"))


@pytest.mark.assets
def test_a_close_on_nothing_stops_the_plan():
    def kfs(env):
        return [Keyframe((0.35, 0.0, 0.15), 0.0, "open", None), Keyframe((0.35, 0.0, 0.02), 0.0, "closed", "red cube"),
                Keyframe((0.35, 0.0, 0.2), 0.0, "closed", "red cube")]
    out = _execute("cubes", "pick", 0, kfs)
    assert out["status"] == "stopped" and out["success"] is False
    assert out["stopped"] == {"after": 3, "of": 4, "reason": "the gripper closed on nothing"}
    assert out["results"][-1].endswith("skipped") and len(out["results"]) == 4


@pytest.mark.assets
def test_stop_cancels_between_actions():
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 2

    out = _execute("cubes", "pick", 0, stop=stop)
    assert out["status"] == "cancelled" and out["success"] is False and len(out["results"]) >= 2


def test_execute_logs_into_the_plan_folder(tmp_path):
    ENVS.clear()
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK, "seed": 2}).json()["run_id"]
        wait_idle(c)
        with c.websocket_connect("/ws") as ws:
            assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
            msgs = []
            while not msgs or msgs[-1].get("kind") != "exec_end":
                msgs.append(ws.receive_json())
        wait_idle(c)
        summary = c.get("/api/runs").json()[0]
        replay = c.get(f"/api/runs/{run_id}").json()
    env = ENVS[-1]
    assert env.seed == 2 and env.instruction == PICK and len(ENVS) == 2  # a fresh env from the same scene and seed
    assert env.executed[0].yaw == 90.0  # the saved, clamped keyframes
    assert any(m["type"] == "frame" for m in msgs)
    kinds = [e["kind"] for e in events(tmp_path)]
    assert kinds[-3:] == ["exec_start", "step", "exec_end"]
    step, end = events(tmp_path)[-2], events(tmp_path)[-1]
    assert step["results"] == ["move_to(...): reached"] and step["actions"] == ["move_to(0.500, -0.100, 0.150)"]
    assert end["status"] == "success" and end["success"] is True and end["stopped"] is None
    assert summary["exec_status"] == "success" and summary["status"] == "planned"
    assert [e["kind"] for e in replay][-3:] == ["exec_start", "step", "exec_end"]


def test_execute_is_409_while_running_and_needs_same_origin(tmp_path):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
    with TestClient(create_app(HangingBackend(), plan_factory, tmp_path / "runs",
                               lessons_dir=tmp_path / "lessons")) as c:
        assert c.post(f"/api/plan/{run_id}/execute", headers={"Origin": "http://evil.example"}).status_code == 403
        assert c.post("/api/run", json={"instruction": PICK}).status_code == 200
        assert c.post(f"/api/plan/{run_id}/execute").status_code == 409
        c.post("/api/stop")
        wait_idle(c)


def test_execute_of_an_unknown_plan_is_404(tmp_path):
    with client(tmp_path, FakeBackend([])) as c:
        for run_id in ("20261017-000000", "%2E%2E", "nope", "20261016-181045%2F.."):
            assert c.post(f"/api/plan/{run_id}/execute").status_code == 404


def test_stop_during_an_execution(tmp_path):
    import threading

    gate = threading.Event()

    def factory(scene, task, instruction, seed, show_positions, on_frame):
        env = plan_factory(scene, task, instruction, seed, show_positions, on_frame)

        def execute_plan(keyframes, should_stop):
            while not should_stop():
                gate.wait(0.01)
            return {"actions": [], "results": [], "stopped": None, "success": False, "status": "cancelled"}

        env.execute_plan = execute_plan
        return env

    with TestClient(create_app(FakeBackend([good_plan()]), factory, tmp_path / "runs",
                               lessons_dir=tmp_path / "lessons")) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
        assert c.get("/api/state").json()["running"] is True
        assert c.post("/api/run", json={"instruction": PICK}).status_code == 409
        c.post("/api/stop")
        wait_idle(c)
    assert events(tmp_path)[-1]["kind"] == "exec_end" and events(tmp_path)[-1]["status"] == "cancelled"


@pytest.mark.assets
def test_a_real_execution_streams_frames_and_logs_the_check(tmp_path):
    from armmind.llm.fake import DEMO_PLAN
    from armmind.server.app import default_env_factory
    app = create_app(FakeBackend([good_plan(DEMO_PLAN)]), default_env_factory, tmp_path / "runs",
                     lessons_dir=tmp_path / "lessons")
    with TestClient(app) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK, "seed": 0}).json()["run_id"]
        wait_idle(c)
        with c.websocket_connect("/ws") as ws:
            assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
            frames, kinds = 0, []
            while "exec_end" not in kinds:
                m = ws.receive_json()
                frames += m["type"] == "frame"
                kinds += [m["kind"]] if m["type"] == "event" else []
        wait_idle(c, timeout=60)
    assert frames > 10 and kinds == ["exec_start", "step", "exec_end"]
    start, step, end = events(tmp_path)[-3:]
    assert start["check"] == "pick" and start["keyframes"] == 3
    assert step["step"] == 2 and len(step["actions"]) == 4
    assert end["status"] in ("success", "stopped", "failed") and isinstance(end["success"], bool)


def test_exec_start_comes_before_the_env_is_made(tmp_path):
    def factory(scene, task, instruction, seed, show_positions, on_frame):
        if factory.calls:
            raise RuntimeError("no renderer")
        factory.calls += 1
        return plan_factory(scene, task, instruction, seed, show_positions, on_frame)
    factory.calls = 0

    with TestClient(create_app(FakeBackend([good_plan()]), factory, tmp_path / "runs",
                               lessons_dir=tmp_path / "lessons")) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
        wait_idle(c)
    start, end = events(tmp_path)[-2:]
    assert start["kind"] == "exec_start" and start["scene"] == "cubes"
    assert end["kind"] == "exec_end" and end["status"] == "error" and "no renderer" in end["message"]


def test_execute_after_a_scene_change_uses_the_plan_scene(tmp_path):
    ENVS.clear()
    tray = SCENES["tray"].tasks["red_in_tray"].instruction
    with client(tmp_path, FakeBackend([good_plan(), good_plan()])) as c:
        run_id = c.post("/api/plan", json={"instruction": tray, "scene": "tray", "seed": 4}).json()["run_id"]
        wait_idle(c)
        assert c.get("/api/preview", params={"scene": "cubes", "seed": 9}).status_code in (200, 503)
        assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
        wait_idle(c)
    env = ENVS[-1]
    assert (env.scene_name, env.seed, env.check_name) == ("tray", 4, "red_in_tray")
    assert events(tmp_path)[-3]["scene"] == "tray"
