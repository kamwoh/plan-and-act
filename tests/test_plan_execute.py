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
    from armmind.env import keyframe_ends
    assert keyframe_ends(kfs) == [0, 2, 3, 5, 7]  # the last action of each keyframe


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

        def execute_plan(keyframes, should_stop, on_action=None):
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
    assert frames > 10 and kinds[0] == "exec_start" and kinds[-2:] == ["step", "exec_end"]
    actions = [e for e in events(tmp_path) if e["kind"] == "action"]
    start, step, end = [e for e in events(tmp_path) if e["kind"] != "action"][-3:]
    assert [(a["step"], a["index"], a["of"]) for a in actions] == [(2, i, 10) for i in range(len(actions))]
    assert kinds[1:-2] == ["action"] * len(actions) and 1 <= len(actions) <= 10  # a stop leaves out the rest
    assert start["check"] == "pick" and start["keyframes"] == 7 and start["step"] == 2
    assert [a["type"] for a in start["action_data"]] == ["move_to", "close", "move_to", "move_to", "move_to",
                                                         "move_to", "open", "move_to", "close", "move_to"]
    assert step["step"] == 2 and len(step["actions"]) == 10
    assert end["status"] in ("success", "stopped", "failed") and isinstance(end["success"], bool)
    pred = end["prediction"]  # kept in the run log
    assert {r["true_name"] for r in pred["rows"]} == {"red_cube", "blue_cube", "green_cube"}
    assert end["status"] == "success" and pred["summary"]["ran"] == 7 and pred["summary"]["partial"] is False
    line = pred["keyframes"][0]  # the demo plan of seed 0 runs to the end
    assert (line["keyframe"], line["name"], line["true_name"]) == (3, "green cube", "green_cube")
    assert line["disp_cm"] < 2.0 and pred["summary"]["agree"] == 3 and pred["unmatched"] == []


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


# ---- v1.11: a scripted push and its prediction score ----
PUSH_START_M = 0.06  # the closed fingers start 6 cm from the cube centre, on the far side of the push
PUSH_TRAVEL_M = 0.12  # then they travel 12 cm along y, so the cube goes about 9 cm
PUSH_Z = 0.02
FINGER_HALF_M = 0.0105  # half the depth of the closed fingers along y


def push_plan(world, name="red_cube"):
    """Keyframes that push the cube along y toward the middle of the table, and the predicted end pose."""
    from armmind.plan.formats import Move
    x, y, z = (float(v) for v in world.object_pos(name))
    sign = 1.0 if y < 0 else -1.0
    y0 = y - sign * PUSH_START_M
    y1 = y0 + sign * PUSH_TRAVEL_M
    end = (x, y1 + sign * (FINGER_HALF_M + 0.02), z)  # the cube face rests on the finger face
    believed = name.replace("_", " ")
    return [Keyframe((x, y0, 0.15), 0.0, "closed", None, "above, fingers closed"),
            Keyframe((x, y0, PUSH_Z), 0.0, "closed", None, "down beside the cube"),
            Keyframe((x, y1, PUSH_Z), 0.0, "closed", None, "push", (Move(believed, end, 0.0),)),
            Keyframe((x, y1, 0.15), 0.0, "closed", None, "up")]


@pytest.mark.assets
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_a_scripted_push_moves_the_cube_as_predicted(seed):
    from armmind.env import MujocoEnv
    from armmind.plan.formats import Entity
    from armmind.plan.scores import prediction
    from armmind.sim.planview import true_entities
    sc = SCENES["cubes"]
    env = MujocoEnv(sc, None, seed, "push the red cube")
    try:
        belief = tuple(Entity(e.name.replace("_", " "), e.shape, e.size, e.xyz, e.yaw) for e in true_entities(env.world))
        kfs = tuple(push_plan(env.world))
        checks = env.check_plan(belief, kfs)
        view = env.plan_view(belief, kfs, checks["marks"])
        out = env.execute_plan(kfs, lambda: False)
    finally:
        env.close()
    assert all(m["ok"] for m in checks["marks"]), checks["marks"]
    assert out["status"] == "done" and out["stopped"] is None and len(out["true_after"]) == 4
    real = out["true_after"][-1]["red_cube"]["xyz"][1] - out["true_start"]["red_cube"]["xyz"][1]
    assert abs(real) > 0.08  # the physics moved the cube
    s = prediction(view["track"], view["believed"], out["true_start"], out["true_after"], view["true_match"])
    rows = {r["true_name"]: r for r in s["rows"]}
    assert rows["red_cube"]["disp_cm"] < 2.0 and rows["red_cube"]["final_xy_cm"] < 2.0, rows["red_cube"]
    assert s["keyframes"][0]["keyframe"] == 3 and s["keyframes"][0]["disp_cm"] < 2.0
    assert s["summary"]["agree"] == 3 and s["summary"]["partial"] is False  # the other cubes stay still


@pytest.mark.assets
def test_a_stopped_execution_has_the_poses_of_the_keyframes_that_ran():
    def kfs(env):
        return [Keyframe((0.35, 0.0, 0.15), 0.0, "open", None), Keyframe((0.35, 0.0, 0.02), 0.0, "closed", "red cube"),
                Keyframe((0.35, 0.0, 0.2), 0.0, "closed", "red cube")]
    out = _execute("cubes", "pick", 0, kfs)
    assert out["status"] == "stopped" and len(out["true_after"]) == 2  # keyframe 2 ran its close, then it stopped
    assert set(out["true_start"]) == {"red_cube", "blue_cube", "green_cube"}


def test_the_prediction_of_an_old_or_broken_view(caplog):
    from armmind.server.app import _prediction
    out = {"true_start": {"red_cube": {"xyz": [0.5, 0.0, 0.02], "yaw": 0.0}},
           "true_after": [{"red_cube": {"xyz": [0.5, 0.05, 0.02], "yaw": 0.0}}]}
    track = {"start": {"xyz": [0.3, 0.0, 0.4], "yaw": 0.0}, "duration": 1.5,
             "keyframes": [{"xyz": [0.5, 0.0, 0.1], "yaw": 0.0, "carrying": None, "t": 1.0, "t_done": 1.5}]}
    believed = [{"name": "red cube", "xyz": [0.5, 0.0, 0.02], "yaw": 0.0}]
    old = _prediction({"track": track, "believed": believed}, out)  # no true_match and no moves
    assert old["rows"] == [] and old["unmatched"] == [{"true_name": "red_cube", "moved_cm": 5.0}]
    assert _prediction({"track": track, "believed": believed, "true_match": {"red_cube": "red cube"}},
                       out)["rows"][0]["real_cm"] == [0.0, 5.0, 0.0]
    assert _prediction(None, out) is None and _prediction({"cameras": []}, out) is None
    assert _prediction({"track": track, "believed": believed}, {"status": "done"}) is None
    assert caplog.records == []
    assert _prediction({"track": {"keyframes": "bad"}, "believed": believed}, out) is None
    assert "cannot score the prediction" in caplog.text and "Traceback" in caplog.text


@pytest.mark.assets
def test_a_cancel_inside_a_keyframe_leaves_it_out_of_the_poses():
    from armmind.agent.parse import parse_plan
    from armmind.llm.fake import DEMO_PLAN
    calls = []

    def stop():  # the first keyframe is a move and a close: cancel between the two
        calls.append(1)
        return len(calls) > 1

    out = _execute("cubes", "pick", 0, lambda env: list(parse_plan(DEMO_PLAN).keyframes), stop)
    assert out["status"] == "cancelled" and len(out["results"]) == 1 and out["true_after"] == []
    calls.clear()
    stop3 = lambda: (calls.append(1), len(calls) > 3)[1]
    out = _execute("cubes", "pick", 0, lambda env: list(parse_plan(DEMO_PLAN).keyframes), stop3)
    assert len(out["results"]) == 3 and len(out["true_after"]) == 2


@pytest.mark.assets
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("toward_mm", [2, 4, 6, -6])
def test_a_small_belief_error_on_a_push_passes_every_check(seed, toward_mm):
    """The true cube starts toward_mm closer to the hand than the belief (negative: further away)."""
    from armmind.env import MujocoEnv
    from armmind.plan.formats import Entity
    from armmind.sim.planview import true_entities
    env = MujocoEnv(SCENES["cubes"], None, seed, "push the red cube")
    try:
        kfs = tuple(push_plan(env.world))
        sign = 1.0 if env.world.object_pos("red_cube")[1] < 0 else -1.0
        belief = []
        for e in true_entities(env.world):
            xyz = (e.xyz[0], e.xyz[1] + sign * toward_mm / 1000, e.xyz[2]) if e.name == "red_cube" else e.xyz
            belief.append(Entity(e.name.replace("_", " "), e.shape, e.size, xyz, e.yaw))
        marks = env.check_plan(tuple(belief), kfs)["marks"]
    finally:
        env.close()
    assert [m["checks"] for m in marks if not m["ok"]] == []
