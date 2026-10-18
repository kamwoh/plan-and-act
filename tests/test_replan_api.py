"""POST /api/plan/{id}/replan with fake parts."""
import json
import time

from fastapi.testclient import TestClient

from armmind.llm.fake import FakeBackend
from armmind.server.app import create_app
from tests.fakes import HangingBackend, judged
from tests.test_agent_plan import PLAN, good_plan
from tests.test_plan_api import ENVS, PICK, ApiPlanEnv, client
from tests.test_server import make_env, wait_idle

TRUE_XYZ = [0.5871, -0.1234, 0.0213]  # the true cube, 9 cm from the belief


def run_events(tmp_path, run_id):
    return [json.loads(x) for x in (tmp_path / "runs" / run_id / "events.jsonl").read_text().splitlines()]


class TrueStateEnv(ApiPlanEnv):
    """Its scores, its view and its execution are full of true values that the model must never see."""

    def check_plan(self, belief, keyframes):
        return {"perception": [{"name": "red_cube", "true_name": "red_cube", "xy_cm": 7.31, "z_cm": 0.13,
                                "size_cm": [0.0, 0.0, 0.0], "yaw_deg": 4.4}],
                "missing": ["blue_cube"], "invented": [], "summary": {"mean_xy_cm": 7.31, "max_xy_cm": 7.31},
                "marks": [{"ok": False, "checks": {"grasp": "missed by 7.3 cm"}} for _ in keyframes]}

    def plan_view(self, belief, keyframes, marks=None):
        return {"believed": [b.name for b in belief], "marks": marks,
                "true": [{"name": "red_cube", "xyz": TRUE_XYZ, "yaw": 33.3}]}

    def execute_plan(self, keyframes, should_stop, on_action=None):
        true = {"red_cube": {"xyz": TRUE_XYZ, "yaw": 33.3}}
        return {"actions": ["move_to(0.500, -0.100, 0.150)", "close"],
                "results": ["move_to(0.500, -0.100, 0.150): reached, gripper at (0.587, -0.123, 0.150)",
                            "close: reached"],
                "stopped": {"after": 2, "of": 2, "reason": "the gripper closed on nothing"},
                "success": False, "status": "stopped", "true_start": true, "true_after": [true]}


def true_factory(scene, task, instruction, seed, show_positions, on_frame):
    env = make_env(scene, task, instruction, cls=TrueStateEnv)
    env.seed = seed
    ENVS.append(env)
    return env


def planned_and_executed(c):
    run_id = c.post("/api/plan", json={"instruction": PICK, "seed": 5, "use_lessons": True}).json()["run_id"]
    wait_idle(c)
    assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
    wait_idle(c)
    return run_id


def test_execute_saves_the_end_images(tmp_path):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        run_id = planned_and_executed(c)
        end = [e for e in run_events(tmp_path, run_id) if e["kind"] == "exec_end"][-1]
        assert end["step"] == 2
        assert (end["front_png"], end["top_png"]) == ("step002_front.png", "step002_top.png")
        assert (tmp_path / "runs" / run_id / "step002_front.png").read_bytes() == b"front"
        assert c.get(f"/api/runs/{run_id}/images/step002_top.png").status_code == 200


def test_replan_makes_a_new_round_with_the_feedback(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan()])
    with TestClient(create_app(backend, true_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        first = planned_and_executed(c)
        r = c.post(f"/api/plan/{first}/replan", json={"mode": "start"})
        assert r.status_code == 200
        wait_idle(c)
        second = r.json()["run_id"]
        listed = {x["id"]: x for x in c.get("/api/runs").json()}
    assert second != first
    start = run_events(tmp_path, second)[0]
    assert start["mode"] == "plan" and start["parent"] == first and start["round"] == 2 and start["end_step"] == 2
    assert start["seed"] == 5 and start["instruction"] == PICK and start["use_lessons"] is True
    assert ENVS[-1].seed == 5
    assert run_events(tmp_path, second)[-1]["status"] == "planned"
    assert listed[second]["round"] == 2 and listed[second]["parent"] == first
    assert listed[first]["round"] == 1 and listed[first]["parent"] is None
    msg = backend.sent[-1][0]
    assert msg.images == (b"front", b"top", b"front", b"top")
    text = msg.text
    assert "Your last plan" in text and '"note":"up"' in text
    assert "stopped" in text and "action 2 of 2 (close)" in text and "the gripper closed on nothing" in text
    for secret in ("0.587", "0.5871", "-0.123", "0.0213", "33.3", "7.31", "7.3", "0.13", "4.4", "missed by",
                   "blue_cube", "prediction", "perception", "debug"):
        assert secret not in text, secret


def test_replan_of_an_unknown_run_or_a_plan_without_an_execution_is_404(tmp_path):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        for run_id in ("20261017-000000", "nope", "%2E%2E"):
            assert c.post(f"/api/plan/{run_id}/replan", json={"mode": "start"}).status_code == 404
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
        r = c.post(f"/api/plan/{run_id}/replan", json={"mode": "start"})
        assert r.status_code == 404 and "execut" in r.json()["detail"]


def test_replan_during_a_run_is_409_and_needs_same_origin(tmp_path):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        run_id = planned_and_executed(c)
    with TestClient(create_app(HangingBackend(), true_factory, tmp_path / "runs",
                               lessons_dir=tmp_path / "lessons")) as c:
        assert c.post(f"/api/plan/{run_id}/replan", headers={"Origin": "http://evil.example"}).status_code == 403
        assert c.post("/api/plan", json={"instruction": PICK}).status_code == 200
        assert c.post(f"/api/plan/{run_id}/replan", json={"mode": "start"}).status_code == 409
        c.post("/api/stop")
        wait_idle(c)


def test_replan_of_round_5_is_409(tmp_path):
    backend = FakeBackend([r for _ in range(5) for r in (good_plan(), judged())])
    with client(tmp_path, backend) as c:
        run_id = planned_and_executed(c)
        for n in range(2, 6):
            r = c.post(f"/api/plan/{run_id}/replan", json={"mode": "start"})
            assert r.status_code == 200, r.text
            wait_idle(c)
            run_id = r.json()["run_id"]
            assert run_events(tmp_path, run_id)[0]["round"] == n
            if n == 5:  # the round comes before the missing execution
                r = c.post(f"/api/plan/{run_id}/replan", json={"mode": "start"})
                assert r.status_code == 409 and "5" in r.json()["detail"]
            assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
            wait_idle(c)
        r = c.post(f"/api/plan/{run_id}/replan", json={"mode": "start"})
        assert r.status_code == 409 and "5" in r.json()["detail"]


def test_replan_lessons_reach_the_store(tmp_path):
    body = {**PLAN, "lessons": ["Closing on the near edge slipped."]}
    with client(tmp_path, FakeBackend([good_plan(), judged(), good_plan(body)])) as c:
        first = planned_and_executed(c)
        second = c.post(f"/api/plan/{first}/replan", json={"mode": "start"}).json()["run_id"]
        wait_idle(c)
        saved = c.get("/api/lessons", params={"scene": "cubes"}).json()
    assert [(s["text"], s["run"], s["outcome"]) for s in saved] == [
        ("Closing on the near edge slipped.", second, "success")]


def test_replan_after_a_stopped_execution_says_the_person_stopped_it(tmp_path):
    import threading

    gate = threading.Event()

    def factory(scene, task, instruction, seed, show_positions, on_frame):
        env = true_factory(scene, task, instruction, seed, show_positions, on_frame)

        def execute_plan(keyframes, should_stop, on_action=None):
            for i in range(3):
                on_action(i, 7)
            while not should_stop():
                gate.wait(0.01)
            return {"actions": ["a"] * 7, "results": ["r"] * 3, "stopped": None, "success": False,
                    "status": "cancelled"}

        env.execute_plan = execute_plan
        return env

    backend = FakeBackend([good_plan(), good_plan()])
    with TestClient(create_app(backend, factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
        while sum(e["kind"] == "action" for e in run_events(tmp_path, run_id)) < 3:
            time.sleep(0.01)
        c.post("/api/stop")
        wait_idle(c)
        assert c.post(f"/api/plan/{run_id}/replan", json={"mode": "start"}).status_code == 200
        wait_idle(c)
    msg = backend.sent[-1][0]
    assert "The person stopped the execution after action 3 of 7." in msg.text
    assert "automatic check" not in msg.text and "All actions ran" not in msg.text
    assert len(msg.images) == 4  # the end images exist, so they go
