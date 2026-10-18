"""POST /api/plan, GET /api/plan/{id}/view and POST /api/plan/{id}/execute with fake parts."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from armmind.llm.fake import FakeBackend
from armmind.scenes import SCENES
from armmind.server.app import create_app
from tests.fakes import HangingBackend, bad, good
from tests.test_agent_plan import PlanEnv, good_plan
from tests.test_server import make_env, wait_idle

PICK = SCENES["cubes"].tasks["pick"].instruction


class ApiPlanEnv(PlanEnv):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.executed = None

    def execute_plan(self, keyframes, should_stop, on_action=None):
        self.executed = keyframes
        return {"actions": ["move_to(0.500, -0.100, 0.150)"], "results": ["move_to(...): reached"],
                "stopped": None, "success": True, "status": "success"}


ENVS: list = []


def plan_factory(scene, task, instruction, seed, show_positions, on_frame):
    on_frame("front", b"\xff\xd8jpeg")
    on_frame("top", b"\xff\xd8jpeg")
    env = make_env(scene, task, instruction, cls=ApiPlanEnv)
    env.seed, env.show_positions = seed, show_positions
    ENVS.append(env)
    return env


def client(tmp_path, backend):
    return TestClient(create_app(backend, plan_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons"))


def events(tmp_path):
    folder = next((tmp_path / "runs").iterdir())
    return [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines()]


def test_plan_writes_the_run_folder_events_and_frames(tmp_path):
    ENVS.clear()
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        with c.websocket_connect("/ws") as ws:
            r = c.post("/api/plan", json={"scene": "cubes", "seed": 3, "instruction": PICK, "use_lessons": False})
            assert r.status_code == 200
            msgs = []
            while not msgs or msgs[-1].get("kind") != "run_end":
                msgs.append(ws.receive_json())
        wait_idle(c)
        run_id = r.json()["run_id"]
        view = c.get(f"/api/plan/{run_id}/view")
        listed = c.get("/api/runs").json()
    assert [m["cam"] for m in msgs if m["type"] == "frame"] == ["front", "top"]
    kinds = [e["kind"] for e in events(tmp_path)]
    assert kinds == ["run_start", "observation", "ask", "plan", "run_end"]
    start = events(tmp_path)[0]
    assert start["mode"] == "plan" and start["seed"] == 3 and start["check"] == "pick" and start["run_id"] == run_id
    assert ENVS[-1].seed == 3 and ENVS[-1].show_positions is False  # never the true positions
    folder = tmp_path / "runs" / run_id
    assert (folder / "step001_front.png").read_bytes() == b"front"
    assert view.status_code == 200 and view.json() == {"believed": ["red_cube"], "marks": [{"ok": True, "checks": {}}]}
    assert listed[0]["mode"] == "plan" and listed[0]["status"] == "planned"


def test_plan_debug_switch_never_reaches_the_env(tmp_path):
    ENVS.clear()
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        assert c.post("/api/plan", json={"instruction": PICK, "show_positions": True}).status_code == 200
        wait_idle(c)
    assert ENVS[-1].show_positions is False


def test_plan_checks_like_a_run(tmp_path):
    with client(tmp_path, HangingBackend()) as c:
        assert c.post("/api/plan", json={"instruction": PICK, "scene": "nope"}).status_code == 404
        assert c.post("/api/plan", json={"instruction": ""}).status_code == 422
        evil = {"Origin": "http://evil.example"}
        assert c.post("/api/plan", json={"instruction": PICK}, headers=evil).status_code == 403
        assert c.post("/api/plan", json={"instruction": PICK}).status_code == 200
        assert c.post("/api/plan", json={"instruction": PICK}).status_code == 409
        assert c.post("/api/run", json={"instruction": PICK}).status_code == 409
        assert c.get("/api/state").json()["running"] is True
        c.post("/api/stop")  # Stop works for a plan
        wait_idle(c)
    assert events(tmp_path)[-1] == {**events(tmp_path)[-1], "kind": "run_end", "status": "stopped"}


def test_a_run_blocks_a_plan(tmp_path):
    with client(tmp_path, HangingBackend()) as c:
        assert c.post("/api/run", json={"instruction": PICK}).status_code == 200
        assert c.post("/api/plan", json={"instruction": PICK}).status_code == 409
        c.post("/api/stop")
        wait_idle(c)


def test_view_of_an_unknown_or_unfinished_plan_is_404(tmp_path):
    with client(tmp_path, FakeBackend([bad(), bad(), good(done=True)])) as c:
        assert c.post("/api/plan", json={"instruction": PICK}).status_code == 200
        wait_idle(c)
        failed = c.get("/api/runs").json()[0]["id"]
        time.sleep(1.05)  # a new folder name
        assert c.post("/api/run", json={"instruction": "Touch it."}).status_code == 200
        wait_idle(c)
        normal = c.get("/api/runs").json()[0]["id"]
        for run_id in (failed, normal, "20261017-000000", "..", "20261016-181045%2F.."):
            assert c.get(f"/api/plan/{run_id}/view").status_code == 404


@pytest.mark.parametrize("name", ["step001_front.png", "step001_top.png"])
def test_plan_frame_images_are_served_with_the_v18_rules(tmp_path, name):
    with client(tmp_path, FakeBackend([good_plan()])) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
        assert c.get(f"/api/runs/{run_id}/images/{name}").status_code == 200
        for bad_name in ("..%2Fevents.jsonl", "events.jsonl", "step001_front.png.bak", "step1_front.png"):
            assert c.get(f"/api/runs/{run_id}/images/{bad_name}").status_code == 404
        assert c.get(f"/api/plan/{run_id}%2F../view").status_code == 404


def test_plan_body_is_strict_about_its_fields(tmp_path):
    with client(tmp_path, FakeBackend([])) as c:
        assert c.post("/api/plan", json={"instruction": PICK, "seed": -1}).status_code == 422


def test_demo_backend_answers_a_plan(tmp_path):
    from armmind.llm.fake import DemoBackend
    with client(tmp_path, DemoBackend(delay_s=0.0)) as c:
        assert c.post("/api/plan", json={"instruction": PICK}).status_code == 200
        wait_idle(c)
    plan = next(e for e in events(tmp_path) if e["kind"] == "plan")
    assert plan["objects"] and plan["keyframes"] and plan["summary"]



@pytest.mark.assets
def test_a_real_plan_has_scores_a_view_and_png_images(tmp_path):
    from armmind.llm.fake import DEMO_PLAN
    from armmind.server.app import default_env_factory
    app = create_app(FakeBackend([good_plan(DEMO_PLAN)]), default_env_factory, tmp_path / "runs",
                     lessons_dir=tmp_path / "lessons")
    with TestClient(app) as c:
        with c.websocket_connect("/ws") as ws:
            run_id = c.post("/api/plan", json={"instruction": PICK, "seed": 0}).json()["run_id"]
            msgs = []
            while not msgs or msgs[-1].get("kind") != "run_end":
                msgs.append(ws.receive_json())
        wait_idle(c)
        view = c.get(f"/api/plan/{run_id}/view").json()
    plan = next(e for e in events(tmp_path) if e["kind"] == "plan")
    assert events(tmp_path)[-1]["status"] == "planned"
    assert any(m["type"] == "frame" for m in msgs)
    assert (tmp_path / "runs" / run_id / "step001_front.png").read_bytes()[:4] == b"\x89PNG"
    rows = {r["true_name"]: r for r in plan["scores"]["perception"]}
    assert set(rows) == {"red_cube", "blue_cube", "green_cube"}  # the labels match
    assert rows["red_cube"]["xy_cm"] > 0 and len(plan["scores"]["marks"]) == 3
    assert view == plan["view"] and [c["name"] for c in view["cameras"]] == ["front", "top"]
    assert view["track"]["keyframes"][1]["carrying"] == "red cube"
