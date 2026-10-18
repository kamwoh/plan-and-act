"""The Auto loop: plan, execute, then re-plan until the auto-check passes or the rounds end."""
import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from armmind.llm.fake import FakeBackend
from armmind.server.app import create_app
from tests.fakes import bad
from tests.test_agent_plan import good_plan
from tests.test_plan_api import ENVS, PICK, ApiPlanEnv
from tests.test_server import make_env, wait_idle

OUTCOMES: list[str] = []  # the status of each execution, in order. An empty list gives "failed"
GATE = threading.Event()  # an execution with status "hang" waits for Stop


class LoopEnv(ApiPlanEnv):
    def execute_plan(self, keyframes, should_stop, on_action=None):
        status = OUTCOMES.pop(0) if OUTCOMES else "failed"
        if status == "hang":
            while not should_stop():
                GATE.wait(0.01)
            status = "cancelled"
        return {"actions": ["move_to(0.500, -0.100, 0.150)", "close"], "results": ["r1", "r2"],
                "stopped": None, "success": status == "success", "status": status}


def loop_factory(scene, task, instruction, seed, show_positions, on_frame):
    env = make_env(scene, task, instruction, cls=LoopEnv)
    env.seed = seed
    ENVS.append(env)
    return env


def loop_client(tmp_path, backend):
    return TestClient(create_app(backend, loop_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons"))


def folders(tmp_path):
    return sorted((tmp_path / "runs").iterdir(), key=lambda p: (p.name[:15], int(p.name[16:] or 0)))


def events(folder):
    return [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines()]


def run_loop(tmp_path, backend, outcomes, rounds=3, instruction=PICK, timeout=10.0):
    OUTCOMES[:] = outcomes
    with loop_client(tmp_path, backend) as c:
        r = c.post("/api/plan", json={"instruction": instruction, "seed": 7, "auto": True, "rounds": rounds})
        assert r.status_code == 200, r.text
        wait_idle(c, timeout)
        first = r.json()["run_id"]
    return first, [events(f) for f in folders(tmp_path)]


def loop_end(rounds):
    ends = [e for evs in rounds for e in evs if e["kind"] == "loop_end"]
    assert len(ends) == 1 and rounds[-1][-1] == ends[0]  # only in the last round, as its last event
    return ends[0]


def test_the_loop_stops_at_success(tmp_path):
    first, rounds = run_loop(tmp_path, FakeBackend([good_plan(), good_plan(), good_plan()]), ["failed", "success"])
    assert len(rounds) == 2  # each round has its own run folder
    starts = [r[0] for r in rounds]
    assert [s["round"] for s in starts] == [1, 2] and starts[0]["run_id"] == first
    assert starts[0]["parent"] is None and starts[1]["parent"] == first
    assert all(s["seed"] == 7 and s["auto_rounds"] == 3 for s in starts)
    for r in rounds:
        kinds = [e["kind"] for e in r]
        assert kinds[:5] == ["run_start", "observation", "ask", "plan", "run_end"]
        assert "exec_start" in kinds and "exec_end" in kinds
    assert loop_end(rounds) | {"t": 0} == {"t": 0, "kind": "loop_end", "rounds_run": 2, "rounds_max": 3,
                                          "success": True, "reason": "the auto-check passed"}


def test_the_loop_stops_after_the_rounds(tmp_path):
    _, rounds = run_loop(tmp_path, FakeBackend([good_plan(), good_plan(), good_plan()]), ["failed", "stopped"],
                         rounds=2)
    assert len(rounds) == 2
    end = loop_end(rounds)
    assert (end["rounds_run"], end["rounds_max"], end["success"]) == (2, 2, False) and "2 rounds" in end["reason"]


def test_the_loop_stops_on_a_bad_reply(tmp_path):
    _, rounds = run_loop(tmp_path, FakeBackend([good_plan(), bad(), bad()]), ["failed"])
    assert len(rounds) == 2
    assert [e["kind"] for e in rounds[1]][-3:] == ["bad_reply", "run_end", "loop_end"]
    end = loop_end(rounds)
    assert end["rounds_run"] == 2 and end["success"] is False and "bad_reply" in end["reason"]


def test_the_loop_ends_after_round_1_without_an_auto_check(tmp_path):
    _, rounds = run_loop(tmp_path, FakeBackend([good_plan(), good_plan()]), ["done"], instruction="Wave at me.")
    assert len(rounds) == 1
    end = loop_end(rounds)
    assert end["rounds_run"] == 1 and end["success"] is False and "no automatic check" in end["reason"]
    assert "exec_end" in [e["kind"] for e in rounds[0]]


def test_stop_ends_the_loop(tmp_path):
    OUTCOMES[:] = ["failed", "hang"]
    with loop_client(tmp_path, FakeBackend([good_plan(), good_plan(), good_plan()])) as c:
        r = c.post("/api/plan", json={"instruction": PICK, "auto": True, "rounds": 3})
        assert r.status_code == 200
        end = time.monotonic() + 5
        while len(list((tmp_path / "runs").iterdir())) < 2 or "exec_start" not in \
                [e["kind"] for e in events(folders(tmp_path)[-1])]:
            assert time.monotonic() < end
            time.sleep(0.02)
        assert c.get("/api/state").json()["running"] is True
        assert c.post("/api/plan", json={"instruction": PICK}).status_code == 409  # one job holds the loop
        assert c.post("/api/stop").status_code == 200
        wait_idle(c)
    rounds = [events(f) for f in folders(tmp_path)]
    assert len(rounds) == 2
    end = loop_end(rounds)
    assert end["rounds_run"] == 2 and end["success"] is False and end["reason"] == "stopped"
    assert next(e for e in rounds[1] if e["kind"] == "exec_end")["status"] == "cancelled"


def test_state_follows_the_round(tmp_path):
    OUTCOMES[:] = ["failed", "hang"]
    with loop_client(tmp_path, FakeBackend([good_plan(), good_plan()])) as c:
        c.post("/api/plan", json={"instruction": PICK, "auto": True, "rounds": 2})
        end = time.monotonic() + 5
        while len(list((tmp_path / "runs").iterdir())) < 2:
            assert time.monotonic() < end
            time.sleep(0.02)
        time.sleep(0.1)
        active = [r for r in c.get("/api/runs").json() if r["active"]]
        assert [r["round"] for r in active] == [2]
        assert c.get("/api/state").json()["auto"] is True  # a page that loads now knows the loop
        c.post("/api/stop")
        wait_idle(c)
        assert c.get("/api/state").json()["auto"] is False


def test_exec_start_has_the_round(tmp_path):
    _, rounds = run_loop(tmp_path, FakeBackend([good_plan(), good_plan()]), ["failed", "failed"], rounds=2)
    assert [next(e for e in r if e["kind"] == "exec_start")["round"] for r in rounds] == [1, 2]


def test_rounds_must_be_1_to_5(tmp_path):
    with loop_client(tmp_path, FakeBackend([])) as c:
        for rounds in (0, 6, "x"):
            assert c.post("/api/plan", json={"instruction": PICK, "auto": True, "rounds": rounds}).status_code == 422


def test_without_auto_a_plan_is_one_round(tmp_path):
    with loop_client(tmp_path, FakeBackend([good_plan()])) as c:
        c.post("/api/plan", json={"instruction": PICK, "rounds": 5})
        wait_idle(c)
    (only,) = [events(f) for f in folders(tmp_path)]
    assert only[0]["auto_rounds"] is None and [e["kind"] for e in only][-1] == "run_end"


@pytest.mark.assets
def test_the_demo_loop_fails_round_1_and_solves_round_2(tmp_path):
    from armmind.llm.fake import DemoBackend
    from armmind.server.app import default_env_factory
    app = create_app(DemoBackend(delay_s=0.0), default_env_factory, tmp_path / "runs",
                     lessons_dir=tmp_path / "lessons")
    app.state.runner.auto_view_s = 0.0  # no wait for the imagined video
    with TestClient(app) as c:
        r = c.post("/api/plan", json={"instruction": PICK, "seed": 0, "auto": True, "rounds": 3})
        assert r.status_code == 200
        wait_idle(c, timeout=180)
        saved = c.get("/api/lessons", params={"scene": "cubes"}).json()
    rounds = [events(f) for f in folders(tmp_path)]
    ends = [next(e for e in evs if e["kind"] == "exec_end") for evs in rounds]
    assert [e["status"] for e in ends] == ["failed", "success"]
    assert loop_end(rounds)["success"] is True and loop_end(rounds)["rounds_run"] == 2
    assert [(s["run"], s["outcome"]) for s in saved] == [(rounds[1][0]["run_id"], "failed")]
