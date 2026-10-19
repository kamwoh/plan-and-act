"""v1.16: "Run per round". Execute runs only the first K keyframes, and a re-plan continues from there."""
import pytest
from fastapi.testclient import TestClient

from armmind.llm.fake import FakeBackend
from armmind.server.app import create_app
from armmind.sim.state import END_STATE, START_STATE
from tests.fakes import judged
from tests.test_agent_plan import PLAN, good_plan
from tests.test_auto_loop import events, folders, loop_end
from tests.test_continue_api import StateEnv
from tests.test_page_plan import _body, _web
from tests.test_plan_api import PICK, fake_frames
from tests.test_replan_api import run_events
from tests.test_server import make_env, wait_idle

PLAN5 = {**PLAN, "keyframes": [{"xyz": [0.5, -0.1, z], "yaw": 0, "gripper": "open", "holding": "nothing",
                                 "note": f"step {k}"} for k, z in enumerate((0.3, 0.25, 0.2, 0.15, 0.1), 1)]}
RAN: list = []  # the keyframe count of each execution


class KEnv(StateEnv):
    stop_in: int | None = None

    def execute_plan(self, keyframes, should_stop, on_action=None):
        StateEnv.executions += 1
        RAN.append(len(keyframes))
        stop = KEnv.stop_in
        true = {"red_cube": {"xyz": [0.5, -0.1, 0.02], "yaw": 0.0}}
        return {"actions": ["a"] * len(keyframes), "results": ["r"] * len(keyframes),
                "stopped": {"after": 1, "of": len(keyframes), "reason": "blocked"} if stop else None,
                "success": False, "status": "stopped" if stop else "failed", "true_start": true,
                "true_after": [true] * (stop or len(keyframes)), "frames": fake_frames(keyframes, stop)}


def factory(scene, task, instruction, seed, show_positions, on_frame, start=None):
    env = make_env(scene, task, instruction, cls=KEnv)
    env.seed, env.start = seed, start
    return env


@pytest.fixture(autouse=True)
def _clear():
    RAN.clear()
    KEnv.stop_in = None
    StateEnv.executions = 0


def client(tmp_path, backend):
    return TestClient(create_app(backend, factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons"))


def execute(c, k):
    run_id = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
    wait_idle(c)
    r = c.post(f"/api/plan/{run_id}/execute", json={"max_keyframes": k} if k is not None else None)
    assert r.status_code == 200, r.text
    wait_idle(c)
    return run_id


def kinds(evs):
    return [e["kind"] for e in evs]


def test_k_0_runs_all_keyframes_as_before(tmp_path):
    for k in (None, 0, 5, 9):  # no body, 0, and K not below the keyframe count: the whole plan
        RAN.clear()
        with client(tmp_path, FakeBackend([good_plan(PLAN5), judged()])) as c:
            run_id = execute(c, k)
        evs = run_events(tmp_path, run_id)
        end = next(e for e in evs if e["kind"] == "exec_end")
        assert RAN == [5] and end["status"] == "failed" and "verdict" in kinds(evs), k


def test_k_2_runs_2_keyframes_and_a_replan_continues_there(tmp_path):
    backend = FakeBackend([good_plan(PLAN5), good_plan(PLAN5)])
    with client(tmp_path, backend) as c:
        first = execute(c, 2)
        evs = run_events(tmp_path, first)
        start = next(e for e in evs if e["kind"] == "exec_start")
        end = next(e for e in evs if e["kind"] == "exec_end")
        assert RAN == [2] and end["status"] == "partial" and end["stopped"] is None
        assert (end["ran_keyframes"], end["keyframes"]) == (2, 5)
        assert (start["keyframes"], start["plan_keyframes"]) == (2, 5) and len(start["action_data"]) == 2
        assert [(f["keyframe"], f["of"]) for f in end["frames"]] == [(1, 5), (2, 5)]
        assert [f["sent"] for f in end["frames"]] == [True, False]  # the end images show keyframe 2
        assert "verdict" not in kinds(evs) and "verdict_start" not in kinds(evs)  # the next re-plan judges it
        assert (tmp_path / "runs" / first / END_STATE).is_file()
        r = c.post(f"/api/plan/{first}/replan", json={"mode": "continue"})
        assert r.status_code == 200, r.text
        wait_idle(c)
        second = r.json()["run_id"]
    assert run_events(tmp_path, second)[0]["round_mode"] == "continue"
    assert (tmp_path / "runs" / second / START_STATE).is_file()
    text = backend.sent[-1][0].text
    assert "Execution status: partial." in text and "All actions ran." not in text
    assert ("Only the first 2 of your 5 keyframes ran, by design. The arm is now at the end of keyframe 2."
            in text) and "did not reset" in text


def test_a_replan_from_the_start_after_a_partial_run_still_continues(tmp_path):
    with client(tmp_path, FakeBackend([good_plan(PLAN5), good_plan(PLAN5)])) as c:
        first = execute(c, 2)
        r = c.post(f"/api/plan/{first}/replan", json={"mode": "start"})
        assert r.status_code == 200, r.text
        wait_idle(c)
    assert run_events(tmp_path, r.json()["run_id"])[0]["round_mode"] == "continue"


def test_a_stop_inside_the_k_keyframes_is_a_stop(tmp_path):
    KEnv.stop_in = 1
    with client(tmp_path, FakeBackend([good_plan(PLAN5), judged()])) as c:
        run_id = execute(c, 2)
    evs = run_events(tmp_path, run_id)
    assert next(e for e in evs if e["kind"] == "exec_end")["status"] == "stopped" and "verdict" in kinds(evs)


def test_mode_start_with_k_is_409(tmp_path):
    with client(tmp_path, FakeBackend([good_plan(PLAN5)])) as c:
        first = execute(c, None)
        r = c.post(f"/api/plan/{first}/replan", json={"mode": "start", "max_keyframes": 2})
        assert r.status_code == 409 and "continue" in r.json()["detail"]
        r = c.post("/api/plan", json={"instruction": PICK, "auto": True, "mode": "start", "max_keyframes": 2})
        assert r.status_code == 409 and "continue" in r.json()["detail"]
        assert c.post(f"/api/plan/{first}/execute", json={"max_keyframes": 31}).status_code == 422


def test_the_auto_loop_runs_k_keyframes_per_round(tmp_path):
    backend = FakeBackend([good_plan(PLAN5), good_plan(PLAN5), judged()])
    with client(tmp_path, backend) as c:
        r = c.post("/api/plan", json={"instruction": PICK, "seed": 5, "auto": True, "rounds": 2, "mode": "continue",
                                      "max_keyframes": 2})
        assert r.status_code == 200, r.text
        wait_idle(c, 10)
    rounds = [events(f) for f in folders(tmp_path)]
    assert RAN == [2, 2]
    ends = [e["status"] for r in rounds for e in r if e["kind"] == "exec_end"]
    assert ends == ["partial", "partial"] and rounds[1][0]["round_mode"] == "continue"
    end = loop_end(rounds)
    assert [v["round"] for v in end["verdicts"]] == [1, 2]  # the re-plan judges round 1, the verdict call round 2
    assert "Only the first 2 of your 5 keyframes ran" in backend.sent[1][0].text


# ---- The page ----
def test_the_page_has_the_field_and_locks_continue():
    html, js = _web("index.html"), _web("app.js")
    assert 'id="per_round" type="number" value="0" min="0" max="30"' in html
    lock = _body(js, "function lockContinue()")
    assert '$("continue").checked = true;' in lock and '$("continue").disabled = locked;' in lock
    assert "max_keyframes: perRound()" in _body(js, "async function executePlan()")
    assert "max_keyframes: perRound()" in js[js.index('$("run").addEventListener("submit"'):]


def test_the_page_shows_a_partial_run():
    js = _web("app.js")
    badge = _body(js, "function outcomeBadge(ev)")
    assert '`◐ RAN ${ev.ran_keyframes}/${ev.keyframes} KEYFRAMES`' in badge and 'cls: "info"' in badge
    assert ".out-info { color: var(--accent)" in _web("style.css")
