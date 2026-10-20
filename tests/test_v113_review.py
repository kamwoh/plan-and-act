"""The v1.13 review follow-ups."""
import asyncio

from fastapi.testclient import TestClient

from plan_and_act.llm.fake import FakeBackend
from plan_and_act.server.app import create_app
from plan_and_act.sim.state import END_STATE
from tests.fakes import bad, judged
from tests.test_agent_plan import good_plan
from tests.test_auto_loop import loop_end
from tests.test_continue_api import StateEnv, plan_and_execute
from tests.test_continue_api import factory as state_factory
from tests.test_done_verdict import DONE, FREE, run_loop, verdict
from tests.test_page_plan import _body, _web
from tests.test_plan_api import PICK
from tests.test_replan_api import run_events
from tests.test_server import wait_idle


# 1. A failed re-plan still judges the execution before it.
def test_a_failed_replan_in_a_free_text_loop_judges_the_last_execution(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), bad(), bad(), verdict(True)], ["done"], FREE)
    assert len(rounds) == 2
    end = loop_end(rounds)
    assert (end["success"], end["reason"]) == (True, "Claude says the task is done")
    assert [(v["round"], v["claude"], v["check"]) for v in end["verdicts"]] == [(1, True, None)]
    assert next(e for e in rounds[0] if e["kind"] == "verdict")["done"] is True  # in round 1's folder


def test_a_failed_replan_with_an_auto_check_still_counts_the_verdict(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), bad(), bad(), verdict(False)], ["failed"], PICK)
    end = loop_end(rounds)
    assert (end["success"], end["reason"]) == (False, "the plan ended with bad_reply")
    assert [(v["claude"], v["check"]) for v in end["verdicts"]] == [(False, False)]
    assert (end["agree"], end["compared"]) == (1, 1)


# 2. An Execute whose env fails to start removes the old end state.
def test_a_failed_env_start_leaves_no_old_end_state(tmp_path):
    fail = {"on": False}

    def factory(*args, **kw):
        if fail["on"]:
            raise RuntimeError("no EGL")
        return state_factory(*args, **kw)

    with TestClient(create_app(FakeBackend([good_plan(), judged(), good_plan()]), factory, tmp_path / "runs",
                               lessons_dir=tmp_path / "lessons")) as c:
        first = plan_and_execute(c)
        assert (tmp_path / "runs" / first / END_STATE).exists()
        fail["on"] = True
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        assert run_events(tmp_path, first)[-1]["status"] == "error"
        assert not (tmp_path / "runs" / first / END_STATE).exists()
        fail["on"] = False
        r = c.post(f"/api/plan/{first}/replan")  # v1.18: a continue without an end state starts where round 1 did
        assert r.status_code == 200
        wait_idle(c)
    start = run_events(tmp_path, r.json()["run_id"])[0]
    assert (start["round"], start["round_mode"]) == (2, "start")


# 4 and 5. The verdict call checks Stop first, and it always ends with a verdict event.
def _executed(tmp_path):
    StateEnv.executions = 0
    app = create_app(FakeBackend([good_plan(), judged()]), state_factory, tmp_path / "runs",
                     lessons_dir=tmp_path / "lessons")
    return app


def test_the_verdict_call_checks_stop_before_it_asks(tmp_path):
    app = _executed(tmp_path)
    with TestClient(app) as c:
        run_id = plan_and_execute(c)
        backend = FakeBackend([judged(True)])
        runner = app.state.runner
        runner.backend = backend

        async def noop():
            pass

        got = asyncio.run(runner._judge(tmp_path / "runs" / run_id, runner_scene(), PICK, [noop], lambda: True))
    assert got is None and backend.sent == []
    v = run_events(tmp_path, run_id)[-1]
    assert v["kind"] == "verdict" and v["done"] is None and v["error"] == "stopped"


def test_an_error_in_the_verdict_step_still_gives_a_verdict_event(tmp_path, monkeypatch):
    import plan_and_act.server.app as server

    def broken(plan):
        raise KeyError("keyframes")

    monkeypatch.setattr(server, "_reply_format", broken)
    with TestClient(create_app(FakeBackend([good_plan()]), state_factory, tmp_path / "runs",
                               lessons_dir=tmp_path / "lessons")) as c:
        run_id = plan_and_execute(c)
        assert c.get("/api/state").json()["running"] is False
    evs = run_events(tmp_path, run_id)
    assert evs[-2]["verdict_next"] is True
    assert evs[-1]["kind"] == "verdict" and evs[-1]["done"] is None and "KeyError" in evs[-1]["error"]


def runner_scene():
    from plan_and_act.scenes import SCENES
    return SCENES["cubes"]


# 6. The plan prompt says that only a re-plan from the start resets the scene.
# 8. A done re-plan shows the last auto-check result.
def test_a_replan_run_start_has_the_parent_check(tmp_path):
    with TestClient(create_app(FakeBackend([good_plan(), judged(), good_plan(DONE)]), state_factory,
                               tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        first = plan_and_execute(c)
        second = c.post(f"/api/plan/{first}/replan").json()["run_id"]
        wait_idle(c)
    assert run_events(tmp_path, first)[0]["parent_check"] is None
    start = run_events(tmp_path, second)[0]
    assert start["parent_check"] is False  # the fake execution stopped: the check says not solved
    plan = next(e for e in run_events(tmp_path, second) if e["kind"] == "plan")
    assert plan["done"] is True


def test_the_done_note_shows_the_last_auto_check():
    js = _web("app.js")
    check = _body(js, "function planCheckEntry")
    assert "parentCheck" in check and "The last auto-check says" in check
    assert "parentCheck = " in _body(js, "function onEvent(ev, replay = false)")


# 3. The action data of exec_start starts from the real gripper.
def test_exec_start_action_data_starts_from_the_gripper_of_the_plan_view():
    from plan_and_act.plan.formats import Keyframe
    from plan_and_act.server.app import _start_gripper
    lift = (Keyframe((0.5, 0.0, 0.2), 0.0, "closed", "red cube"),)
    assert _start_gripper({"track": {"start": {"state": "closed"}}}) == "closed"
    assert _start_gripper(None) == "open" and _start_gripper({"track": {}}) == "open"
    from plan_and_act.env import keyframe_actions
    assert [a.type for a in keyframe_actions(lift, _start_gripper({"track": {"start": {"state": "closed"}}}))] == \
        ["move_to"]


# 9. The Re-plan button keeps the mode of the moment it was made.
def test_the_replan_button_captures_its_mode_when_it_is_made():
    js = _web("app.js")
    button = _body(js, "function replanButton")
    assert "const fromStart = execError;" in button and "() => replan(fromStart, partial)" in button
    body = _body(js, "async function replan")
    assert "async function replan(fromStart = false, partial = false)" in js
    assert "fromStart ? \"start\" : partial ? \"continue\" : roundMode()" in body
    assert "execError" not in body and "execPartial" not in body


# 10. In continue mode, the round after an execution error starts from the start.
def test_the_loop_round_after_an_execution_error_starts_from_the_start(tmp_path):
    from tests.test_done_verdict import ENVS
    ENVS.clear()
    rounds = run_loop(tmp_path, [good_plan(), good_plan(), judged(True)], ["error", "success"], PICK)
    assert len(rounds) == 2 and rounds[1][0]["round_mode"] == "start"
    assert all(getattr(e, "start", None) is None for e in ENVS)  # no state file: a fresh reset
    end = loop_end(rounds)
    assert (end["success"], end["reason"]) == (True, "the auto-check passed")


def test_an_execution_error_in_the_last_round_ends_the_loop(tmp_path):
    rounds = run_loop(tmp_path, [good_plan()], ["error"], PICK, rounds=1)
    end = loop_end(rounds)
    assert (end["success"], end["reason"], end["rounds_run"]) == (False, "the execution failed", 1)


def test_a_start_round_after_a_continue_round_gets_no_state_file(tmp_path):
    from tests.test_done_verdict import ENVS
    ENVS.clear()
    rounds = run_loop(tmp_path, [good_plan(), good_plan(), good_plan(), judged(True)], ["failed", "error", "success"],
                      PICK)
    assert [r[0].get("round_mode") for r in rounds] == [None, "continue", "start"]
    assert getattr(ENVS[2], "start", None) is not None  # round 2 continues from the end of round 1
    assert all(getattr(e, "start", None) is None for e in ENVS[-2:])  # round 3: its plan and execute reset
    assert not list((tmp_path / "runs" / rounds[2][0]["run_id"]).glob("start_state.npz"))
    assert loop_end(rounds)["success"] is True
