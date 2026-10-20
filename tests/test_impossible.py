"""v1.16: Claude can stop a task as not possible. Nothing then executes, and the Auto loop ends at once."""
import pytest
from fastapi.testclient import TestClient

from plan_and_act.agent.loop import Agent
from plan_and_act.agent.parse import ReplyError, parse_plan
from plan_and_act.agent.prompts import PLAN_REPLY_SCHEMA, PLAN_SYSTEM
from plan_and_act.llm.fake import FakeBackend
from plan_and_act.server.app import create_app
from tests.fakes import Recorder, direct
from tests.test_agent_plan import PLAN, PlanEnv, good_plan
from tests.test_auto_loop import loop_end
from tests.test_done_verdict import DONE, FREE, executions, run_loop
from tests.test_plan_api import PICK, plan_factory
from tests.test_replan import feedback
from tests.test_server import wait_idle

WHY = "The scene has no yellow cube."
NOPE = {**PLAN, "keyframes": [], "impossible": WHY, "summary": "The task cannot be done."}


def test_the_schema_and_the_prompt_have_impossible():
    assert PLAN_REPLY_SCHEMA["properties"]["impossible"] == {"type": "string"}
    assert "impossible" not in PLAN_REPLY_SCHEMA["required"]
    assert '"impossible"' in PLAN_SYSTEM and "One failed attempt is not a reason" in PLAN_SYSTEM


def test_parse_impossible():
    p = parse_plan(NOPE)
    assert p.impossible == WHY and p.keyframes == ()  # round 1 too
    assert parse_plan(NOPE, replan=True).impossible == WHY
    assert parse_plan({**PLAN, "impossible": "  "}).impossible == ""  # empty means possible
    assert parse_plan(PLAN).impossible == ""
    with pytest.raises(ReplyError, match="keyframes"):
        parse_plan({**NOPE, "impossible": ""})
    with pytest.raises(ReplyError, match="impossible"):
        parse_plan({**NOPE, "impossible": 3})


def test_done_wins_over_impossible():
    p = parse_plan({**DONE, "impossible": WHY}, replan=True)
    assert p.done is True and p.impossible == ""


async def test_round_1_can_say_impossible():
    rec = Recorder()
    status = await Agent(FakeBackend([good_plan(NOPE)])).plan(PlanEnv(), direct, rec)
    plan = dict(rec.events)["plan"]
    assert status == "planned" and plan["impossible"] == WHY and plan["keyframes"] == []
    rec = Recorder()
    await Agent(FakeBackend([good_plan()])).plan(PlanEnv(), direct, rec, feedback=feedback())
    assert dict(rec.events)["plan"]["impossible"] == ""


def test_execute_of_an_impossible_plan_is_409(tmp_path):
    for body in (NOPE, {**PLAN, "impossible": WHY}):  # also with keyframes
        with TestClient(create_app(FakeBackend([good_plan(body)]), plan_factory, tmp_path / "runs",
                                   lessons_dir=tmp_path / "lessons")) as c:
            run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
            wait_idle(c)
            r = c.post(f"/api/plan/{run_id}/execute")
        assert r.status_code == 409 and "not possible" in r.json()["detail"] and WHY in r.json()["detail"]


def test_the_auto_loop_ends_at_once_in_round_1(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(NOPE)], [], PICK)
    assert executions(rounds) == [0]
    end = loop_end(rounds)
    assert end["impossible"] == {"round": 1, "reason": WHY} and end["success"] is False
    assert end["reason"] == f"Claude says the task is not possible: {WHY}"


def test_a_replan_in_the_auto_loop_ends_it(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), good_plan({**PLAN, "impossible": WHY})], ["done"], FREE)
    assert len(rounds) == 2 and executions(rounds) == [1, 0]
    end = loop_end(rounds)
    assert end["impossible"] == {"round": 2, "reason": WHY} and end["success"] is False
    assert [v["round"] for v in end["verdicts"]] == [1]  # the re-plan still judges round 1


def test_a_loop_without_impossible_has_none(tmp_path):
    end = loop_end(run_loop(tmp_path, [good_plan(), good_plan(DONE)], ["done"], FREE))
    assert end["impossible"] is None
