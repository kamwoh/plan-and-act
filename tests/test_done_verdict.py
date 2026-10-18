"""Claude's own done verdict in a re-plan, and the Auto loop that ends on it."""
import pytest
from fastapi.testclient import TestClient

from armmind.agent.loop import Agent
from armmind.agent.parse import ReplyError, parse_plan
from armmind.agent.prompts import PLAN_REPLY_SCHEMA, PLAN_SYSTEM
from armmind.llm.fake import FakeBackend
from armmind.llm.protocol import Final, Usage
from armmind.server.app import create_app
from tests.fakes import Recorder, direct
from tests.test_agent_plan import PLAN, PlanEnv, good_plan
from tests.test_auto_loop import OUTCOMES, LoopEnv, events, folders, loop_end
from tests.test_continue_api import write_state
from tests.test_plan_api import ENVS, PICK
from tests.test_replan import feedback
from tests.test_server import make_env, wait_idle


def verdict(done, reason="The cube is up."):
    return [Final("", {"done": done, "reason": reason}), Usage("fake", 0.1, 5, 2, 0.001)]


DONE = {**PLAN, "keyframes": [], "done": True, "summary": "The images show the task is complete."}
FREE = "Wave the gripper over the cubes."  # no task of the scene: no auto-check


def test_the_schema_and_the_system_prompt_have_done():
    assert PLAN_REPLY_SCHEMA["properties"]["done"] == {"type": "boolean"}
    assert "done" not in PLAN_REPLY_SCHEMA["required"]
    assert "minItems" not in PLAN_REPLY_SCHEMA["properties"]["keyframes"]  # parse_plan checks the count
    assert '"done"' in PLAN_SYSTEM and "task is complete" in PLAN_SYSTEM


def test_done_in_a_replan_can_have_no_keyframes():
    p = parse_plan(DONE, replan=True)
    assert p.done is True and p.keyframes == ()
    assert parse_plan({**PLAN, "done": True}, replan=True).done is True  # keyframes are fine too


def test_no_keyframes_without_done_fails():
    for replan in (False, True):
        with pytest.raises(ReplyError, match="keyframes"):
            parse_plan({**DONE, "done": False}, replan=replan)
        with pytest.raises(ReplyError, match="keyframes"):
            parse_plan({k: v for k, v in DONE.items() if k != "done"}, replan=replan)


def test_done_in_round_1_is_ignored():
    assert parse_plan({**PLAN, "done": True}).done is False
    with pytest.raises(ReplyError, match="keyframes"):
        parse_plan(DONE)  # without done, no keyframes is not a plan


def test_done_must_be_a_bool():
    with pytest.raises(ReplyError, match="done"):
        parse_plan({**DONE, "done": "yes"}, replan=True)


async def test_a_done_replan_gives_a_plan_event_with_done_and_no_keyframes():
    rec = Recorder()
    status = await Agent(FakeBackend([good_plan(DONE)])).plan(PlanEnv(), direct, rec, feedback=feedback())
    plan = dict(rec.events)["plan"]
    assert status == "planned" and plan["done"] is True and plan["keyframes"] == []
    rec = Recorder()
    await Agent(FakeBackend([good_plan({**PLAN, "done": True})])).plan(PlanEnv(), direct, rec)
    assert dict(rec.events)["plan"]["done"] is False  # round 1


# ---- The Auto loop ----
class SavingLoopEnv(LoopEnv):
    def save_state(self, path):
        write_state(path, self.scene_name, self.seed)


def factory(scene, task, instruction, seed, show_positions, on_frame, start=None):
    env = make_env(scene, task, instruction, cls=SavingLoopEnv)
    env.seed, env.start = seed, start
    ENVS.append(env)
    return env


def run_loop(tmp_path, replies, outcomes, instruction, rounds=3):
    OUTCOMES[:] = outcomes
    with TestClient(create_app(FakeBackend(replies), factory, tmp_path / "runs",
                               lessons_dir=tmp_path / "lessons")) as c:
        r = c.post("/api/plan", json={"instruction": instruction, "seed": 5, "auto": True, "rounds": rounds})
        assert r.status_code == 200, r.text
        wait_idle(c)
    return [events(f) for f in folders(tmp_path)]


def executions(rounds):
    return [sum(e["kind"] == "exec_end" for e in r) for r in rounds]


def test_free_text_ends_on_claudes_done(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), good_plan(DONE)], ["done"], FREE)
    assert len(rounds) == 2 and executions(rounds) == [1, 0]  # a done plan has nothing to execute
    assert rounds[1][0]["round_mode"] == "continue"
    end = loop_end(rounds)
    assert (end["success"], end["judge"], end["rounds_run"]) == (True, "claude", 2)
    assert end["reason"] == "Claude says the task is done"


def test_free_text_runs_until_the_rounds_end_without_done(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), good_plan(), verdict(False, "Nothing moved.")], ["done", "done"], FREE,
                      rounds=2)
    assert executions(rounds) == [1, 1]
    end = loop_end(rounds)
    assert (end["success"], end["judge"], end["rounds_run"]) == (False, "claude", 2)
    assert end["reason"] == "the 2 rounds ended, and Claude says not done"
    assert [(v["round"], v["claude"], v["check"]) for v in end["verdicts"]] == [(1, False, None), (2, False, None)]
    assert end["claude"]["reason"] == "Nothing moved." and (end["agree"], end["compared"]) == (0, 0)


def test_the_last_execution_without_an_auto_check_ends_on_the_verdict_call(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), verdict(True)], ["done"], FREE, rounds=1)
    end = loop_end(rounds)
    assert (end["success"], end["judge"], end["reason"]) == (True, "claude", "Claude says the task is done")
    kinds = [e["kind"] for e in rounds[0]]
    assert kinds[-3:] == ["exec_end", "verdict", "loop_end"]
    v = rounds[0][-2]
    assert (v["done"], v["reason"], v["check"], v["round"]) == (True, "The cube is up.", None, 1)


def test_a_failed_verdict_call_is_not_a_success(tmp_path):
    rounds = run_loop(tmp_path, [good_plan()], ["done"], FREE, rounds=1)  # no reply left for the verdict
    end = loop_end(rounds)
    assert end["success"] is False and end["reason"] == "the 1 rounds ended, and the verdict call failed"
    v = next(e for e in rounds[0] if e["kind"] == "verdict")
    assert v["done"] is None and v["error"]


def test_with_an_auto_check_both_verdicts_and_the_agreements_are_in_loop_end(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), good_plan(), verdict(False, "It looks low.")], ["failed", "success"],
                      PICK)
    end = loop_end(rounds)
    assert (end["success"], end["judge"]) == (True, "check")  # the check decides
    assert [(v["round"], v["claude"], v["check"]) for v in end["verdicts"]] == [(1, False, False), (2, False, True)]
    assert (end["agree"], end["compared"]) == (1, 2)
    v = next(e for e in rounds[1] if e["kind"] == "verdict")
    assert (v["done"], v["check"]) == (False, True)


def test_claudes_done_against_a_failed_auto_check_ends_the_loop(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), good_plan(DONE), good_plan()], ["failed"], PICK)
    assert len(rounds) == 2 and executions(rounds) == [1, 0]
    end = loop_end(rounds)
    assert (end["success"], end["judge"]) == (False, "check")
    assert end["reason"] == "Claude says done, the auto-check says not solved"
    assert [(v["claude"], v["check"]) for v in end["verdicts"]] == [(True, False)]
    assert (end["agree"], end["compared"]) == (0, 1)


def test_with_an_auto_check_the_check_is_the_judge(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), good_plan(), verdict(True)], ["failed", "success"], PICK)
    end = loop_end(rounds)
    assert (end["success"], end["judge"], end["reason"]) == (True, "check", "the auto-check passed")


def test_execute_of_a_done_plan_without_keyframes_is_refused(tmp_path):
    rounds = run_loop(tmp_path, [good_plan(), good_plan(DONE)], ["done"], FREE)
    with TestClient(create_app(FakeBackend([]), factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        r = c.post(f"/api/plan/{rounds[1][0]['run_id']}/execute")
        assert r.status_code == 404 and "done" in r.json()["detail"]


@pytest.mark.assets
def test_the_demo_free_text_loop_ends_on_claudes_done_after_2_rounds(tmp_path):
    from armmind.llm.fake import DemoBackend
    from armmind.server.app import default_env_factory
    app = create_app(DemoBackend(delay_s=0.0), default_env_factory, tmp_path / "runs",
                     lessons_dir=tmp_path / "lessons")
    app.state.runner.auto_view_s = 0.0
    with TestClient(app) as c:
        r = c.post("/api/plan", json={"instruction": FREE, "seed": 0, "auto": True, "rounds": 3})
        assert r.status_code == 200
        wait_idle(c, timeout=180)
    rounds = [events(f) for f in folders(tmp_path)]
    assert executions(rounds) == [1, 0]
    assert next(e for e in rounds[1] if e["kind"] == "plan")["done"] is True
    end = loop_end(rounds)
    assert (end["success"], end["judge"], end["rounds_run"]) == (True, "claude", 2)


# ---- The verdict call after a manual Execute ----
def test_a_manual_execute_ends_with_a_verdict_call_without_true_values(tmp_path):
    from armmind.agent.prompts import VERDICT_SCHEMA, VERDICT_SYSTEM
    from tests.test_replan_api import planned_and_executed, run_events, true_factory
    backend = FakeBackend([good_plan(), verdict(False, "The gripper is empty.")])
    with TestClient(create_app(backend, true_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        run_id = planned_and_executed(c)
    evs = run_events(tmp_path, run_id)
    assert [e["kind"] for e in evs][-2:] == ["exec_end", "verdict"] and evs[-2]["verdict_next"] is True
    v = evs[-1]
    assert (v["done"], v["reason"], v["check"], v["round"], v["step"]) == (False, "The gripper is empty.", False, 1, 2)
    assert backend.systems[-1] == VERDICT_SYSTEM
    msg, schema = backend.sent[-1]
    assert schema == VERDICT_SCHEMA and msg.images == (b"front", b"top")  # the end images
    assert PICK in msg.text and '"note":"up"' in msg.text and v["prompt"] == msg.text
    for secret in ("0.587", "0.5871", "-0.123", "0.0213", "33.3", "7.31", "7.3", "0.13", "4.4", "missed by",
                   "blue_cube", "prediction", "perception", "debug", "auto-check", "stopped", "solved"):
        assert secret not in msg.text, secret


def test_the_verdict_call_retries_a_bad_reply_once(tmp_path):
    from tests.fakes import bad
    from tests.test_replan_api import planned_and_executed, run_events, true_factory
    backend = FakeBackend([good_plan(), bad(), verdict(True)])
    with TestClient(create_app(backend, true_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        run_id = planned_and_executed(c)
    v = run_events(tmp_path, run_id)[-1]
    assert v["kind"] == "verdict" and v["done"] is True and len(v["attempts"]) == 2 and v["error"] is None


def test_a_stopped_execution_gets_no_verdict_call(tmp_path):
    from tests.test_replan_api import run_events
    OUTCOMES[:] = ["hang"]
    backend = FakeBackend([good_plan()])
    with TestClient(create_app(backend, factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
        import time
        time.sleep(0.2)
        c.post("/api/stop")
        wait_idle(c)
    evs = run_events(tmp_path, run_id)
    assert evs[-1]["kind"] == "exec_end" and evs[-1]["verdict_next"] is False and len(backend.sent) == 1


def test_the_verdict_reply_is_checked():
    from armmind.agent.parse import parse_verdict
    assert parse_verdict({"done": True, "reason": "  ok  "}).reason == "ok"
    for bad in ({"done": "yes", "reason": "r"}, {"done": True}, {"reason": "r"}, [1]):
        with pytest.raises(ReplyError):
            parse_verdict(bad)


def test_the_demo_verdict():
    from armmind.agent.prompts import build_verdict_text
    from armmind.llm.fake import DEMO_PLAN, DEMO_PLAN_LOW, demo_verdict
    def text(plan):
        return build_verdict_text("s", PICK, tuple(plan["objects"]), tuple(plan["keyframes"]))
    assert demo_verdict(text(DEMO_PLAN_LOW))["done"] is False and demo_verdict(text(DEMO_PLAN))["done"] is True


def test_the_run_summary_counts_the_cost_of_the_verdict_call():
    from armmind.store.runs import summarize
    s = summarize("r", iter([{"kind": "run_start", "mode": "plan"},
                             {"kind": "plan", "usage": {"in": 10, "out": 5, "cost_usd": 0.01}},
                             {"kind": "verdict", "usage": {"in": 4, "out": 1, "cost_usd": 0.002}}]))
    assert s["steps"] == 1 and s["cost_usd"] == 0.012 and (s["in_tokens"], s["out_tokens"]) == (14, 6)


class PlanThenHang:
    """Answers the plan call, then hangs on the verdict call until Stop cancels it."""
    name = "plan-then-hang"

    def __init__(self):
        from tests.fakes import HangingBackend
        self.plan, self.hang, self.calls = FakeBackend([good_plan()]), HangingBackend(), 0

    async def open(self, system):
        self.calls += 1
        return await (self.plan if self.calls == 1 else self.hang).open(system)


@pytest.mark.parametrize("auto", [False, True])
def test_stop_during_the_verdict_call_ends_the_run(tmp_path, auto):
    import time

    from tests.test_replan_api import run_events
    OUTCOMES[:] = ["failed"]
    backend = PlanThenHang()
    with TestClient(create_app(backend, factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        body = {"instruction": PICK, "auto": True, "rounds": 1} if auto else {"instruction": PICK}
        run_id = c.post("/api/plan", json=body).json()["run_id"]
        if not auto:
            wait_idle(c)
            assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
        end = time.monotonic() + 5
        while backend.calls < 2:  # the verdict call hangs now
            assert time.monotonic() < end
            time.sleep(0.02)
        assert c.get("/api/state").json()["running"] is True
        assert c.post("/api/stop").status_code == 200
        wait_idle(c)
    evs = run_events(tmp_path, run_id)
    v = next(e for e in evs if e["kind"] == "verdict")
    assert v["done"] is None and v["error"]  # no verdict
    if auto:
        assert evs[-1]["kind"] == "loop_end" and evs[-1]["reason"] == "stopped" and evs[-1]["verdicts"] == []


def test_an_llm_error_in_the_verdict_call_ends_the_run_with_no_verdict(tmp_path):
    from armmind.llm.protocol import LLMError
    from tests.test_replan_api import planned_and_executed, run_events, true_factory
    backend = FakeBackend([good_plan(), [LLMError("backend", "the CLI failed")]])
    with TestClient(create_app(backend, true_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        run_id = planned_and_executed(c)
        assert c.get("/api/state").json()["running"] is False
    v = run_events(tmp_path, run_id)[-1]
    assert v["kind"] == "verdict" and v["done"] is None and v["error"] == "the CLI failed"
    assert len(v["attempts"]) == 1  # no retry after an error of the model call
