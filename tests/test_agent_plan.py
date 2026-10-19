"""Agent.plan: one call that gives a belief of the scene and a whole-task plan."""
import asyncio

from armmind.agent.loop import Agent
from armmind.agent.prompts import CHAIN_SYSTEM, PLAN_REPLY_SCHEMA
from armmind.llm.fake import FakeBackend
from armmind.llm.protocol import Final, LLMError, Usage
from armmind.store.lessons import LessonStore
from armmind.types import Observation
from tests.fakes import FakeEnv, HangingBackend, Recorder, bad, direct

PLAN = {"objects": [{"name": "red_cube", "shape": "box", "size": [0.04, 0.04, 0.04], "xyz": [0.5, -0.1, 0.02],
                     "yaw": 0}],
        "keyframes": [{"xyz": [0.5, -0.1, 0.15], "yaw": 120, "gripper": "open", "holding": "nothing", "note": "up"}],
        "summary": "Go above the cube."}


def good_plan(body=PLAN):
    return [Final("", body), Usage("fake", 0.2, 10, 5, 0.01)]


class PlanEnv(FakeEnv):
    """A FakeEnv with the plan methods. Its observation holds true positions, as with the debug switch."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.checked = None

    def observe(self):
        return Observation(b"front", b"top", (0.3, 0.0, 0.4), 0.08, (), {"red_cube": (0.512, -0.108, 0.02)},
                           object_yaws={"red_cube": 33.0}, joints=("drawer slide 0.000 m",))

    def check_plan(self, belief, keyframes):
        self.checked = (belief, keyframes)
        return {"perception": [], "missing": [], "invented": [], "summary": {"mean_xy_cm": 0.0},
                "marks": [{"ok": True, "checks": {}} for _ in keyframes]}

    def plan_view(self, belief, keyframes, marks=None):
        return {"believed": [b.name for b in belief], "marks": marks}


async def test_plan_events_and_message(tmp_path):
    store = LessonStore(tmp_path)
    store.append("cubes", ["Closing 2 cm up held."], run="r0", outcome="success")
    backend, rec, env = FakeBackend([good_plan()]), Recorder(), PlanEnv()
    status = await Agent(backend, store=store, run_id="20261017-000000").plan(env, direct, rec, seed=4)
    assert status == "planned"
    assert rec.kinds() == ["run_start", "observation", "ask", "plan", "run_end"]
    start, plan_ev = rec.events[0][1], dict(rec.events)["plan"]
    assert start["mode"] == "plan" and start["seed"] == 4 and start["run_id"] == "20261017-000000"
    assert start["system_prompt"] == CHAIN_SYSTEM and start["saved_lessons"] == ["Closing 2 cm up held."]
    msg, schema = backend.sent[0]
    assert schema == PLAN_REPLY_SCHEMA and backend.systems == [CHAIN_SYSTEM]
    assert msg.images == (b"front", b"top")
    for part in (env.scene_text, env.instruction, "Closing 2 cm up held.", "Image 1: front camera"):
        assert part in msg.text
    for secret in ("0.512", "-0.108", "33", "debug", "drawer slide"):
        assert secret not in msg.text  # never the true state
    assert plan_ev["summary"] == "Go above the cube." and plan_ev["warnings"] == ["keyframe 1: yaw clamped to 90"]
    assert plan_ev["keyframes"][0] == {"xyz": [0.5, -0.1, 0.15], "yaw": 90.0, "state": "open", "carrying": None,
                                       "note": "up", "moves": []}
    assert plan_ev["objects"][0]["name"] == "red_cube"
    assert plan_ev["scores"]["marks"] == [{"ok": True, "checks": {}}]
    assert plan_ev["view"] == {"believed": ["red_cube"], "marks": [{"ok": True, "checks": {}}]}
    assert plan_ev["usage"]["cost_usd"] == 0.01 and plan_ev["prompt"] == msg.text
    assert len(plan_ev["attempts"]) == 1
    assert env.checked[1][0].yaw == 90.0
    assert rec.events[-1][1] == {"status": "planned"}


async def test_plan_lessons_off():
    backend = FakeBackend([good_plan()])
    store_rec = Recorder()
    await Agent(backend, use_lessons=False).plan(PlanEnv(), direct, store_rec, seed=0)
    assert "Lessons: (none)" in backend.sent[0][0].text


async def test_schema_retry_works_for_a_plan():
    bad_body = {**PLAN, "keyframes": []}
    backend, rec = FakeBackend([[Final("", bad_body)], good_plan()]), Recorder()
    status = await Agent(backend).plan(PlanEnv(), direct, rec, seed=0)
    assert status == "planned"
    assert "not valid" in backend.sent[1][0].text and "1 to 40" in backend.sent[1][0].text
    assert [e[1]["attempt"] for e in rec.events if e[0] == "ask"] == [1, 2]
    assert len(dict(rec.events)["plan"]["attempts"]) == 2


async def test_two_bad_plans_end_with_bad_reply():
    backend, rec = FakeBackend([bad(), bad()]), Recorder()
    status = await Agent(backend).plan(PlanEnv(), direct, rec, seed=0)
    assert status == "bad_reply" and "bad_reply" in rec.kinds() and "plan" not in rec.kinds()


async def test_model_error_ends_the_plan():
    backend, rec = FakeBackend([[LLMError("auth", "please log in")]]), Recorder()
    status = await Agent(backend).plan(PlanEnv(), direct, rec, seed=0)
    assert status == "llm_error" and rec.kinds()[-2:] == ["llm_error", "run_end"]


async def test_stop_while_planning():
    agent, rec = Agent(HangingBackend()), Recorder()
    task = asyncio.create_task(agent.plan(PlanEnv(), direct, rec, seed=0))
    await asyncio.sleep(0.05)
    await agent.stop()
    assert await asyncio.wait_for(task, 1.0) == "stopped"
    assert rec.events[-1] == ("run_end", {"status": "stopped"})
