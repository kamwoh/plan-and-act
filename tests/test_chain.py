"""v1.14: one chain is one chat. Each re-plan and each verdict goes into the chat of round 1."""
import json

import pytest
from fastapi.testclient import TestClient

from armmind.agent.loop import Agent, Chain
from armmind.agent.prompts import (
    CHAIN_SYSTEM,
    JUDGE,
    PLAN_SYSTEM,
    VERDICT_SYSTEM,
    build_chain_verdict_text,
    build_replan_text,
)
from armmind.llm.fake import FakeBackend
from armmind.llm.protocol import Message
from armmind.server.app import create_app
from tests.fakes import Recorder, direct, judged
from tests.test_agent_plan import PLAN, PlanEnv, good_plan
from tests.test_continue_api import ENVS, StateEnv, factory
from tests.test_plan_api import PICK
from tests.test_replan import OBS, feedback
from tests.test_replan_api import run_events
from tests.test_server import wait_idle

# The true values of TrueStateEnv and PlanEnv, and the words of the scores. No message may hold one.
SECRETS = ("0.587", "0.5871", "-0.123", "0.0213", "33.3", "7.31", "7.3", "0.13", "4.4", "missed by", "blue_cube",
           "prediction", "perception", "debug", "0.512", "-0.108", "drawer slide")
NOTE = "Claude lost the earlier chat. This round starts a new chat with a summary."


@pytest.fixture(autouse=True)
def _clear():
    ENVS.clear()
    StateEnv.executions = 0


def app(tmp_path, backend):
    return TestClient(create_app(backend, factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons"))


def plan_body(summary):
    return good_plan({**PLAN, "summary": summary})


def texts(seen):
    return [m.text for m in seen]


def assert_no_secrets(text):
    for secret in SECRETS:
        assert secret not in text, secret


# ---- The prompts ----
def test_the_chain_system_is_the_plan_system_and_the_verdict_rules():
    assert CHAIN_SYSTEM.startswith(PLAN_SYSTEM) and CHAIN_SYSTEM.endswith(JUDGE)
    assert VERDICT_SYSTEM.endswith(JUDGE) and "verdict" in CHAIN_SYSTEM[len(PLAN_SYSTEM):]
    assert "earlier in this chat" in PLAN_SYSTEM


def test_the_replan_text_has_only_what_is_new():
    text = build_replan_text(OBS, None, feedback())
    assert "This is a re-plan, round 2." in text and "Your last plan:" not in text
    assert "Scene:" not in text and "Task:" not in text and "Lessons" not in text
    assert "action 2 of 4 (close)" in text and "not solved" in text
    assert "Plan the whole task now, in one reply." in text and "Image 3: the front camera" in text
    assert "Gripper at (0.300, 0.000, 0.400)" in text


def test_the_replan_text_that_continues_says_what_the_scene_is_now():
    text = build_replan_text(OBS, None, feedback(mode="continue", images=()), gripper="closed")
    assert "did not reset" in text and "Plan what is left of the task now, in one reply." in text
    assert "The gripper is closed now." in text and "Image 3" not in text


def test_the_replan_text_never_has_true_values():
    obs = PlanEnv().observe()
    assert_no_secrets(build_replan_text(obs, None, feedback()))


def test_the_chain_verdict_text_has_no_plan():
    text = build_chain_verdict_text(None)
    assert "keyframes" not in text and "Image 1: front camera. Image 2: top camera." in text
    assert "Is the task complete?" in text


# ---- The agent ----
async def test_round_1_opens_the_chat_of_its_chain():
    backend, rec = FakeBackend([good_plan()]), Recorder()
    await Agent(backend, run_id="20261018-100000").plan(PlanEnv(), direct, rec)
    assert backend.opens == [{"system": CHAIN_SYSTEM, "chain": "20261018-100000", "resume": None}]
    start, plan = rec.events[0][1], dict(rec.events)["plan"]
    assert start["chain"] == "20261018-100000" and start["system_prompt"] == CHAIN_SYSTEM
    assert plan["session_id"] == "fake-session-1"


async def test_a_replan_continues_the_chat_with_a_short_message():
    backend = FakeBackend([good_plan(), good_plan()])
    await Agent(backend, run_id="20261018-100000").plan(PlanEnv(), direct, Recorder())
    rec = Recorder()
    agent = Agent(backend, run_id="20261018-100100", chain=Chain("20261018-100000", "fake-session-1"))
    await agent.plan(PlanEnv(), direct, rec, feedback=feedback())
    assert backend.opens[-1] == {"system": CHAIN_SYSTEM, "chain": "20261018-100000", "resume": "fake-session-1"}
    seen = backend.seen[-1]
    assert [m.role for m in seen] == ["user", "assistant", "user"]
    assert '"summary": "Go above the cube."' in seen[1].text  # Claude has its last plan in the chat
    assert "Your last plan:" not in seen[2].text and seen[2].images == (b"front", b"top", b"end-front", b"end-top")
    assert "chat_lost" not in rec.kinds() and dict(rec.events)["plan"]["session_id"] == "fake-session-1"


async def test_a_lost_chat_falls_back_to_the_summary_message():
    backend, rec = FakeBackend([good_plan()]), Recorder()
    agent = Agent(backend, run_id="20261018-100100", chain=Chain("20261018-100000", "gone"))
    assert await agent.plan(PlanEnv(), direct, rec, feedback=feedback()) == "planned"
    note = dict(rec.events)["chat_lost"]
    assert note["message"] == NOTE and "No conversation found" in note["error"]
    assert [o["resume"] for o in backend.opens] == ["gone", None]
    assert "Your last plan:" in backend.seen[-1][-1].text and len(backend.seen[-1]) == 1  # a new chat
    plan = dict(rec.events)["plan"]
    assert plan["session_id"] == "fake-session-1" and "Your last plan:" in plan["prompt"]
    assert plan["attempts"][0]["error"].startswith("No conversation found")


async def test_a_replan_without_a_saved_chat_sends_the_summary_with_the_note():
    backend, rec = FakeBackend([good_plan()]), Recorder()
    await Agent(backend, run_id="r2", chain=Chain("r1", None)).plan(PlanEnv(), direct, rec, feedback=feedback())
    assert dict(rec.events)["chat_lost"]["message"] == "This round starts a new chat with a summary."
    assert backend.opens == [{"system": CHAIN_SYSTEM, "chain": "r1", "resume": None}]
    assert "Your last plan:" in backend.sent[0][0].text


async def test_a_verdict_continues_the_chat_without_the_plan():
    backend = FakeBackend([good_plan(), judged(True, "It is up.")])
    await Agent(backend, run_id="r1").plan(PlanEnv(), direct, Recorder())
    verdict, detail = await Agent(backend, run_id="r1", chain=Chain("r1", "fake-session-1")).verdict(
        "s", PICK, (), (), (b"f", b"t"))
    assert verdict.done is True and detail["session_id"] == "fake-session-1"
    last = backend.seen[-1][-1]
    assert last.text == build_chain_verdict_text(None) and last.images == (b"f", b"t")
    assert len(backend.seen[-1]) == 3


# ---- The server ----
def test_an_auto_loop_of_3_rounds_is_one_chat(tmp_path):
    backend = FakeBackend([plan_body("plan one"), plan_body("plan two"), plan_body("plan three"),
                           judged(False, "The cube is on the table.")])
    with app(tmp_path, backend) as c:
        r = c.post("/api/plan", json={"instruction": PICK, "seed": 5, "auto": True, "rounds": 3, "mode": "continue"})
        assert r.status_code == 200, r.text
        wait_idle(c, 10)
        first = r.json()["run_id"]
    assert {o["chain"] for o in backend.opens} == {first}
    assert [o["resume"] for o in backend.opens] == [None, "fake-session-1", "fake-session-1", "fake-session-1"]
    assert len(backend.chats) == 1
    second = texts(backend.seen[1])
    assert len(second) == 3 and '"summary": "plan one"' in second[1]
    assert "This is a re-plan, round 2" in second[2] and "Your last plan:" not in second[2]
    last = texts(backend.seen[-1])  # the verdict after round 3 sees all three plans
    assert len(last) == 7 and all(f'"summary": "plan {n}"' in "".join(last) for n in ("one", "two", "three"))
    for msg, _ in backend.sent:  # the no-true-values rule holds for every message of the chain
        assert_no_secrets(msg.text)
    runs = sorted((tmp_path / "runs").iterdir())
    starts = [run_events(tmp_path, f.name)[0] for f in runs]
    assert [s["chain"] for s in starts] == [first] * 3
    assert [s["session_id"] for s in starts] == [None, "fake-session-1", "fake-session-1"]


def test_a_manual_replan_sees_the_plan_and_the_verdict(tmp_path):
    backend = FakeBackend([plan_body("plan one"), judged(False, "The gripper is empty."), plan_body("plan two")])
    with app(tmp_path, backend) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/replan", json={"mode": "start"}).status_code == 200
        wait_idle(c)
    seen = texts(backend.seen[-1])
    assert len(seen) == 5 and '"summary": "plan one"' in seen[1] and "The gripper is empty." in seen[3]
    assert "Your last plan:" not in seen[4]
    for msg, _ in backend.sent:
        assert_no_secrets(msg.text)


def test_a_replan_after_a_server_restart_continues_the_same_session(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan()])
    with app(tmp_path, backend) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
    with app(tmp_path, backend) as c:  # a new app over the same runs folder: nothing in memory
        second = c.post(f"/api/plan/{first}/replan", json={"mode": "continue"}).json()["run_id"]
        wait_idle(c)
    assert backend.opens[-1] == {"system": CHAIN_SYSTEM, "chain": first, "resume": "fake-session-1"}
    start = run_events(tmp_path, second)[0]
    assert (start["chain"], start["session_id"]) == (first, "fake-session-1")
    assert len(backend.seen[-1]) == 5 and "chat_lost" not in [e["kind"] for e in run_events(tmp_path, second)]


def test_a_failed_resume_on_the_server_falls_back_and_notes_it(tmp_path):
    backend = FakeBackend([good_plan(), judged(), good_plan()])
    with app(tmp_path, backend) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        assert c.post(f"/api/plan/{first}/execute").status_code == 200
        wait_idle(c)
        backend.chats.clear()  # the CLI lost the session
        second = c.post(f"/api/plan/{first}/replan", json={"mode": "start"}).json()["run_id"]
        wait_idle(c)
    evs = run_events(tmp_path, second)
    kinds = [e["kind"] for e in evs]
    assert kinds.index("chat_lost") < kinds.index("plan")
    assert next(e for e in evs if e["kind"] == "chat_lost")["message"] == NOTE
    plan = next(e for e in evs if e["kind"] == "plan")
    assert "Your last plan:" in plan["prompt"] and plan["session_id"] == "fake-session-2"
    assert_no_secrets(plan["prompt"])


async def test_the_fake_backend_keeps_each_chat():
    backend = FakeBackend([good_plan(), good_plan()])
    conv = await backend.open("s", chain="c")
    _ = [e async for e in conv.send(Message("user", "a"))]
    again = await backend.open("s", chain="c", resume=conv.session_id)
    _ = [e async for e in again.send(Message("user", "b"))]
    assert [m.text for m in backend.seen[-1]][::2] == ["a", "b"]
    assert json.loads(backend.seen[-1][1].text)["summary"] == PLAN["summary"]


def test_the_run_store_finds_the_chain_of_a_round(tmp_path):
    from armmind.store.runs import RunStore
    backend = FakeBackend([good_plan(), judged(), good_plan(), judged(), good_plan()])
    ids = []
    with app(tmp_path, backend) as c:
        ids.append(c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"])
        wait_idle(c)
        for _ in range(2):
            assert c.post(f"/api/plan/{ids[-1]}/execute").status_code == 200
            wait_idle(c)
            ids.append(c.post(f"/api/plan/{ids[-1]}/replan", json={"mode": "start"}).json()["run_id"])
            wait_idle(c)
    store = RunStore(tmp_path / "runs")
    assert store.chain(ids[2]) == ids and store.chain(ids[1]) == ids[:2] and store.chain(ids[0]) == ids[:1]
    assert store.chain("20261018-999999") is None
