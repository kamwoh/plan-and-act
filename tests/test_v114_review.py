"""The v1.14 review follow-ups: branches, the image limit, failed turns and the shared sessions folder."""
import json

import pytest

from armmind.agent.loop import (
    BRANCH_CHAT,
    CHAT_TOO_LONG,
    MAX_CHAT_IMAGES,
    NEW_CHAT,
    Agent,
    Chain,
)
from armmind.agent.prompts import build_replan_text
from armmind.llm.claude_cli import classify
from armmind.llm.fake import FakeBackend
from armmind.llm.protocol import LLMError
from armmind.store.runs import RunStore
from tests.fakes import Recorder, direct, judged
from tests.test_agent_plan import PLAN, PlanEnv, good_plan
from tests.test_chain import app, assert_no_secrets
from tests.test_continue_api import ENVS, StateEnv
from tests.test_page_plan import _body, _web
from tests.test_plan_api import PICK
from tests.test_replan import OBS, feedback
from tests.test_replan_api import run_events
from tests.test_server import wait_idle


@pytest.fixture(autouse=True)
def _clear():
    ENVS.clear()
    StateEnv.executions = 0


def plan_body(summary):
    return good_plan({**PLAN, "summary": summary})


def execute(c, run_id):
    assert c.post(f"/api/plan/{run_id}/execute").status_code == 200
    wait_idle(c)


def replan(c, run_id, mode="start"):
    r = c.post(f"/api/plan/{run_id}/replan", json={"mode": mode})
    assert r.status_code == 200, r.text
    wait_idle(c)
    return r.json()["run_id"]


def chain_of_3(c):
    """Rounds 1, 2 and 3 of one chain, each executed once, with a verdict after each execution."""
    ids = [c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]]
    wait_idle(c)
    for _ in range(2):
        execute(c, ids[-1])
        ids.append(replan(c, ids[-1]))
    return ids


def three_rounds():
    return [plan_body("plan one"), judged(), plan_body("plan two"), judged(), plan_body("plan three")]


# ---- 1. A re-plan of an older round starts a new chat ----
def test_a_replan_of_an_older_round_starts_a_new_chat(tmp_path):
    backend = FakeBackend([*three_rounds(), plan_body("branch")])
    with app(tmp_path, backend) as c:
        ids = chain_of_3(c)
        branch = replan(c, ids[0])  # round 1 has a later round: its chat holds plan two and plan three
    assert backend.opens[-1]["resume"] is None and backend.opens[-1]["chain"] == ids[0]
    chat = [m.text for m in backend.seen[-1]]
    assert len(chat) == 1 and "plan three" not in chat[0] and "plan two" not in chat[0]
    assert "Your last plan:" in chat[0] and '"note":"up"' in chat[0]  # the summary of round 1
    evs = run_events(tmp_path, branch)
    assert next(e for e in evs if e["kind"] == "chat_lost")["message"] == BRANCH_CHAT
    assert next(e for e in evs if e["kind"] == "plan")["session_id"] != backend.opens[1]["resume"]


# ---- 3. A verdict on an old round does not see a later round ----
def test_a_verdict_on_an_older_round_starts_a_new_chat(tmp_path):
    backend = FakeBackend([*three_rounds(), judged(True, "Done.")])
    with app(tmp_path, backend) as c:
        ids = chain_of_3(c)
        execute(c, ids[0])  # a second Execute of round 1, then its verdict
    assert backend.opens[-1]["resume"] is None
    chat = "\n".join(m.text for m in backend.seen[-1])
    assert "plan three" not in chat and "round 3" not in chat and "The plan that ran:" in chat
    verdict = run_events(tmp_path, ids[0])[-1]
    assert verdict["kind"] == "verdict" and verdict["done"] is True


# ---- 4. A chain without a saved chat says so ----
async def test_a_round_without_a_saved_chat_does_not_say_the_chat_is_lost():
    rec = Recorder()
    await Agent(FakeBackend([good_plan()]), run_id="r2", chain=Chain("r1")).plan(PlanEnv(), direct, rec,
                                                                                   feedback=feedback())
    assert dict(rec.events)["chat_lost"]["message"] == NEW_CHAT and "lost" not in NEW_CHAT


# ---- 2. The image limit ----
async def test_a_full_chat_starts_a_new_chat_with_a_summary():
    backend, rec = FakeBackend([good_plan(), good_plan()]), Recorder()
    await Agent(backend, run_id="r1").plan(PlanEnv(), direct, Recorder())
    full = Chain("r1", "fake-session-1", images=MAX_CHAT_IMAGES - 1, start=True)
    await Agent(backend, run_id="r2", chain=full).plan(PlanEnv(), direct, rec, feedback=feedback())
    assert dict(rec.events)["chat_lost"]["message"] == CHAT_TOO_LONG
    assert backend.opens[-1]["resume"] is None and len(backend.seen[-1]) == 1
    assert dict(rec.events)["plan"]["chat_images"] == 4


async def test_the_events_count_the_images_of_the_chat():
    backend = FakeBackend([good_plan(), good_plan(), judged()])
    rec1, rec2 = Recorder(), Recorder()
    await Agent(backend, run_id="r1").plan(PlanEnv(), direct, rec1)
    assert dict(rec1.events)["plan"]["chat_images"] == 2 and dict(rec1.events)["plan"]["chat_start"] is True
    chain = Chain("r1", "fake-session-1", images=2, start=True)
    await Agent(backend, run_id="r2", chain=chain).plan(PlanEnv(), direct, rec2, feedback=feedback())
    plan = dict(rec2.events)["plan"]
    assert plan["chat_images"] == 4 and backend.seen[-1][-1].images == (b"end-front", b"end-top")
    _, detail = await Agent(backend, run_id="r2", chain=Chain("r1", "fake-session-1", images=4, start=True)).verdict(
        "s", PICK, (), (), (b"f", b"t"))
    assert detail["chat_images"] == 6


def test_a_replan_from_the_start_leaves_out_the_start_images_that_the_chat_has():
    text = build_replan_text(OBS, None, feedback(), start_seen=True)
    assert "the same as the start images earlier in this chat" in text
    assert "Image 1: the front camera at the end of the last execution" in text and "Image 3" not in text
    assert "Image 1: front camera. Image 2: top camera." not in text
    full = build_replan_text(OBS, None, feedback())
    assert "Image 3: the front camera" in full


async def test_a_replan_from_the_start_sends_all_4_images_in_a_chat_without_the_start_images():
    backend = FakeBackend([good_plan()])
    chain = Chain("r1", "s1", images=2, start=False)
    backend.store.chats["s1"] = []
    await Agent(backend, run_id="r2", chain=chain).plan(PlanEnv(), direct, Recorder(), feedback=feedback())
    assert backend.seen[-1][-1].images == (b"front", b"top", b"end-front", b"end-top")


def test_a_too_many_images_error_counts_as_a_full_chat():
    for text in ("messages: too many images in the request", "Too many images: at most 100 images",
                 "prompt is too long: 1200000 tokens > 1000000 maximum"):
        assert classify(text) == "too_long", text
    assert classify("usage limit reached") == "limit"


async def test_a_too_long_error_falls_back_to_the_summary():
    backend, rec = FakeBackend([[LLMError("too_long", "too many images")], good_plan()]), Recorder()
    backend.store.chats["s1"] = []
    agent = Agent(backend, run_id="r2", chain=Chain("r1", "s1", images=10, start=True))
    assert await agent.plan(PlanEnv(), direct, rec, feedback=feedback()) == "planned"
    assert dict(rec.events)["chat_lost"]["message"] == CHAT_TOO_LONG
    assert backend.opens[-1]["resume"] is None and "Your last plan:" in backend.seen[-1][-1].text


def test_an_auto_loop_from_the_start_adds_2_images_per_round(tmp_path):
    backend = FakeBackend([good_plan(), good_plan(), good_plan(), judged()])
    with app(tmp_path, backend) as c:
        c.post("/api/plan", json={"instruction": PICK, "seed": 5, "auto": True, "rounds": 3, "mode": "start"})
        wait_idle(c, 10)
    sent = [len(m.images) for m, _ in backend.sent]
    assert sent == [2, 2, 2, 2]  # the start images go once, then only the end images
    for msg, _ in backend.sent:
        assert_no_secrets(msg.text)


# ---- 5. A failed or cancelled turn ends the chat ----
async def test_a_failed_turn_marks_the_chat_as_unusable():
    rec = Recorder()
    backend = FakeBackend([[LLMError("timeout", "no answer")]])
    backend.store.chats["s1"] = []
    await Agent(backend, run_id="r2", chain=Chain("r1", "s1", images=2, start=True)).plan(
        PlanEnv(), direct, rec, feedback=feedback())
    assert dict(rec.events)["llm_error"]["session_id"] is None


def test_a_verdict_that_never_ended_leaves_no_session(tmp_path):
    folder = tmp_path / "20261018-100000"
    folder.mkdir()
    events = [{"kind": "run_start", "mode": "plan", "chain": folder.name, "session_id": None},
              {"kind": "plan", "keyframes": [], "view": {}, "session_id": "s1", "chat_images": 2, "chat_start": True},
              {"kind": "exec_start"}, {"kind": "exec_end", "status": "failed"},
              {"kind": "verdict_start", "session_id": None}]  # the server died during the verdict call
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    found = RunStore(tmp_path).plan(folder.name)
    assert found["session"] is None
    (folder / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events[:-1]))
    found = RunStore(tmp_path).plan(folder.name)
    assert (found["session"], found["chat_images"], found["chat_start"]) == ("s1", 2, True)


def test_a_verdict_marks_its_turn_before_the_call(tmp_path):
    backend = FakeBackend([good_plan(), judged()])
    with app(tmp_path, backend) as c:
        run_id = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        execute(c, run_id)
    kinds = [e["kind"] for e in run_events(tmp_path, run_id)]
    assert kinds[-3:] == ["exec_end", "verdict_start", "verdict"]


def test_a_stopped_plan_turn_leaves_the_chain_without_a_chat(tmp_path):
    from tests.fakes import HangingBackend
    from tests.test_continue_api import factory
    backend = FakeBackend([good_plan(), judged()])
    with app(tmp_path, backend) as c:
        first = c.post("/api/plan", json={"instruction": PICK, "seed": 5}).json()["run_id"]
        wait_idle(c)
        execute(c, first)
    from fastapi.testclient import TestClient

    from armmind.server.app import create_app
    with TestClient(create_app(HangingBackend(), factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")) as c:
        assert c.post(f"/api/plan/{first}/replan", json={"mode": "start"}).status_code == 200
        c.post("/api/stop")
        wait_idle(c)
    with app(tmp_path, FakeBackend([good_plan()])) as c:  # the stopped round 2 is newer: round 1 branches
        third = replan(c, first)
    evs = run_events(tmp_path, third)
    assert next(e for e in evs if e["kind"] == "chat_lost")["message"] == BRANCH_CHAT


# ---- 6. One shared sessions folder ----
async def test_every_chain_uses_the_shared_sessions_folder(tmp_path, monkeypatch):
    from armmind.llm.claude_cli import ClaudeCLIBackend
    from tests.test_claude_cli_session import FAKE, ask, calls
    monkeypatch.setenv("FAKE_CLAUDE_DIR", str(tmp_path))
    backend = ClaudeCLIBackend(command=FAKE, timeout_s=20, sessions_dir=tmp_path / "chats")
    for chain in ("20261018-100000", "20261018-110000"):
        conv = await backend.open("sys", chain=chain)
        await ask(conv, "one")
        await conv.close()
    assert {c["cwd"] for c in calls(tmp_path)} == {str(tmp_path / "chats")}


def test_the_readme_says_where_the_chats_live():
    from pathlib import Path
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "~/.claude/projects/" in readme


# ---- 7. The note keeps the thinking state ----
def test_the_lost_chat_note_keeps_the_thinking_entry():
    handler = _body(_web("app.js"), "function onEvent(ev, replay = false)")
    lost = handler[handler.index('ev.kind === "chat_lost"'):handler.index('ev.kind === "loop_end"')]
    assert "endThinking()" not in lost and "thinking.item" in lost


# ---- 9. Each CLI process has a temp folder of its own ----
@pytest.fixture
def cli(tmp_path, monkeypatch):
    from armmind.llm.claude_cli import ClaudeCLIBackend
    from tests.test_claude_cli_session import FAKE
    monkeypatch.setenv("FAKE_CLAUDE_DIR", str(tmp_path))
    return ClaudeCLIBackend(command=FAKE, timeout_s=20, sessions_dir=tmp_path / "chats", temp_dir=tmp_path / "tmp")


async def test_a_step_chat_deletes_its_temp_folder_on_close(cli, tmp_path):
    from pathlib import Path

    from tests.test_claude_cli_session import ask, calls
    conv = await cli.open("sys")
    await ask(conv, "one")
    tmp = Path(calls(tmp_path)[0]["tmpdir"])
    assert tmp.is_dir() and tmp.is_relative_to(Path(calls(tmp_path)[0]["cwd"]))
    await conv.close()
    assert not tmp.exists()


async def test_a_chain_keeps_its_temp_folder_until_its_next_chat(cli, tmp_path):
    from pathlib import Path

    from tests.test_claude_cli_session import SCHEMA_B, ask, calls
    conv = await cli.open("sys", chain="20261018-100000")
    await ask(conv, "one")
    await ask(conv, "two", SCHEMA_B)  # a second process of the same chat
    first, second = (Path(c["tmpdir"]) for c in calls(tmp_path))
    assert first == second and first.is_relative_to(tmp_path / "tmp")
    other = await cli.open("sys", chain="20261018-100000-2")  # a run id that starts with the first one
    await ask(other, "x")
    await other.close()
    await conv.close()
    assert first.is_dir()  # a later --resume can need the files
    again = await cli.open("sys", chain="20261018-100000", resume=conv.session_id)
    await ask(again, "three")
    await again.close()
    assert not first.exists() and Path(calls(tmp_path)[-1]["tmpdir"]).is_dir()
    assert Path(calls(tmp_path)[2]["tmpdir"]).is_dir()  # the other chain keeps its folder


def test_a_branch_round_on_the_server_has_no_session_to_resume(tmp_path):
    """The page says "the same chat" only for a run_start with a session id: a branch must not have one."""
    backend = FakeBackend([*three_rounds(), plan_body("branch")])
    with app(tmp_path, backend) as c:
        ids = chain_of_3(c)
        branch = replan(c, ids[0])
    start = run_events(tmp_path, branch)[0]
    assert start["parent"] == ids[0] and start["chain"] == ids[0] and start["session_id"] is None
    assert run_events(tmp_path, ids[1])[0]["session_id"] == "fake-session-1"  # a round that continues has one
    js = _web("app.js")
    assert "const same = chained && Boolean(ev.session_id);" in _body(js, "function onRunStart(ev, chained = false)")


def test_the_newest_round_of_a_branch_resumes_its_own_chat(tmp_path):
    backend = FakeBackend([*three_rounds(), plan_body("b2"), judged(), plan_body("b3")])
    with app(tmp_path, backend) as c:
        ids = chain_of_3(c)
        b2 = replan(c, ids[0])  # round 1 has a child: a branch with a new chat
        branch_session = next(e for e in run_events(tmp_path, b2) if e["kind"] == "plan")["session_id"]
        execute(c, b2)  # the verdict of b2: b2 has no child, so it resumes the chat of the branch
        b3 = replan(c, b2)
    assert [o["resume"] for o in backend.opens[-2:]] == [branch_session, branch_session]
    assert not [e for e in run_events(tmp_path, b3) if e["kind"] == "chat_lost"]
    verdict = next(e for e in run_events(tmp_path, b2) if e["kind"] == "verdict")
    assert verdict["session_id"] is not None
