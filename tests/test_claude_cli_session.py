"""The CLI backend keeps the chat of a chain in a CLI session: a fixed folder, the session id and --resume."""
import json
import sys
from pathlib import Path

import pytest

from armmind.llm.claude_cli import ClaudeCLIBackend
from armmind.llm.protocol import Final, LLMError, Message

FAKE = [sys.executable, str(Path(__file__).with_name("fake_claude.py"))]
SCHEMA_A = {"type": "object", "properties": {"seen": {"type": "integer"}}}
SCHEMA_B = {"type": "object", "properties": {"texts": {"type": "array"}}}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_DIR", str(tmp_path))
    return ClaudeCLIBackend(command=FAKE, timeout_s=20, sessions_dir=tmp_path / "chats")


def calls(tmp_path):
    return [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text().splitlines()]


async def ask(conv, text, schema=SCHEMA_A):
    events = [e async for e in conv.send(Message("user", text), schema)]
    final = next((e for e in events if isinstance(e, Final)), None)
    error = next((e for e in events if isinstance(e, LLMError)), None)
    return final.parsed if final else None, error


async def test_a_chain_chat_keeps_its_session_in_a_fixed_folder(fake, tmp_path):
    conv = await fake.open("sys", chain="20261018-100000")
    got, _ = await ask(conv, "one")
    assert got["seen"] == 1 and conv.session_id
    first = calls(tmp_path)[0]
    assert first["cwd"] == str(tmp_path / "chats" / "20261018-100000")
    assert "--no-session-persistence" not in first["args"] and "--resume" not in first["args"]
    await conv.close()
    assert (tmp_path / "chats" / "20261018-100000").is_dir()  # close keeps the folder of the chain


async def test_a_schema_change_resumes_the_same_session(fake, tmp_path):
    conv = await fake.open("sys", chain="c1")
    await ask(conv, "one")
    session = conv.session_id
    got, _ = await ask(conv, "two", SCHEMA_B)
    assert got["texts"] == ["one", "two"] and conv.session_id == session
    second = calls(tmp_path)[1]
    assert second["args"][second["args"].index("--resume") + 1] == session
    assert second["cwd"] == calls(tmp_path)[0]["cwd"]
    assert json.loads(second["args"][second["args"].index("--json-schema") + 1]) == SCHEMA_B
    await conv.close()


async def test_a_cancel_keeps_the_memory(fake):
    conv = await fake.open("sys", chain="c1")
    await ask(conv, "one")
    await conv.cancel()
    got, _ = await ask(conv, "two")
    assert got["seen"] == 2
    await conv.close()


async def test_a_new_conversation_resumes_a_saved_session(fake, tmp_path):
    conv = await fake.open("sys", chain="c1")
    await ask(conv, "one")
    session = conv.session_id
    await conv.close()
    again = await ClaudeCLIBackend(command=FAKE, sessions_dir=tmp_path / "chats").open("sys", chain="c1",
                                                                                         resume=session)
    got, _ = await ask(again, "two", SCHEMA_B)
    assert got["texts"] == ["one", "two"] and again.session_id == session
    assert calls(tmp_path)[-1]["args"][calls(tmp_path)[-1]["args"].index("--resume") + 1] == session
    await again.close()


async def test_a_lost_session_gives_a_resume_error(fake, tmp_path):
    conv = await fake.open("sys", chain="c1", resume="11111111-2222-3333-4444-555555555555")
    got, error = await ask(conv, "two")
    assert got is None and error.kind == "resume" and "No conversation found" in error.message
    assert conv.session_id is None  # a later send starts a new session
    got, error = await ask(conv, "three")
    assert error is None and got["seen"] == 1
    await conv.close()


async def test_without_a_chain_the_chat_is_not_saved(fake, tmp_path):
    conv = await fake.open("sys")
    got, _ = await ask(conv, "one")
    assert got["seen"] == 0  # the fake CLI saves nothing with --no-session-persistence
    first = calls(tmp_path)[0]
    assert "--no-session-persistence" in first["args"] and "--resume" not in first["args"]
    assert not first["cwd"].startswith(str(tmp_path / "chats"))
    await conv.close()
    assert not Path(first["cwd"]).exists()  # the temp folder goes


async def test_a_chain_name_must_be_a_plain_folder_name(fake):
    for bad in ("../x", "a/b", "", "."):
        with pytest.raises(ValueError):
            await fake.open("sys", chain=bad)


def test_the_default_sessions_folder_is_outside_the_repo():
    folder = ClaudeCLIBackend().sessions_dir
    assert folder == Path.home() / ".cache" / "armmind" / "sessions"
