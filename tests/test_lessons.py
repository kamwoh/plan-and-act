"""The saved lessons of each scene."""
import json
import logging

import pytest

from armmind.agent.loop import Agent, AgentConfig
from armmind.llm.fake import FakeBackend
from armmind.store.lessons import MAX_SAVED, LessonStore
from tests.fakes import FakeEnv, Recorder, direct, good


def test_missing_file_is_an_empty_list(tmp_path):
    assert LessonStore(tmp_path / "lessons").load("cubes") == []


def test_append_writes_items_with_run_and_outcome(tmp_path):
    store = LessonStore(tmp_path / "lessons")
    store.append("cubes", ["a", "b"], run="r1", outcome="success")
    store.append("cubes", ["b", "c"], run="r2", outcome="step_limit")
    items = store.load("cubes")
    assert [i["text"] for i in items] == ["a", "b", "c"]  # "b" is there already
    assert items[0]["run"] == "r1" and items[0]["outcome"] == "success" and isinstance(items[0]["t"], float)
    assert items[2]["run"] == "r2" and items[2]["outcome"] == "step_limit"
    assert json.loads((tmp_path / "lessons" / "cubes.json").read_text()) == items
    assert store.load("tray") == []


def test_append_keeps_the_newest_30(tmp_path):
    store = LessonStore(tmp_path)
    store.append("cubes", [f"l{i}" for i in range(25)], run="r1", outcome="done")
    store.append("cubes", [f"m{i}" for i in range(10)], run="r2", outcome="done")
    texts = store.texts("cubes")
    assert MAX_SAVED == 30 and len(texts) == 30
    assert texts[0] == "l5" and texts[-1] == "m9"


@pytest.mark.parametrize("body", ["{not json", '{"a": 1}', '[{"no": "text"}]', "[1, 2]"])
def test_corrupt_file_is_an_empty_list_and_a_warning(tmp_path, caplog, body):
    (tmp_path / "cubes.json").write_text(body)
    with caplog.at_level(logging.WARNING):
        assert LessonStore(tmp_path).load("cubes") == []
    assert "cubes.json" in caplog.text


def test_the_write_is_atomic(tmp_path, monkeypatch):
    import os
    store = LessonStore(tmp_path)
    store.append("cubes", ["old"], run="r1", outcome="done")
    calls = []

    def broken(src, dst):
        calls.append((src, dst))
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        store.append("cubes", ["new"], run="r2", outcome="done")
    monkeypatch.undo()
    assert calls and str(calls[0][1]).endswith("cubes.json") and calls[0][0] != calls[0][1]
    assert store.texts("cubes") == ["old"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cubes.json"]  # no temp file is left


def test_delete_one_and_clear(tmp_path):
    store = LessonStore(tmp_path)
    store.append("cubes", ["a", "b", "c"], run="r", outcome="done")
    store.delete("cubes", 1)
    assert store.texts("cubes") == ["a", "c"]
    with pytest.raises(IndexError):
        store.delete("cubes", 5)
    store.delete("cubes")
    assert store.texts("cubes") == []


async def test_agent_saves_the_run_lessons_at_the_end(tmp_path):
    store = LessonStore(tmp_path)
    store.append("cubes", ["old lesson"], run="r0", outcome="done")
    backend, rec = FakeBackend([good(lessons=["new lesson"]), good()]), Recorder()
    status = await Agent(backend, AgentConfig(max_steps=2), store=store, run_id="r1").run(FakeEnv(), direct, rec)
    assert "Lessons:\n- old lesson\n- new lesson\n" in backend.sent[1][0].text
    items = store.load("cubes")
    assert [i["text"] for i in items] == ["old lesson", "new lesson"]
    assert items[1]["run"] == "r1" and items[1]["outcome"] == status == "step_limit"
    start = dict(rec.events)["run_start"]
    assert start["use_lessons"] is True and start["saved_lessons"] == ["old lesson"]


async def test_use_lessons_false_hides_the_saved_lessons_but_saves_new_ones(tmp_path):
    store = LessonStore(tmp_path)
    store.append("cubes", ["old lesson"], run="r0", outcome="done")
    backend, rec = FakeBackend([good(lessons=["new lesson"]), good()]), Recorder()
    await Agent(backend, AgentConfig(max_steps=2), store=store, use_lessons=False).run(FakeEnv(), direct, rec)
    assert all("old lesson" not in m.text for m, _ in backend.sent)
    assert "- new lesson" in backend.sent[1][0].text
    assert store.texts("cubes") == ["old lesson", "new lesson"]
    start = dict(rec.events)["run_start"]
    assert start["use_lessons"] is False and start["saved_lessons"] == []


async def test_a_corrupt_store_does_not_stop_the_run(tmp_path):
    (tmp_path / "cubes.json").write_text("{oops")
    store = LessonStore(tmp_path)
    status = await Agent(FakeBackend([good(lessons=["x"])]), AgentConfig(max_steps=1), store=store).run(
        FakeEnv(), direct, Recorder())
    assert status == "step_limit" and store.texts("cubes") == ["x"]
