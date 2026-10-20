import json

import pytest

from plan_and_act.store.runlog import RunLog


def test_event_writes_jsonl_and_png(tmp_path):
    log = RunLog(tmp_path)
    rec = log.event("observation", {"step": 3, "top_png": b"\x89PNGdata", "note": "x"})
    log.close()
    assert rec["top_png"] == "step003_top.png"
    assert (log.dir / "step003_top.png").read_bytes() == b"\x89PNGdata"
    line = json.loads((log.dir / "events.jsonl").read_text().strip())
    assert line["kind"] == "observation" and line["note"] == "x"


def test_two_logs_in_the_same_second_get_different_folders(tmp_path):
    a, b = RunLog(tmp_path), RunLog(tmp_path)
    assert a.dir != b.dir
    a.close()
    b.close()


def test_reserved_keys_are_refused(tmp_path):
    log = RunLog(tmp_path)
    for key in ("t", "kind", "type"):
        with pytest.raises(ValueError):
            log.event("x", {key: 1})
    log.close()


def test_folder_suffix_has_at_most_4_digits(tmp_path, monkeypatch):
    from pathlib import Path

    tried = []

    def taken(self, *a, **kw):
        tried.append(self.name)
        raise FileExistsError(self)

    monkeypatch.setattr(Path, "mkdir", taken)
    with pytest.raises(RuntimeError):
        RunLog(tmp_path)
    assert len(tried) == 10000 and tried[-1].endswith("-9999")
