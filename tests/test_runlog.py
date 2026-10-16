import json

from armmind.store.runlog import RunLog


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
