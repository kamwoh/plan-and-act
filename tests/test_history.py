"""The run history API: GET /api/runs, /api/runs/{id} and /api/runs/{id}/images/{name}."""
import json
import os
import re

import pytest
from fastapi.testclient import TestClient

from plan_and_act.llm.fake import FakeBackend
from plan_and_act.server.app import create_app
from tests.fakes import HangingBackend
from tests.test_server import fake_factory, wait_idle

PNG = b"\x89PNG\r\n\x1a\nfake"


def write_run(root, name, events, images=()):
    folder = root / name
    folder.mkdir(parents=True)
    lines = [e if isinstance(e, str) else json.dumps(e) for e in events]
    (folder / "events.jsonl").write_text("\n".join(lines) + "\n")
    for img in images:
        (folder / img).write_bytes(PNG)
    return folder


def new_run(t=100.0, scene="kitchen", status="success"):
    usage = {"in": 2, "cache_write": 100, "cache_read": 50, "out": 30, "cost_usd": 0.25, "latency_s": 2.0}
    events = [
        {"t": t, "kind": "run_start", "scene": scene, "instruction": "Put the cup down.", "check": None, "max_steps": 30},
        {"t": t + 1, "kind": "observation", "step": 1, "front_png": "step001_front.png", "top_png": "step001_top.png"},
        {"t": t + 2, "kind": "ask", "step": 1, "attempt": 1, "prompt": "Step 1 of 30."},
        {"t": t + 3, "kind": "reply", "step": 1, "plan": "p", "actions": ["open"], "usage": usage},
        {"t": t + 4, "kind": "step", "step": 1, "plan": "p", "actions": ["open"], "results": [], "todo": [],
         "notes": "", "success": False, "done": False, "usage": usage, "cost_usd": 0.25},
        {"t": t + 5, "kind": "llm_error", "step": 2, "error_kind": "timeout", "message": "slow",
         "usage": {"in": 1, "out": 0, "cost_usd": 0.5}},
    ]
    if status:
        events.append({"t": t + 6, "kind": "run_end", "status": status})
    return events


def old_run(t=50.0):  # the format before the scene field and before usage
    return [
        {"t": t, "kind": "run_start", "instruction": "Touch the cube.", "max_steps": 30},
        {"t": t + 1, "kind": "observation", "step": 1, "front_png": "step001_front.png", "top_png": "step001_top.png"},
        {"t": t + 2, "kind": "reply", "step": 1, "plan": "p", "actions": ["open"]},
        {"t": t + 3, "kind": "step", "step": 1, "plan": "p", "actions": ["open"], "results": [], "todo": [],
         "notes": "", "success": True, "cost_usd": 0.125, "latency_s": 3.0},
        {"t": t + 4, "kind": "run_end", "status": "success"},
    ]


IMAGES = ("step001_front.png", "step001_top.png")


def client(runs, backend=None):
    return TestClient(create_app(backend or FakeBackend([]), fake_factory, runs, lessons_dir=runs.parent / "lessons"))


def test_list_is_newest_first_with_the_summary(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "20261016-181045", old_run(), IMAGES)
    write_run(runs, "20261017-142412", new_run(), IMAGES)
    write_run(runs, "20261017-142412-1", new_run(t=200.0, status="stopped"))
    with client(runs) as c:
        items = c.get("/api/runs").json()
    assert [i["id"] for i in items] == ["20261017-142412-1", "20261017-142412", "20261016-181045"]
    new = items[1]
    assert new == {"id": "20261017-142412", "started": 100.0, "scene": "kitchen", "instruction": "Put the cup down.",
                   "check": None, "steps": 2, "cost_usd": 0.75, "in_tokens": 153, "out_tokens": 30,
                   "status": "success", "active": False, "mode": "run", "exec_status": None,
                   "round": 1, "parent": None, "chain": None}
    old = items[2]
    assert old["scene"] == "cubes" and old["check"] is None and old["status"] == "success"
    assert old["steps"] == 1 and old["cost_usd"] == 0.125 and old["in_tokens"] == 0
    assert items[0]["status"] == "stopped"


def test_unfinished_run_and_a_broken_line(tmp_path):
    runs = tmp_path / "runs"
    events = new_run(status=None)
    events.insert(2, '{"t": 1, "kind": "ask", "step"')  # a line cut by a crash
    write_run(runs, "20261017-120000", events)
    with client(runs) as c:
        [item] = c.get("/api/runs").json()
        got = c.get("/api/runs/20261017-120000").json()
    assert item["status"] == "unfinished" and item["steps"] == 2
    assert [e["kind"] for e in got] == ["run_start", "observation", "ask", "reply", "step", "llm_error"]


def test_folders_without_events_are_skipped(tmp_path):
    runs = tmp_path / "runs"
    (runs / "20261017-120000").mkdir(parents=True)
    (runs / "20261017-120001" / "events.jsonl").mkdir(parents=True)  # not a readable file
    (runs / "notes.txt").write_text("x")
    write_run(runs, "20261017-120002", old_run())
    with client(runs) as c:
        assert [i["id"] for i in c.get("/api/runs").json()] == ["20261017-120002"]


def test_no_runs_folder_gives_an_empty_list(tmp_path):
    with client(tmp_path / "runs") as c:
        assert c.get("/api/runs").json() == []


def test_summary_is_cached_by_mtime_and_size(tmp_path):
    runs = tmp_path / "runs"
    path = write_run(runs, "20261017-120000", old_run()) / "events.jsonl"
    with client(runs) as c:
        assert c.get("/api/runs").json()[0]["instruction"] == "Touch the cube."
        st = path.stat()
        path.write_text(path.read_text().replace("Touch the cube.", "Touch the ball."))  # the same size
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert c.get("/api/runs").json()[0]["instruction"] == "Touch the cube."  # from the cache
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
        assert c.get("/api/runs").json()[0]["instruction"] == "Touch the ball."


def test_list_holds_at_most_200_runs(tmp_path):
    runs = tmp_path / "runs"
    for i in range(203):
        write_run(runs, f"20261017-{i:06d}", old_run(t=float(i)))
    with client(runs) as c:
        items = c.get("/api/runs").json()
    assert len(items) == 200
    assert items[0]["id"] == "20261017-000202" and items[-1]["id"] == "20261017-000003"


def test_the_active_run_is_flagged(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "20261016-181045", old_run())
    with client(runs, HangingBackend()) as c:
        assert c.post("/api/run", json={"instruction": "Touch it."}).status_code == 200
        items = c.get("/api/runs").json()
        c.post("/api/stop")
        wait_idle(c)
        after = c.get("/api/runs").json()
    assert [i["active"] for i in items] == [True, False]
    assert items[0]["status"] == "unfinished"
    assert [i["active"] for i in after] == [False, False]


def test_run_events_come_in_order_with_image_urls(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "20261016-181045", old_run(), IMAGES)
    with client(runs) as c:
        r = c.get("/api/runs/20261016-181045")
        events = r.json()
        img = c.get(events[1]["front_png"])
    assert r.status_code == 200
    assert [e["kind"] for e in events] == ["run_start", "observation", "reply", "step", "run_end"]
    assert all(e["type"] == "event" for e in events)
    assert events[0]["scene"] == "cubes"  # an old run: the scene before the scene field
    obs = events[1]
    assert obs["front_png"] == "/api/runs/20261016-181045/images/step001_front.png"
    assert obs["top_png"] == "/api/runs/20261016-181045/images/step001_top.png"
    assert "PNG" not in r.text and len(r.content) < 2000
    assert img.status_code == 200 and img.content == PNG and img.headers["content-type"] == "image/png"


@pytest.mark.parametrize("url", [
    "/api/runs/..",
    "/api/runs/../events.jsonl",
    "/api/runs/..%2F..%2Fsecret",
    "/api/runs/20261016-181045%2F..",
    "/api/runs/2026-10-16",
    "/api/runs/20261016-18104",
    "/api/runs/20261016-181045x",
    "/api/runs/20261017-000000",  # the right format, but no such run
    "/api/runs/outside",
    "/api/runs/20261016-181045/images/events.jsonl",
    "/api/runs/20261016-181045/images/step1_front.png",
    "/api/runs/20261016-181045/images/step001_side.png",
    "/api/runs/20261016-181045/images/step001_front.png.bak",
    "/api/runs/20261016-181045/images/..%2Fevents.jsonl",
    "/api/runs/20261016-181045/images/%2E%2E",
    "/api/runs/20261017-000001",  # a symlink to a folder outside the runs folder
    "/api/runs/20261017-000001/images/step001_front.png",
    "/api/runs/20261016-181045/images/step002_front.png",  # a symlink to a file outside the runs folder
])
def test_unsafe_paths_are_404(tmp_path, url):
    runs = tmp_path / "runs"
    folder = write_run(runs, "20261016-181045", old_run(), IMAGES)
    outside = write_run(tmp_path, "outside", old_run(), IMAGES)
    (tmp_path / "secret").write_text("secret")
    os.symlink(outside, runs / "20261017-000001")
    os.symlink(outside / "step001_front.png", folder / "step002_front.png")
    with client(runs) as c:
        r = c.get(url)
        listed = [i["id"] for i in c.get("/api/runs").json()]
    assert r.status_code == 404
    assert "secret" not in r.text
    assert listed == ["20261016-181045"]  # the symlinked folder is not listed


def test_history_endpoints_need_no_origin_check(tmp_path):
    runs = tmp_path / "runs"
    write_run(runs, "20261016-181045", old_run(), IMAGES)
    evil = {"Origin": "http://evil.example"}
    with client(runs) as c:
        assert c.get("/api/runs", headers=evil).status_code == 200
        assert c.post("/api/runs").status_code == 405


def test_a_real_run_shows_in_the_history(tmp_path):
    from tests.fakes import good
    runs = tmp_path / "runs"
    with client(runs, FakeBackend([good(done=True)])) as c:
        assert c.post("/api/run", json={"instruction": "Touch it."}).status_code == 200
        wait_idle(c)
        [item] = c.get("/api/runs").json()
        events = c.get(f"/api/runs/{item['id']}").json()
    assert item["status"] == "done" and item["scene"] == "cubes" and item["steps"] == 1
    assert events[0]["kind"] == "run_start" and events[-1]["kind"] == "run_end"


# ---- The page ----
def _web(name):
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "web" / name).read_text()


def _body(js, start, end="\n}\n"):
    body = js[js.index(start):]
    return body[:body.index(end)]


def test_page_has_the_history_button_and_dialog():
    html = _web("index.html")
    assert 'id="history_btn"' in html and 'id="history_dialog"' in html and 'id="history_list"' in html
    buttons = _body(html, '<div class="buttons">', "</form>")  # the switches have their own div
    after_lessons = buttons.split('id="lessons_btn"', 1)[1].split("</button>", 1)[1]
    assert after_lessons.lstrip().startswith("<button") and 'id="history_btn"' in after_lessons.split("</button>")[0]


def test_history_loader_uses_the_live_event_handler():
    js = _web("app.js")
    body = _body(js, "async function loadRun")
    assert "fetch(`/api/runs/${" in body
    assert "onEvent(ev, true)" in body  # every event goes through the live handler
    for call in ("onReply(", "onStepResults(", "onAsk(", "onRunStart(", "onRunEnd(", "addEntry("):
        assert call not in body  # no second renderer
    for call in ("resetChat()", "emptyMemo()", "resetTotals()"):
        assert call in body  # the same clear as Reset
    assert "writeScene" not in body and "localStorage" not in body


def test_replayed_events_change_no_run_state_or_saved_scene():
    js = _web("app.js")
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert re.search(r"if \(!replay\) \{[^}]*setRunning\(true\)", handler)
    assert "if (!replay && !autoLoop) setRunning(false)" in handler  # an Auto loop stays busy until loop_end
    assert "chooseKey(ev.scene, true, !replay)" in handler  # the scene and the robot of the run
    choose = _body(js, "function chooseScene(name, keepText = false, save = true)")
    assert "if (save) writeScene(sceneName)" in choose
    key = _body(js, "function chooseKey(key, keepText = false, save = true)")
    assert "chooseRobot(robot, save)" in key and "chooseScene(scene, keepText, save)" in key
    assert "if (save) store(ROBOT_KEY, robotName)" in _body(js, "function chooseRobot(name, save = true)")


def test_observation_images_are_base64_or_server_urls():
    js = _web("app.js")
    src = _body(js, "function imageSrc")
    assert 'startsWith("/api/runs/")' in src and "pngUrl(" in src
    handler = _body(js, "function onEvent(ev, replay = false)")
    assert "imageSrc(ev.front_png)" in handler and "imageSrc(ev.top_png)" in handler
    assert "img.loading = \"lazy\"" in _body(js, "function thumbs")


def test_history_rows_cannot_open_the_active_run():
    js = _web("app.js")
    body = _body(js, "function historyRow")
    assert "run.active" in body and "running" in body


@pytest.mark.parametrize("run_id", [
    "20261016-181045-" + "1" * 300,  # too long for the file system: never a 500
    "20261016-181045-12345",  # RunLog writes at most 4 digits after the second dash
    "２０２６１０１６-１８１０４５",  # Unicode digits
    "20261016-181045%0A",  # a trailing newline
])
def test_odd_run_ids_are_404(tmp_path, run_id):
    runs = tmp_path / "runs"
    write_run(runs, "20261016-181045", old_run(), IMAGES)
    with client(runs) as c:
        assert c.get(f"/api/runs/{run_id}").status_code == 404
        assert c.get(f"/api/runs/{run_id}/images/step001_front.png").status_code == 404


@pytest.mark.parametrize("name", ["step００1_front.png", "step001_front.png%0A"])
def test_odd_image_names_are_404(tmp_path, name):
    runs = tmp_path / "runs"
    write_run(runs, "20261016-181045", old_run(), IMAGES)
    with client(runs) as c:
        assert c.get(f"/api/runs/20261016-181045/images/{name}").status_code == 404


def test_store_rejects_a_trailing_newline_and_unicode_digits(tmp_path):
    from plan_and_act.store.runs import RunStore
    runs = tmp_path / "runs"
    write_run(runs, "20261016-181045", old_run(), IMAGES)
    store = RunStore(runs)
    assert store.folder("20261016-181045") is not None
    assert store.folder("20261016-181045\n") is None
    assert store.folder("２０２６１０１６-１８１０４５") is None
    assert store.image("20261016-181045", "step001_front.png\n") is None


def test_id_and_name_patterns_are_ascii_and_exact():
    from plan_and_act.store.runs import valid_image_name, valid_run_id
    assert valid_run_id("20261016-181045") and valid_run_id("20261016-181045-9999")
    for bad in ("20261016-181045\n", "２０２６１０１６-１８１０４５", "20261016-181045-12345", "x20261016-181045"):
        assert not valid_run_id(bad)
    assert valid_image_name("step001_front.png") and valid_image_name("step200_top.png")
    for bad in ("step001_front.png\n", "step００1_front.png", "step1_front.png"):
        assert not valid_image_name(bad)


def test_replay_skips_the_scroll_work_of_each_entry():
    js = _web("app.js")
    add = _body(js, "function addEntry")
    assert add.index("if (replaying)") < add.index("atBottom()")
    load = _body(js, "async function loadRun")
    assert "replaying = true" in load and "finally" in load and "replaying = false" in load
    assert load.index("replaying = false") < load.index("scrollChat(true)")


def test_step_details_use_the_old_usage_too():
    assert "const u = stepUsage(ev)" in _body(_web("app.js"), "function openStepDetails")


def test_viewed_llm_error_run_gets_the_bad_pill():
    load = _body(_web("app.js"), "async function loadRun")
    assert 'run.status === "error" || run.status === "llm_error"' in load


def test_open_history_list_refreshes_when_a_run_ends():
    js = _web("app.js")
    handler = _body(js, "function onEvent(ev, replay = false)")
    end = handler[handler.index('ev.kind === "run_end"'):]
    assert 'if (!replay && $("history_dialog").open) fillHistory()' in end
    assert "fillHistory()" in _body(js, "async function openHistory")


def test_the_history_button_has_a_short_text_label():
    html = _web("index.html")
    history = html[html.index('id="history_btn"'):]
    assert "</svg>History</button>" in history[:history.index("</button>") + 9]


def test_a_run_that_ended_on_the_page_is_not_active_in_the_list():
    # At run_end the server still cleans up and lists the run as active. The page knows it ended.
    assert "const active = run.active && running" in _body(_web("app.js"), "function historyRow")


def test_replay_leaves_out_a_bad_event_and_says_so():
    js = _web("app.js")
    body = js[js.index("async function loadRun"):]
    body = body[:body.index("\n}\n")]
    assert "skipped += 1" in body and "bad event" in body
