import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from armmind.llm.fake import FakeBackend
from armmind.llm.protocol import LLMError
from armmind.scenes import SCENES
from armmind.server.app import Hub, create_app
from tests.fakes import FakeEnv, HangingBackend, bad, good

TASKS = SCENES["cubes"].tasks
REACH = TASKS["reach"].instruction


def make_env(scene, task, instruction, succeed_after=1, cls=FakeEnv):
    env = cls(succeed_after=succeed_after, has_check=task is not None)
    env.instruction, env.check_name = instruction, task.name if task else None
    env.scene_name, env.scene_text = scene.name, scene.describe()
    return env


def fake_factory(scene, task, instruction, seed, show_positions, on_frame):
    on_frame("front", b"\xff\xd8jpeg")
    on_frame("top", b"\xff\xd8jpeg")
    return make_env(scene, task, instruction)


def wait_idle(client, timeout=5.0):
    end = time.monotonic() + timeout
    while client.get("/api/state").json()["running"]:
        assert time.monotonic() < end, "run did not finish"
        time.sleep(0.02)


def test_tasks_and_index(tmp_path):
    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        names = [t["name"] for t in c.get("/api/tasks").json()]
        assert names == ["reach", "pick", "stack"]
        assert c.get("/").status_code == 200


def test_run_writes_log_and_sends_events(tmp_path):
    with TestClient(create_app(FakeBackend([good()]), fake_factory, tmp_path)) as c:
        with c.websocket_connect("/ws") as ws:
            r = c.post("/api/run", json={"instruction": f"  {REACH}  "})
            assert r.status_code == 200
            types, kinds, cams = [], [], []
            while "run_end" not in kinds:
                msg = ws.receive_json()
                assert msg["type"] in ("frame", "event")
                types.append(msg["type"])
                if msg["type"] == "frame":
                    cams.append(msg["cam"])
                if msg["type"] == "event":
                    kinds.append(msg["kind"])
        wait_idle(c)
    assert "frame" in types and "step" in kinds
    assert cams == ["front", "top"]
    events = [json.loads(x) for x in (next(tmp_path.iterdir()) / "events.jsonl").read_text().splitlines()]
    assert events[0]["kind"] == "run_start" and events[0]["instruction"] == REACH
    assert events[-1]["kind"] == "run_end" and events[-1]["status"] == "success"


def test_bad_reply_and_auth_error_reach_the_page(tmp_path):
    replies = [bad(), bad(), [LLMError("auth", "please log in")]]
    with TestClient(create_app(FakeBackend(replies), fake_factory, tmp_path)) as c:
        with c.websocket_connect("/ws") as ws:
            assert c.post("/api/run", json={"instruction": REACH, "max_steps": 2}).status_code == 200
            events = []
            while not events or events[-1]["kind"] != "run_end":
                msg = ws.receive_json()
                if msg["type"] == "event":
                    events.append(msg)
        wait_idle(c)
    assert all(e["type"] == "event" for e in events)
    by_kind = {e["kind"]: e for e in events}
    assert by_kind["bad_reply"]["step"] == 1 and by_kind["bad_reply"]["error"]
    err = by_kind["llm_error"]
    assert err["step"] == 2 and err["error_kind"] == "auth" and err["message"] == "please log in"
    assert events[-1]["kind"] == "run_end"


@pytest.mark.parametrize("body", [{}, {"instruction": ""}, {"instruction": "   "}, {"instruction": "x" * 501}])
def test_bad_instruction_is_422(tmp_path, body):
    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json=body).status_code == 422


def _run_start(tmp_path, instruction, **extra):
    with TestClient(create_app(FakeBackend([good(done=True)]), fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": instruction, **extra}).status_code == 200
        wait_idle(c)
    return _events(tmp_path)[0]


def test_example_instruction_turns_the_check_on(tmp_path):
    start = _run_start(tmp_path, f"  {REACH}  ")
    assert start["kind"] == "run_start" and start["check"] == "reach"


def test_custom_instruction_has_no_check(tmp_path):
    start = _run_start(tmp_path, "Touch the blue cube.")
    assert start["check"] is None and start["instruction"] == "Touch the blue cube."
    assert _events(tmp_path)[-1]["status"] == "done"


def test_preview_returns_both_images(tmp_path):
    calls = []

    def preview(scene, seed):
        calls.append((scene.name, seed))
        return {"front": "ZnJvbnQ=", "top": "dG9w"}

    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path, preview=preview)) as c:
        r = c.get("/api/preview", params={"seed": 3})
        assert r.status_code == 200 and r.json() == {"front": "ZnJvbnQ=", "top": "dG9w"}
        assert c.get("/api/preview", params={"seed": -1}).status_code == 422
        assert c.get("/api/preview", params={"scene": "tray", "seed": 2}).status_code == 200
        assert c.get("/api/preview", params={"scene": "nope", "seed": 2}).status_code == 404
    assert calls == [("cubes", 3), ("tray", 2)]


def test_preview_is_refused_while_running(tmp_path):
    with TestClient(create_app(HangingBackend(), fake_factory, tmp_path, preview=lambda scene, s: {})) as c:
        assert c.post("/api/run", json={"instruction": REACH}).status_code == 200
        assert c.get("/api/preview", params={"seed": 0}).status_code == 409
        c.post("/api/stop")
        wait_idle(c)


def test_second_run_is_rejected(tmp_path):
    with TestClient(create_app(HangingBackend(), fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": REACH}).status_code == 200
        assert c.post("/api/run", json={"instruction": TASKS["pick"].instruction}).status_code == 409
        c.post("/api/stop")
        wait_idle(c)
        assert c.get("/api/state").json()["running"] is False


async def test_hub_drops_dead_socket():
    class Dead:
        async def send_json(self, msg):
            raise RuntimeError("closed")

    class Live:
        def __init__(self):
            self.got = []

        async def send_json(self, msg):
            self.got.append(msg)

    hub, dead, live = Hub(), Dead(), Live()
    hub.clients |= {dead, live}
    await hub.publish({"type": "event"})
    assert dead not in hub.clients and live.got == [{"type": "event"}]


def _events(tmp_path):
    folder = next(tmp_path.iterdir())
    return [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines()]


def test_factory_error_ends_run_with_error(tmp_path):
    def broken(scene, task, instruction, seed, show_positions, on_frame):
        raise ValueError("no renderer")

    with TestClient(create_app(FakeBackend([]), broken, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": REACH}).status_code == 200
        wait_idle(c)
        assert c.get("/api/state").json()["running"] is False
    last = _events(tmp_path)[-1]
    assert last["kind"] == "run_end" and last["status"] == "error"
    assert "ValueError: no renderer" in last["message"]


class CloseFails(FakeEnv):
    def close(self):
        raise RuntimeError("teardown failed")


def test_close_error_is_logged_and_log_is_closed(tmp_path):
    def factory(scene, task, instruction, seed, show_positions, on_frame):
        return make_env(scene, task, instruction, cls=CloseFails)

    with TestClient(create_app(FakeBackend([good()]), factory, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": REACH}).status_code == 200
        wait_idle(c)
        assert c.get("/api/state").json()["running"] is False
    events = _events(tmp_path)
    assert events[-1]["kind"] == "cleanup_error" and "teardown failed" in events[-1]["message"]
    assert any(e["kind"] == "run_end" and e["status"] == "success" for e in events)
    runner = c.app.state.runner
    assert runner.task.done() and runner.task.exception() is None


def test_cross_site_requests_are_refused(tmp_path):
    with TestClient(create_app(FakeBackend([good()]), fake_factory, tmp_path)) as c:
        evil = {"Origin": "http://evil.example"}
        assert c.post("/api/run", json={"instruction": REACH}, headers=evil).status_code == 403
        assert c.post("/api/stop", headers=evil).status_code == 403
        with pytest.raises(WebSocketDisconnect) as info, c.websocket_connect("/ws", headers=evil):
            pass
        assert info.value.code == 1008
        ok = c.post("/api/run", json={"instruction": REACH}, headers={"Origin": "http://testserver"})
        assert ok.status_code == 200
        wait_idle(c)


@pytest.mark.assets
def test_real_preview_returns_decodable_images(tmp_path):
    import base64
    import io

    from PIL import Image

    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        body = c.get("/api/preview", params={"seed": 1}).json()
    front = Image.open(io.BytesIO(base64.b64decode(body["front"])))
    top = Image.open(io.BytesIO(base64.b64decode(body["top"])))
    assert front.format == "JPEG" and top.format == "PNG" and front.size == top.size == (640, 480)


def test_custom_system_prompt_reaches_the_backend_and_the_log(tmp_path):
    backend = FakeBackend([good()])
    with TestClient(create_app(backend, fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": REACH, "system_prompt": "  Be brief.  "}).status_code == 200
        wait_idle(c)
    assert backend.systems == ["Be brief."]
    assert _events(tmp_path)[0]["system_prompt"] == backend.systems[0]


@pytest.mark.parametrize("value", [None, "", "   \n"])
def test_empty_system_prompt_uses_the_default(tmp_path, value):
    from armmind.agent.prompts import SYSTEM
    backend = FakeBackend([good()])
    with TestClient(create_app(backend, fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": REACH, "system_prompt": value}).status_code == 200
        wait_idle(c)
    assert backend.systems == [SYSTEM] and _events(tmp_path)[0]["system_prompt"] == SYSTEM


def test_system_prompt_over_10000_chars_is_422(tmp_path):
    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": REACH, "system_prompt": "x" * 10001}).status_code == 422
        assert c.post("/api/run", json={"instruction": REACH, "system_prompt": "x" * 10000}).status_code == 200
        c.post("/api/stop")
        wait_idle(c)


def test_prompt_endpoint_returns_system_and_an_example(tmp_path):
    from armmind.agent.prompts import SYSTEM
    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        body = c.get("/api/prompt").json()
    assert body["system"] == SYSTEM
    ex = body["step_example"]
    assert TASKS["stack"].instruction in ex and "Step 1 of 30." in ex
    assert "(0.300, 0.000, 0.400)" in ex and "touching: nothing" in ex


def test_scenes_list_every_scene_with_its_tasks(tmp_path):
    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        body = c.get("/api/scenes").json()
    assert [s["name"] for s in body] == ["cubes", "tray", "zones", "shapes", "drawer", "buttons", "kitchen",
                                           "toys"]
    tray = body[1]
    assert tray["title"] == "Tray" and tray["description"] == SCENES["tray"].describe()
    assert tray["cameras"] == {"front": "Front", "top": "Top · grid"}
    assert body[4]["cameras"]["front"] == "Side"
    assert tray["tasks"] == [{"name": t.name, "instruction": t.instruction} for t in SCENES["tray"].tasks.values()]


def test_unknown_scene_is_404(tmp_path):
    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"instruction": REACH, "scene": "nope"}).status_code == 404
        assert c.get("/api/state").json()["running"] is False


def test_tray_example_checks_only_in_the_tray_scene(tmp_path):
    text = SCENES["tray"].tasks["red_in_tray"].instruction
    start = _run_start(tmp_path / "a", text, scene="tray")
    assert start["check"] == "red_in_tray" and start["scene"] == "tray"
    start = _run_start(tmp_path / "b", text, scene="cubes")
    assert start["check"] is None and start["scene"] == "cubes"


def test_run_defaults_to_the_cubes_scene(tmp_path):
    assert _run_start(tmp_path, REACH)["scene"] == "cubes"


def test_preview_without_assets_is_503_with_the_message(tmp_path):
    from armmind.sim.assets import AssetsMissing

    def preview(scene, seed):
        raise AssetsMissing("scanned object X not found. Run scripts/fetch_objects.sh")

    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path, preview=preview)) as c:
        r = c.get("/api/preview", params={"scene": "kitchen"})
    assert r.status_code == 503 and "fetch_objects.sh" in r.json()["detail"]


def test_page_shows_the_preview_error():
    js = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text()
    assert "r.status === 503" in js and "showPreviewError" in js


def _lesson_app(tmp_path, backend=None):
    return create_app(backend or FakeBackend([]), fake_factory, tmp_path / "runs", lessons_dir=tmp_path / "lessons")


def test_run_saves_lessons_and_the_api_lists_and_deletes_them(tmp_path):
    backend = FakeBackend([good(lessons=["a"]), good(lessons=["b", "c"])])
    with TestClient(_lesson_app(tmp_path, backend)) as c:
        assert c.post("/api/run", json={"instruction": "Touch it.", "max_steps": 2}).status_code == 200
        wait_idle(c)
        body = c.get("/api/lessons", params={"scene": "cubes"}).json()
        assert [i["text"] for i in body] == ["a", "b", "c"] and body[0]["run"] and body[0]["outcome"] == "step_limit"
        assert c.get("/api/lessons", params={"scene": "tray"}).json() == []
        assert c.delete("/api/lessons", params={"scene": "cubes", "index": 1}).status_code == 200
        assert [i["text"] for i in c.get("/api/lessons", params={"scene": "cubes"}).json()] == ["a", "c"]
        assert c.delete("/api/lessons", params={"scene": "cubes", "index": 9}).status_code == 404
        assert c.delete("/api/lessons", params={"scene": "cubes"}).status_code == 200
        assert c.get("/api/lessons", params={"scene": "cubes"}).json() == []
    assert (tmp_path / "lessons" / "cubes.json").exists()


def test_lessons_api_unknown_scene_is_404(tmp_path):
    with TestClient(_lesson_app(tmp_path)) as c:
        assert c.get("/api/lessons", params={"scene": "nope"}).status_code == 404
        assert c.delete("/api/lessons", params={"scene": "nope"}).status_code == 404


def test_lessons_delete_is_refused_during_a_run_and_cross_site(tmp_path):
    with TestClient(_lesson_app(tmp_path, HangingBackend())) as c:
        evil = {"Origin": "http://evil.example"}
        assert c.request("DELETE", "/api/lessons", params={"scene": "cubes"}, headers=evil).status_code == 403
        assert c.post("/api/run", json={"instruction": REACH}).status_code == 200
        assert c.delete("/api/lessons", params={"scene": "cubes"}).status_code == 409
        assert c.get("/api/lessons", params={"scene": "cubes"}).status_code == 200
        c.post("/api/stop")
        wait_idle(c)


def test_use_lessons_reaches_the_run_start(tmp_path):
    (tmp_path / "lessons").mkdir()
    (tmp_path / "lessons" / "cubes.json").write_text(json.dumps([{"text": "old", "run": "r", "outcome": "done", "t": 1.0}]))
    for flag, saved in ((False, []), (True, ["old"])):
        backend = FakeBackend([good(done=True)])
        with TestClient(_lesson_app(tmp_path, backend)) as c:
            assert c.post("/api/run", json={"instruction": "Touch it.", "use_lessons": flag}).status_code == 200
            wait_idle(c)
            run_dir = Path(c.get("/api/state").json()["run_dir"])
        start = json.loads((run_dir / "events.jsonl").read_text().splitlines()[0])
        assert start["use_lessons"] is flag and start["saved_lessons"] == saved
        assert ("old" in backend.sent[0][0].text) is flag


def test_lessons_dir_defaults_to_next_to_the_runs(tmp_path):
    app = create_app(FakeBackend([]), fake_factory, tmp_path / "runs")
    assert app.state.runner.lessons.root == tmp_path / "lessons"
