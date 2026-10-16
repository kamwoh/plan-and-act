import json
import time

from fastapi.testclient import TestClient

from armmind.llm.fake import FakeBackend
from armmind.server.app import Hub, create_app
from tests.fakes import FakeEnv, HangingBackend, good


def fake_factory(task, seed, show_positions, on_frame):
    on_frame(b"\xff\xd8jpeg")
    return FakeEnv(succeed_after=1)


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
            r = c.post("/api/run", json={"task": "reach", "instruction": "  Tap the red block.  "})
            assert r.status_code == 200
            kinds = []
            while "run_end" not in kinds:
                msg = ws.receive_json()
                kinds.append(msg.get("kind", msg["type"]))
        wait_idle(c)
    assert "frame" in kinds and "step" in kinds
    events = [json.loads(x) for x in (next(tmp_path.iterdir()) / "events.jsonl").read_text().splitlines()]
    assert events[0]["kind"] == "run_start" and events[0]["instruction"] == "Tap the red block."
    assert events[-1]["kind"] == "run_end" and events[-1]["status"] == "success"


def test_unknown_task_is_404(tmp_path):
    with TestClient(create_app(FakeBackend([]), fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"task": "fly"}).status_code == 404


def test_second_run_is_rejected(tmp_path):
    with TestClient(create_app(HangingBackend(), fake_factory, tmp_path)) as c:
        assert c.post("/api/run", json={"task": "reach"}).status_code == 200
        assert c.post("/api/run", json={"task": "pick"}).status_code == 409
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
    def broken(task, seed, show_positions, on_frame):
        raise ValueError("no renderer")

    with TestClient(create_app(FakeBackend([]), broken, tmp_path)) as c:
        assert c.post("/api/run", json={"task": "reach"}).status_code == 200
        wait_idle(c)
        assert c.get("/api/state").json()["running"] is False
    last = _events(tmp_path)[-1]
    assert last["kind"] == "run_end" and last["status"] == "error"
    assert "ValueError: no renderer" in last["message"]


class CloseFails(FakeEnv):
    def close(self):
        raise RuntimeError("teardown failed")


def test_close_error_is_logged_and_log_is_closed(tmp_path):
    def factory(task, seed, show_positions, on_frame):
        return CloseFails(succeed_after=1)

    with TestClient(create_app(FakeBackend([good()]), factory, tmp_path)) as c:
        assert c.post("/api/run", json={"task": "reach"}).status_code == 200
        wait_idle(c)
        assert c.get("/api/state").json()["running"] is False
    events = _events(tmp_path)
    assert events[-1]["kind"] == "cleanup_error" and "teardown failed" in events[-1]["message"]
    assert any(e["kind"] == "run_end" and e["status"] == "success" for e in events)
    runner = c.app.state.runner
    assert runner.task.done() and runner.task.exception() is None
