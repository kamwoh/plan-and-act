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
