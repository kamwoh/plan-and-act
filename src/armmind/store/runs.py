"""Read the saved runs: a summary of each run, the events of one run and its images."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

RUN_ID = re.compile(r"^\d{8}-\d{6}(-\d+)?$")  # the folder names that RunLog writes
IMAGE_NAME = re.compile(r"^step\d{3}_(front|top)\.png$")
STEP_KINDS = ("step", "bad_reply", "llm_error")  # one of these ends each step
MAX_RUNS = 200


def read_events(path: Path) -> Iterator[dict]:
    """Yield each event of an events.jsonl file. A line that is not a JSON object is skipped."""
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict):
                yield ev


def _num(value) -> float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def summarize(name: str, events: Iterator[dict]) -> dict:
    s = {"id": name, "started": None, "scene": "cubes", "instruction": "", "check": None,
         "steps": 0, "cost_usd": 0.0, "in_tokens": 0, "out_tokens": 0, "status": "unfinished"}
    for ev in events:
        kind = ev.get("kind")
        if s["started"] is None:
            s["started"] = ev.get("t")
        if kind == "run_start":
            s["started"] = ev.get("t", s["started"])
            s["scene"] = ev.get("scene") or "cubes"  # runs from before the scene field
            s["instruction"] = ev.get("instruction") or ""
            s["check"] = ev.get("check")
        elif kind in STEP_KINDS:
            s["steps"] += 1
            u = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
            s["cost_usd"] += _num(u.get("cost_usd", ev.get("cost_usd")))  # old runs have no usage
            s["in_tokens"] += sum(_num(u.get(k)) for k in ("in", "cache_write", "cache_read"))
            s["out_tokens"] += _num(u.get("out"))
        elif kind == "run_end":
            s["status"] = ev.get("status") or "unfinished"
    s["cost_usd"] = round(s["cost_usd"], 6)
    return s


def _order(name: str) -> tuple[str, int]:  # "20261017-142412-2" -> ("20261017-142412", 2)
    return name[:15], int(name[16:] or 0)


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, tuple[tuple[int, int], dict]] = {}

    def folder(self, run_id: str) -> Path | None:
        """The folder of a run, or None for a bad id or a path out of the runs folder."""
        if not RUN_ID.match(run_id):
            return None
        try:
            root = self.root.resolve()
            folder = (self.root / run_id).resolve()
        except OSError:
            return None
        if folder.parent != root or not folder.is_dir():
            return None
        return folder

    def image(self, run_id: str, name: str) -> Path | None:
        folder = self.folder(run_id)
        if folder is None or not IMAGE_NAME.match(name):
            return None
        try:
            path = (folder / name).resolve()
        except OSError:
            return None
        return path if path.parent == folder and path.is_file() else None

    def summary(self, run_id: str) -> dict | None:
        folder = self.folder(run_id)
        if folder is None:
            return None
        path = folder / "events.jsonl"
        try:
            st = path.stat()
            key = (st.st_mtime_ns, st.st_size)
            hit = self._cache.get(run_id)
            if hit is not None and hit[0] == key:
                return dict(hit[1])
            s = summarize(run_id, read_events(path))
        except OSError:  # no events.jsonl, or it is not a readable file
            return None
        self._cache[run_id] = (key, s)
        return dict(s)

    def runs(self, active: str | None = None) -> list[dict]:
        try:
            names = [p.name for p in self.root.iterdir() if RUN_ID.match(p.name)]
        except OSError:
            return []
        out = []
        for name in sorted(names, key=_order, reverse=True):
            s = self.summary(name)
            if s is None:
                continue
            s["active"] = name == active
            out.append(s)
            if len(out) == MAX_RUNS:
                break
        return out

    def events(self, run_id: str, image_url) -> list[dict] | None:
        """The events of a run as the page gets them on /ws. Image file names become URLs."""
        folder = self.folder(run_id)
        if folder is None:
            return None
        try:
            events = list(read_events(folder / "events.jsonl"))
        except OSError:
            return None
        out = []
        for ev in events:
            msg = {**ev, "type": "event"}
            if msg.get("kind") == "run_start" and not msg.get("scene"):
                msg["scene"] = "cubes"
            for key, value in ev.items():
                if key.endswith("_png") and isinstance(value, str):
                    msg[key] = image_url(run_id, value) if IMAGE_NAME.match(value) else None
            out.append(msg)
        return out
