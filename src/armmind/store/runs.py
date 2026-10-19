"""Read the saved runs: a summary of each run, the events of one run and its images."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

RUN_ID = re.compile(r"[0-9]{8}-[0-9]{6}(-[0-9]{1,4})?")  # the folder names that RunLog writes
IMAGE_NAME = re.compile(r"step[0-9]{3}_(front|top)\.png")
STEP_KINDS = ("step", "bad_reply", "llm_error", "plan")  # one of these ends each step or the plan call
MAX_RUNS = 200
MAX_CHAIN = 30  # the rounds of a chain (MAX_ROUND of the server)


def valid_run_id(name: str) -> bool:
    return RUN_ID.fullmatch(name) is not None


def valid_image_name(name: str) -> bool:
    return IMAGE_NAME.fullmatch(name) is not None


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
         "steps": 0, "cost_usd": 0.0, "in_tokens": 0, "out_tokens": 0, "status": "unfinished",
         "mode": "run", "exec_status": None, "round": 1, "parent": None}
    for ev in events:
        kind = ev.get("kind")
        if s["started"] is None:
            s["started"] = ev.get("t")
        if kind == "run_start":
            s["started"] = ev.get("t", s["started"])
            s["scene"] = ev.get("scene") or "cubes"  # runs from before the scene field
            s["instruction"] = ev.get("instruction") or ""
            s["check"] = ev.get("check")
            s["mode"] = "plan" if ev.get("mode") == "plan" else "run"
            s["round"] = ev.get("round") if isinstance(ev.get("round"), int) else 1
            s["parent"] = ev.get("parent") if isinstance(ev.get("parent"), str) else None
        elif kind in (*STEP_KINDS, "verdict"):
            s["steps"] += kind != "verdict"  # a verdict call costs, but it is not a step
            u = ev.get("usage") if isinstance(ev.get("usage"), dict) else {}
            s["cost_usd"] += _num(u.get("cost_usd", ev.get("cost_usd")))  # old runs have no usage
            s["in_tokens"] += sum(_num(u.get(k)) for k in ("in", "cache_write", "cache_read"))
            s["out_tokens"] += _num(u.get("out"))
        elif kind == "run_end":
            s["status"] = ev.get("status") or "unfinished"
        elif kind == "exec_start":
            s["exec_status"] = "unfinished"
        elif kind == "exec_end":
            s["exec_status"] = ev.get("status") or "unfinished"
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
        if not valid_run_id(run_id):
            return None
        try:
            root = self.root.resolve()
            folder = (self.root / run_id).resolve()
            return folder if folder.parent == root and folder.is_dir() else None
        except OSError:  # for example a name that is too long for the file system
            return None

    def image(self, run_id: str, name: str) -> Path | None:
        folder = self.folder(run_id)
        if folder is None or not valid_image_name(name):
            return None
        try:
            path = (folder / name).resolve()
            return path if path.parent == folder and path.is_file() else None
        except OSError:
            return None

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
            names = [p.name for p in self.root.iterdir() if valid_run_id(p.name)]
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
                    msg[key] = image_url(run_id, value) if valid_image_name(value) else None
            out.append(msg)
        return out

    def plan(self, run_id: str) -> dict | None:
        """The run_start and the plan event of a finished plan run, or None. "execution" holds the events of
        the last execution ({"start", "step", "end", "actions"}), or None when the plan has not run yet.
        "session" is the last saved chat of the chain in this run: of the plan, or of a later verdict."""
        folder = self.folder(run_id)
        if folder is None:
            return None
        start = plan = last = session = None
        executions = 0
        try:
            for ev in read_events(folder / "events.jsonl"):
                kind = ev.get("kind")
                if kind in ("run_start", "plan", "verdict") and isinstance(ev.get("session_id"), str):
                    session = ev["session_id"]
                if kind == "run_start":
                    start = ev
                elif kind == "plan":
                    plan = ev
                elif kind == "exec_start":
                    executions += 1
                    last = {"start": ev, "step": None, "end": None, "actions": 0}
                elif kind in ("step", "exec_end") and last is not None:
                    last["step" if kind == "step" else "end"] = ev
                elif kind == "action" and last is not None and last["end"] is None:
                    last["actions"] += 1  # the actions that started
        except OSError:
            return None
        if not (start and start.get("mode") == "plan" and plan and isinstance(plan.get("view"), dict)
                and isinstance(plan.get("keyframes"), list)):
            return None
        return {"start": start, "plan": plan, "folder": folder, "executions": executions, "execution": last,
                "session": session}

    def chain(self, run_id: str) -> list[str] | None:
        """The run ids of the chain of a run, from round 1 to this run, by the parent links. None for no run."""
        if self.folder(run_id) is None:
            return None
        ids = [run_id]
        while len(ids) < MAX_CHAIN:
            try:
                events = read_events(self.folder(ids[-1]) / "events.jsonl")
                try:
                    start = next(events, {})
                finally:
                    events.close()  # the file closes now, not at garbage collection
            except OSError:
                break
            parent = start.get("parent") if start.get("kind") == "run_start" else None
            if not isinstance(parent, str) or parent in ids or self.folder(parent) is None:
                break
            ids.append(parent)
        return ids[::-1]
