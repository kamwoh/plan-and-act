"""The saved lessons of each scene: lessons/<scene>.json, a list of {"text", "run", "outcome", "t"}."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

MAX_SAVED = 30  # lessons per scene; the oldest go first

log = logging.getLogger(__name__)


class LessonChanged(ValueError):
    """The lesson at an index is not the lesson that the caller saw."""


def _valid(items) -> bool:
    return isinstance(items, list) and all(isinstance(i, dict) and isinstance(i.get("text"), str) for i in items)


class LessonStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, scene: str) -> Path:
        return self.root / f"{scene}.json"

    def load(self, scene: str) -> list[dict]:
        """The saved lessons, oldest first. A missing file gives []. A corrupt file gives [] and a warning."""
        path = self._path(scene)
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as e:
            log.warning("cannot read the lessons in %s, so the list is empty: %s", path, e)
            return []
        if not _valid(items):
            log.warning("the lessons in %s are not a list of lessons, so the list is empty", path)
            return []
        return items

    def texts(self, scene: str) -> list[str]:
        return [i["text"] for i in self.load(scene)]

    def append(self, scene: str, texts: list[str], run: str, outcome: str) -> None:
        """Add the lessons that are not saved yet, then keep the newest MAX_SAVED."""
        items = self.load(scene)
        known = {i["text"] for i in items}
        now = round(time.time(), 3)
        for text in texts:
            if text not in known:
                items.append({"text": text, "run": run, "outcome": outcome, "t": now})
                known.add(text)
        self._write(scene, items[-MAX_SAVED:])

    def delete(self, scene: str, index: int | None = None, text: str | None = None) -> None:
        """Remove one lesson, or all of the scene when index is None. A bad index raises IndexError.
        With text, the lesson at the index must have this text, or LessonChanged is raised."""
        items = self.load(scene)
        if index is None:
            items = []
        elif 0 <= index < len(items):
            if text is not None and items[index]["text"] != text:
                raise LessonChanged(f"lesson {index} of scene {scene!r} changed")
            del items[index]
        else:
            raise IndexError(f"no lesson {index} in scene {scene!r}")
        self._write(scene, items)

    def _write(self, scene: str, items: list[dict]) -> None:
        """Write a temp file in the same folder, then rename it over the old file."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{scene}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self._path(scene))
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
