"""One folder for each run: events.jsonl and the observation images."""
from __future__ import annotations

import json
import time
from pathlib import Path


class RunLog:
    def __init__(self, root: Path):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for n in range(10000):  # the runs API accepts a suffix of at most 4 digits
            folder = Path(root) / (stamp if n == 0 else f"{stamp}-{n}")
            try:
                folder.mkdir(parents=True)
                break
            except FileExistsError:
                continue
        else:
            raise RuntimeError(f"no free run folder for {stamp}")
        self.dir = folder
        self._f = (folder / "events.jsonl").open("a", encoding="utf-8")

    def event(self, kind: str, data: dict) -> dict:
        if clash := {"t", "kind", "type"} & data.keys():
            raise ValueError(f"reserved event keys: {sorted(clash)}")
        record: dict = {"t": round(time.time(), 3), "kind": kind}
        for key, value in data.items():
            if isinstance(value, bytes) and key.endswith("_png"):
                name = f"step{int(data.get('step', 0)):03d}_{key.removesuffix('_png')}.png"
                (self.dir / name).write_bytes(value)
                value = name
            record[key] = value
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()
        return record

    def close(self) -> None:
        self._f.close()
