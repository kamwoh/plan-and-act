"""The frames of an execution that go to Claude. Plain Python only."""
from __future__ import annotations

MAX_FRAMES = 4  # the frames of one execution in a message to Claude


def pick_frames(frames: list[dict], end_images: bool) -> list[int]:
    """The indices of the frames for Claude, in order: at most MAX_FRAMES at even gaps, with the last frame.
    Each frame has "keyframe", "of" and "at" ("end" or "stop")."""
    n = len(frames)
    k = min(n, MAX_FRAMES)
    picked = [-(-(i + 1) * n // k) - 1 for i in range(k)]  # the ends of k equal parts
    last = frames[-1] if frames else None
    # The end of the last keyframe and the end image show the same moment: send the end image only.
    if end_images and last and last.get("at") == "end" and last.get("keyframe") == last.get("of"):
        picked.pop()
    return picked
