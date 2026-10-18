"""A stand-in for the `claude` command in the CLI backend tests. No model is called.

It reads stream-json user messages from stdin and answers each one with an init line and a result line.
The environment variable FAKE_CLAUDE_DIR names a folder for its sessions and its log:
- sessions/<id>: one line for each user message of the session
- calls.jsonl: one line for each start, with the arguments and the working folder
The structured answer {"seen": n, "texts": [...]} holds all user texts of the session so far.
"""
import json
import os
import sys
import uuid
from pathlib import Path

root = Path(os.environ["FAKE_CLAUDE_DIR"])
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as f:
    f.write(json.dumps({"args": args, "cwd": os.getcwd()}) + "\n")


def opt(name: str) -> str | None:
    return args[args.index(name) + 1] if name in args else None


sessions = root / "sessions"
sessions.mkdir(exist_ok=True)
resume = opt("--resume")
if resume is not None and not (sessions / resume).exists():
    print(json.dumps({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": None,
                      "session_id": resume, "errors": [f"No conversation found with session ID: {resume}"]}),
          flush=True)
    print(f"No conversation found with session ID: {resume}", file=sys.stderr, flush=True)
    sys.exit(1)
session = resume or str(uuid.uuid4())
persist = "--no-session-persistence" not in args
for line in sys.stdin:
    msg = json.loads(line)
    text = next(c["text"] for c in msg["message"]["content"] if c["type"] == "text")
    texts = []
    if persist:
        with (sessions / session).open("a") as f:
            f.write(json.dumps(text) + "\n")
        texts = [json.loads(x) for x in (sessions / session).read_text().splitlines()]
    print(json.dumps({"type": "system", "subtype": "init", "session_id": session}), flush=True)
    answer = {"seen": len(texts), "texts": texts}
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": json.dumps(answer),
                      "structured_output": answer, "session_id": session,
                      "usage": {"input_tokens": 1, "output_tokens": 1}}), flush=True)
