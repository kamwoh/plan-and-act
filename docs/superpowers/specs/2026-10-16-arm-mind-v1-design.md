# arm-mind v1: design

Date: 2026-10-16
Status: draft, waiting for review

## Goal

An LLM controls a simulated robot arm. You type an instruction, for example "put the red cube on the blue cube". You watch the arm in a browser and read the reasoning of the agent at each step.

Success for v1: you can watch Claude try the three tasks below, live, and read each step. Scores and benchmarks come later.

## Background

ps1-mind lets an LLM play PS1 games. The LLM is slow, and a game runs in real time. A simulator solves this: the physics stops while the LLM thinks. arm-mind uses the same agent ideas as ps1-mind (paused steps, short replies, notes, to-do list, event log) on a MuJoCo robot arm.

## Decisions

| Topic | Decision |
|---|---|
| What to build | Our own agent harness. No Inspect Robots plugin in v1. |
| Success | "Watch it work": live view and readable steps. No scores mode in v1. |
| Repository | A new repository, separate from ps1-mind. |
| Actions | Gripper targets: `move_to`, `open`, `close`, `wait`. Code solves the joint angles. |
| Observation | Two camera images with a coordinate grid, the gripper pose as text, the result of each action. Object positions stay hidden. A debug switch shows them. |
| UI | Python server with a browser page (approach A). |
| LLM | Claude through the Claude Code CLI, as in ps1-mind. No API key. |

## Scope

In v1:

- You can change the instruction text before a run. The success check of the task stays the same.
- Robot: Franka Panda arm with a parallel gripper. The model comes from MuJoCo Menagerie (Apache 2.0). Setup downloads it. The repository does not store it.
- Scene: a table and three colored cubes (red, blue, green).
- Three tasks, each with a success check in code:
  1. Reach: "touch the red cube".
  2. Pick: "lift the red cube 10 cm".
  3. Stack: "put the red cube on the blue cube".
- A seed sets the cube positions. The same seed gives the same layout.

Not in v1: scores mode, LIBERO, Inspect Robots, other robots, real hardware, a 3D view that you can rotate in the browser.

## Architecture

Python 3.12, `uv`, a project venv. The server uses port 8780. It must never use port 8765 (ps1-mind).

### Units

| Unit | Job | Depends on |
|---|---|---|
| `sim/world.py` | Loads the scene XML. Steps the physics. Resets with a seed. | `mujoco` |
| `sim/render.py` | Renders the front and top cameras to images. Draws the coordinate grid. | `world`, Pillow |
| `sim/arm.py` | Runs `move_to`, `open`, `close`, `wait`. Uses IK to get joint targets, moves smoothly, stops on contact or timeout. Returns a result. | `world` |
| `tasks/` | The three tasks: instruction text, cube layout, `success(world) -> bool`. | `world` |
| `llm/` | Claude CLI client and a fake client for tests. Copied from ps1-mind. | none |
| `agent/` | Builds the prompt, parses the JSON reply, keeps notes, to-do list and history. One `step()` is observe, ask, act. | `llm`, `Env` |
| `server/` | FastAPI. Starts and stops runs. Sends frames and step events through a WebSocket. Serves the UI. | `agent`, `sim` |
| `web/` | One page: camera view, instruction box, Start and Stop, step list, notes, to-do list, success banner. | none |
| `store/` | Writes `events.jsonl` and frame files for each run. | none |

### The Env interface

The agent uses only this interface. It does not know MuJoCo.

```python
class Env(Protocol):
    def observe(self) -> Observation: ...
    def act(self, action: Action) -> ActionResult: ...
```

- `Observation`: front image, top image, gripper `x, y, z`, gripper open or closed, results of the last step's actions. With the debug switch on, it also has the object positions.
- `Action`: one of `move_to(x, y, z)`, `open`, `close`, `wait(seconds)`.
- `ActionResult`: `reached`, `blocked` (with the contacts), or `unreachable`, plus the final gripper pose.

Tests use a fake Env. Later, LIBERO or an Inspect Robots embodiment can go behind the same interface.

### LLM reply format

```json
{
  "observation": "Red cube near the front left. Gripper is above the table centre.",
  "plan": "Move above the red cube, go down, close, lift.",
  "actions": [
    {"type": "move_to", "xyz": [0.42, -0.10, 0.15]},
    {"type": "move_to", "xyz": [0.42, -0.10, 0.02]},
    {"type": "close"},
    {"type": "move_to", "xyz": [0.42, -0.10, 0.20]}
  ],
  "notes": "optional, only when something changed",
  "todo": ["find red cube", "grasp it", "lift it"]
}
```

Units are metres, in the world frame. The grid on the images uses the same frame. A step holds up to 8 actions.

### Data flow for one step

1. `Env.observe()` returns the images, the gripper pose and the last results.
2. The agent builds the prompt and calls Claude. The physics stays paused.
3. The agent parses the reply. If the JSON is bad, it asks once more. If the second reply is also bad, it logs `bad_reply` and the step ends with no action.
4. For each action, `arm.py` runs physics steps. The server sends frames at about 30 fps, so you see the motion. A move takes a fixed time for its distance. After it, the result is `reached` if the gripper is within 2 cm of the target. If not, the result is `blocked`, with the contacts.
5. The task's `success()` runs. The step event goes to the UI and to the log.
6. The run ends on success, on the step limit, or on Stop.

### Safety limits

- `arm.py` clamps each target to a workspace box above the table.
- If IK fails, the result is `unreachable` and the arm does not move.
- The next prompt shows each of these results to the LLM.

### Log

One folder for each run: `runs/<time>/events.jsonl` and the frames. Events: `run_start`, `step`, `bad_reply`, `run_end`. `.gitignore` excludes `runs/`.

## Testing

1. Spike first: render one camera frame off screen on WSL (EGL, or CPU if EGL fails).
2. Unit tests with real MuJoCo and no LLM: IK reach, grid overlay, reply parsing, task success checks.
3. A scripted policy (hard-coded pick and place) must solve all three tasks. This proves that the arm and the tasks work before an LLM runs.
4. Agent tests with the fake LLM client.
5. A manual run with Claude on each task, watched in the browser.

## Risks

- Off-screen rendering on WSL without an NVIDIA GPU. The spike checks this first. CPU rendering is the fallback.
- The LLM can be bad at reading coordinates from images. The grid overlay and the debug switch help to find where it fails.
- The Panda IK can fail near the workspace edges. The workspace box keeps targets in a safe region.
