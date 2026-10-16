# arm-mind

An LLM controls a simulated robot arm. Write an instruction or pick an example, press Start, and watch Claude move a Franka Panda arm in MuJoCo. The physics stops while Claude thinks, so a slow model is not a problem.

## Setup (WSL2 or Linux)

    uv sync
    ./scripts/fetch_assets.sh
    uv run armmind scripted --task stack --seed 0

The second command downloads the Panda model from MuJoCo Menagerie (Apache 2.0). The third command checks the arm without a model.

You also need the Claude Code CLI, logged in. arm-mind uses it as the model backend. No API key is necessary.

MuJoCo renders through EGL. If EGL fails, run with `MUJOCO_GL=osmesa`. This uses the CPU and needs the OSMesa system library.

## Run

    uv run armmind serve          # then open http://127.0.0.1:8780

The `serve` command takes these options:

- `--model`: the model name for the Claude CLI.
- `--host`: the address to listen on (default 127.0.0.1).
- `--port`: the port (default 8780). The command refuses 8765, because ps1-mind uses it.
- `--runs`: the folder for run logs (default `runs`).

Type an instruction in the text box. You can also pick an example from the Examples list. This fills the box with the instruction of a built-in task: reach, pick or stack.

A line under the box shows the mode. If your text matches an example exactly, the line reads "Auto-check: reach" (or pick, or stack). The app then checks the physics and ends the run on success. If your text is different, the line reads "No auto-check". Claude then sets "done" in its reply when it thinks the task is complete, and the run ends.

Before you press Start, the page shows the starting scene for the seed in both camera views. It updates when you change the seed. The scene shows the cube layout and the arm in the ready pose.

Set a seed and a step limit, then press Start. The left side shows the front camera live and the top camera from the last step. The right side shows Claude's to-do list, notes and each step.

The debug switch gives Claude the true cube positions. Use it to find out if a failure comes from vision or from planning.

## How it works

Each step: the app renders a front and a top camera image with a grid on the table. Claude gets the images and the gripper position, and replies with up to 8 actions: `move_to [x, y, z]`, `open`, `close`, `wait`. Code finds the joint angles (IK) and moves the arm. Each result (reached, blocked or unreachable) goes into the next prompt.

Each run writes `runs/<time>/events.jsonl` and the observation images.

## Development

    ./scripts/check.sh

Tests marked `assets` skip when the Panda model is missing.

## License

MIT. The Panda model is not in this repository. MuJoCo Menagerie supplies it under Apache 2.0.
