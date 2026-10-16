# arm-mind

An LLM controls a simulated robot arm. Pick a task, change the instruction if you want, press Start, and watch Claude move a Franka Panda arm in MuJoCo. The physics stops while Claude thinks, so a slow model is not a problem.

## Setup (WSL2 or Linux)

    uv sync
    ./scripts/fetch_assets.sh
    uv run armmind scripted --task stack --seed 0

The second command downloads the Panda model from MuJoCo Menagerie (Apache 2.0). The third command checks the arm without a model.

You also need the Claude Code CLI, logged in. arm-mind uses it as the model backend. No API key is necessary.

## Run

    uv run armmind serve          # then open http://127.0.0.1:8780

Pick a task (reach, pick or stack), a seed and a step limit, then press Start. You can change the instruction text. The success check of the task stays the same. The left side shows the front camera live and the top camera from the last step. The right side shows Claude's to-do list, notes and each step.

The debug switch gives Claude the true cube positions. Use it to find out if a failure comes from vision or from planning.

## How it works

Each step: the app renders a front and a top camera image with a grid on the table. Claude gets the images and the gripper position, and replies with up to 8 actions: `move_to [x, y, z]`, `open`, `close`, `wait`. Code finds the joint angles (IK) and moves the arm. Each result (reached, blocked or unreachable) goes into the next prompt.

Each run writes `runs/<time>/events.jsonl` and the observation images.

## Development

    ./scripts/check.sh

Tests marked `assets` skip when the Panda model is missing.

## License

MIT. The Panda model is not in this repository. MuJoCo Menagerie supplies it under Apache 2.0.
