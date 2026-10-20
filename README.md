# plan-and-act

Claude controls a simulated Franka Panda arm in MuJoCo, through the Claude Code CLI.

## Setup

Needs Python 3.12, `uv` and a logged-in Claude Code CLI.

    uv sync
    ./scripts/fetch_assets.sh     # Panda model
    ./scripts/fetch_objects.sh    # scanned objects for the kitchen and toys scenes

## Run

    uv run plan-and-act serve          # http://127.0.0.1:8780

Pick a scene, type an instruction, and press Plan to see Claude's plan as a 3D video. Then press Execute and Re-plan, or set the Auto switch to repeat plan, execute and re-plan until the auto-check passes or Claude says done. With Continue on, a re-plan starts where the last execution ended. Plan goes on, Reset starts over. After the first round, the Plan button goes on from the newest round on the page. With the same instruction, it re-plans (↻ Re-plan). With a new instruction in the box, it says Plan from here: Claude gets the new task in the same chat. The arm goes on from where it is now. Press Reset, or change the scene or the seed, to start a new chain and a new chat from the fresh scene. All rounds of one plan share one chat with Claude. The CLI keeps these chats under ~/.claude/projects/, and its own cleanup removes old ones. After that, a re-plan starts a new chat with a summary.

Run a task with a scripted solution, with no model:

    uv run plan-and-act scripted --scene tray --task all_in_tray --seed 2

If EGL fails, set `MUJOCO_GL=osmesa`.

## Test

    ./scripts/check.sh

## License

MIT. The assets are not in this repository:

- Panda model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Apache 2.0.
- Object scans: [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) (Downs et al., 2022), [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- MJCF conversion: [mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) by Kevin Zakka, MIT.
