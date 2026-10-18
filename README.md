# arm-mind

Claude controls a simulated Franka Panda arm in MuJoCo, through the Claude Code CLI.

## Setup

Needs Python 3.12, `uv` and a logged-in Claude Code CLI.

    uv sync
    ./scripts/fetch_assets.sh     # Panda model
    ./scripts/fetch_objects.sh    # scanned objects for the kitchen and toys scenes

## Run

    uv run armmind serve          # http://127.0.0.1:8780

Pick a scene, type an instruction, and press Plan to see Claude's plan as a 3D video. Then press Execute and Re-plan, or set the Auto switch to repeat plan, execute and re-plan until the auto-check passes.

Run a task with a scripted solution, with no model:

    uv run armmind scripted --scene tray --task all_in_tray --seed 2

If EGL fails, set `MUJOCO_GL=osmesa`.

## Test

    ./scripts/check.sh

## License

MIT. The assets are not in this repository:

- Panda model: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Apache 2.0.
- Object scans: [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) (Downs et al., 2022), [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- MJCF conversion: [mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects) by Kevin Zakka, MIT.
