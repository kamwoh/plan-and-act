# arm-mind

An LLM controls a simulated robot arm. Write an instruction or pick an example, press Start, and watch Claude move a Franka Panda arm in MuJoCo. The physics stops while Claude thinks, so a slow model is not a problem.

## Setup (WSL2 or Linux)

    uv sync
    ./scripts/fetch_assets.sh
    ./scripts/fetch_objects.sh
    uv run armmind scripted --scene cubes --task stack --seed 0

The second command downloads the Panda model from MuJoCo Menagerie (Apache 2.0). The third command downloads the 3D-scanned objects for the kitchen and toys scenes. The fourth command checks the arm without a model.

`fetch_objects.sh` reads the object folder names from `src/armmind/scenes/objects.txt`. It downloads only these folders, about 1 MB each, into `assets/scanned_objects/`. When all of them are present, it does nothing.

You also need the Claude Code CLI, logged in. arm-mind uses it as the model backend. No API key is necessary.

MuJoCo renders through EGL. If EGL fails, run with `MUJOCO_GL=osmesa`. This uses the CPU and needs the OSMesa system library.

## Run

    uv run armmind serve          # then open http://127.0.0.1:8780

The `serve` command takes these options:

- `--model`: the model name for the Claude CLI.
- `--host`: the address to listen on (default 127.0.0.1).
- `--port`: the port (default 8780). The command refuses 8765, because ps1-mind uses it.
- `--runs`: the folder for run logs (default `runs`).

Pick a scene from the Scene row first. Then type an instruction in the text box. You can also pick an example from the Examples list. This fills the box with the instruction of a built-in task of the scene.

A line under the box shows the mode. If your text matches an example of the scene exactly, the line reads "Auto-check: " and the task name. The app then checks the physics and ends the run on success. If your text is different, the line reads "No auto-check". Claude then sets "done" in its reply when it thinks the task is complete, and the run ends.

Before you press Start, the page shows the starting scene for the seed in both camera views. It updates when you change the seed or the scene. The view shows the object layout and the arm in the ready pose.

Set a seed and a step limit, then press Start. The left column shows the front camera and the top camera live, with a grid on the top view. Under the cameras are the controls and the memo: Claude's to-do list, notes and lessons. The right column holds the chat.

The debug switch gives Claude the true object positions. Use it to find out if a failure comes from vision or from planning.

## Chat

The chat on the right shows each run as a conversation:

- **Run started:** the instruction and the system prompt, folded.
- **→ Sent:** the step message and the two images. A long message shows 4 lines and a "show all" button.
- **Claude is thinking…:** a live timer in seconds, until the reply or an error comes.
- **← Claude:** the observation, the plan, the actions, the notes, the to-do list and the lessons. The last line gives the tokens and the cost.
- **⚙ Results:** one line for each action result.
- **⚠ Stopped:** the reason when the code stops the action list early.

A retry, a bad reply, a model error and the run end each get their own entry. The chat stays at the bottom while you read the bottom. If you scroll up, it does not move, and a "↓ latest" button appears. Click a Sent or Claude entry to open the details of that step.

## Stop at the first surprise

The code runs the actions of a reply in order. It stops the list at the first surprise:

1. A `move_to` is blocked or unreachable.
2. A `close` ends on no object, with a width below 0.005 m.
3. The gripper held an object before an action and does not hold it after. An `open` is not a drop.
4. An action has an `expect` that does not match.

An action can carry `"expect": {"holding": "<object>"}` or `{"holding": "nothing"}`. The object is an object name or its label, in any case. An unknown name gives a note in the result and does not stop the list. A close with `expect` nothing does not stop. Use it to press a button with the closed gripper.

After a stop, the results list the skipped actions. The next step message says why the list stopped. The gripper holds an object only when it is closed and touches the object.

## Lessons

Claude can write up to 3 lessons in a reply. A lesson describes a cause and its effect, for example "Closing 9 cm up on the bottle slipped. 6 cm up held." Each step message lists the lessons under "Lessons:".

At the end of a run, the app saves the new lessons in `lessons/<scene>.json`, next to the `runs/` folder. Each scene keeps its newest 30 lessons. The folder is not in git.

The "Use saved lessons" switch sends the saved lessons of the scene to Claude. When it is off, Claude gets only the lessons of this run. The app still saves the new lessons. The Lessons memo column shows the lessons of this run and the number of saved lessons. The Lessons button opens a list of the saved lessons. You can delete one lesson or clear all of them.

The API has two endpoints for this. `GET /api/lessons?scene=X` lists the saved lessons. `DELETE /api/lessons?scene=X&index=N` deletes one lesson. Without `index`, it deletes all lessons of the scene. During a run, a delete gives 409.

## Scenes

A scene sets the objects on the table, the fixed items and the example tasks. The seed sets the start positions of the objects. The fixed items stay in the same place for each seed.

| Scene | What is on the table | Tasks |
| --- | --- | --- |
| cubes | A red, a blue and a green cube, each 4 cm | `reach`, `pick`, `stack` |
| tray | The three cubes and a grey tray with low walls | `red_in_tray`, `all_in_tray` |
| zones | The three cubes and three flat squares: red, blue and green | `red_on_green`, `sort` |
| shapes | A big orange cube (6 cm), a small purple cube (3 cm) and a yellow cylinder | `pick_cylinder`, `small_on_big` |
| drawer | A cabinet with a drawer that slides out toward the robot, and a red cube | `open_drawer`, `close_drawer`, `cube_in_drawer` |
| buttons | A low panel with three spring buttons: red, green and blue | `press_green`, `press_sequence` |
| kitchen | Scanned objects: a heavy bowl and a heavy plate at fixed places, a white bottle, a purple box and a yellow box | `bottle_in_bowl`, `purple_box_on_plate`, `clear_table` |
| toys | Scanned toys: an orange robot figure, a panda figure, a green robot figure, a pink rubber toy, and the grey tray | `panda_in_tray`, `all_toys_in_tray` |

The kitchen and toys scenes use these scanned objects:

| Object | Folder | Size (cm) | Mass (kg) |
| --- | --- | --- | --- |
| white bowl | `Threshold_Bead_Cereal_Bowl_White` | 16.7 × 16.5 × 6.9 | 0.8 |
| turquoise plate | `Room_Essentials_Salad_Plate_Turquoise` | 21.8 × 21.8 × 2.7 | 0.8 |
| white bottle | `Krill_Oil` | 5.4 × 5.3 × 9.7 | 0.15 |
| purple box | `JarroSil_Activated_Silicon` | 5.9 × 4.6 × 10.9 | 0.12 |
| yellow box | `Ultra_JarroDophilus` | 5.9 × 6.0 × 10.9 | 0.12 |
| orange robot figure | `Android_Figure_Orange` | 6.5 × 4.6 × 8.0 | 0.05 |
| panda figure | `Android_Figure_Panda` | 5.9 × 4.6 × 7.5 | 0.05 |
| green robot figure | `Great_Dinos_Triceratops_Toy` | 6.8 × 4.6 × 7.9 | 0.05 |
| pink rubber toy | `Kong_Puppy_Teething_Rubber_Small_Pink` | 4.3 × 4.4 × 7.1 | 0.05 |

The sizes are width (x) × depth (y) × height. The code measures them from the collision meshes. In the kitchen and toys scenes, the seed also sets a random yaw for each small item. The bowl and the plate always stand at the same place, with yaw 0. The step message gives the size of each scanned object. With the debug switch on, it also gives the yaw of each object.

Some scenes have parts that move, such as the drawer and the buttons. A brief change, such as a button press, gives an event. The next step message lists the new events, for example "Events: red button pressed". With the debug switch on, Claude also gets the positions of these parts.

The front camera sees only the back of the cabinet. So the drawer scene uses a side camera in place of the front camera. The side image has no grid.

The page keeps your choice of scene in the browser. When you change the scene, the examples, the auto-check line and the preview change too. The text box keeps your own text. It becomes empty only when it holds an example of the old scene.

Each step message starts with a line `Scene:`. This line lists the objects with their sizes and describes the fixed items. Claude gets this line also when you edit the system prompt.

Each task has a scripted solution that reads the true positions. Use it to check a task without a model:

    uv run armmind scripted --scene tray --task all_in_tray --seed 2

`--scene` is `cubes` when you do not give it. To add a scene, add a file in `src/armmind/scenes/` and add the scene to `SCENES` in `src/armmind/scenes/__init__.py`.

## See and change what Claude gets

The header shows the totals of the run. They list the steps, the input and output tokens, the cost and the average time per step.

Click a Sent or Claude entry in the chat to open the details of its step. The window shows the exact token counts from the API. It also shows an estimate of how many input tokens the two images use. The estimate is not exact.

The window shows the system prompt, the message of the step and both images. It also lists every model call of the step. A retry after a bad reply appears as a second call. Each call shows the raw reply or the error and its token counts.

Press the Prompt button to read or edit the system prompt. The tag on the button says "default" or "edited". Press Save to keep your text in the browser. Press Reset to default to go back.

Start sends the edited prompt with the run. If you did not edit it, the app uses the built-in prompt. The reply format is fixed by a JSON schema, so a prompt edit cannot break parsing.

Each run log also holds this data. The `run_start` event has the system prompt. Each step event has the message and the list of calls.

## How it works

Each step: the app renders a front and a top camera image with a grid on the table. Claude gets the images, the gripper position and the gripper yaw. It replies with up to 8 actions: `move_to [x, y, z]`, `open`, `close`, `wait`.

`move_to` moves the gripper in a straight line. It also takes an optional `yaw` in degrees, from -90 to 90. The yaw turns the gripper about the vertical axis. At yaw 0, the fingers close along x. Code finds the joint angles (IK) for each point of the path and moves the arm. If the path leaves the reachable area, the arm stops there. Each result (reached, blocked or unreachable) goes into the next prompt.

Each run writes `runs/<time>/events.jsonl` and the observation images.

## Development

    ./scripts/check.sh

To check the page without a model, start the server with `ARMMIND_FAKE_LLM=1`. A demo backend then gives invented replies. Use it only for tests.

Tests marked `assets` skip when the Panda model is missing. Tests of the kitchen and toys scenes skip when the scanned objects are missing.

## License

MIT. The Panda model is not in this repository. MuJoCo Menagerie supplies it under Apache 2.0.

## Attribution

The scanned objects are not in this repository. `scripts/fetch_objects.sh` downloads them.

- Object scans: Google Scanned Objects (Downs et al., 2022), CC-BY 4.0. See [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) and the [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). arm-mind uses the scans without changes.
- MJCF conversion: Kevin Zakka, mujoco_scanned_objects, MIT. See [mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects).

The prompt labels of the objects are our own neutral names, such as "white bottle". They do not use product names.
