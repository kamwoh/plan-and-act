"""The full simulation state at the end of an execution, in a file. A new env can start from it."""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import mujoco
import numpy as np

from plan_and_act.scenes.base import Scene

END_STATE = "end_state.npz"  # in the run folder: the state at the end of the last execution of the plan
START_STATE = "start_state.npz"  # in the run folder of a continue round: the state at its start
ARRAYS = ("qpos", "qvel", "act", "ctrl", "qacc_warmstart")

_VERSIONS: dict[str, str] = {}


class StateMismatch(ValueError):
    """The state file is missing, broken, or from another scene, seed or scene version."""


def scene_version(scene: Scene) -> str:
    """A short hash of the MuJoCo XML of the scene. A change to the scene model changes it."""
    if scene.name not in _VERSIONS:
        from plan_and_act.sim.world import scene_xml
        _VERSIONS[scene.name] = hashlib.sha256(scene_xml(scene).encode()).hexdigest()[:16]
    return _VERSIONS[scene.name]


def save(path: Path, scene: Scene, seed: int, data: mujoco.MjData, yaw: float, events: list[str]) -> None:
    """Write the state of `data` and the commanded yaw of the arm. The file is whole at all times."""
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez(tmp, scene=np.array(scene.name), seed=np.array(int(seed)), version=np.array(scene_version(scene)),
             time=np.array(float(data.time)), yaw=np.array(float(yaw)), events=np.array(list(events), dtype=str),
             **{k: np.array(getattr(data, k)) for k in ARRAYS})
    tmp.replace(path)


def load(path: Path, scene: Scene, seed: int | None = None) -> dict:
    """The arrays and values of a state file. Raises StateMismatch when it does not fit the scene and seed."""
    try:
        with np.load(Path(path), allow_pickle=False) as f:
            out = {k: f[k] for k in f.files}
    except FileNotFoundError as e:
        raise StateMismatch("no end state of the last execution") from e
    except (OSError, ValueError, zipfile.BadZipFile) as e:
        raise StateMismatch(f"cannot read the end state: {type(e).__name__}") from e
    missing = {"scene", "seed", "version", "time", "yaw", "events", *ARRAYS} - set(out)
    if missing:
        raise StateMismatch(f"the end state misses {sorted(missing)}")
    if str(out["scene"]) != scene.name:
        raise StateMismatch(f"the end state is of the scene {str(out['scene'])!r}, not {scene.name!r}")
    if seed is not None and int(out["seed"]) != seed:
        raise StateMismatch(f"the end state is of seed {int(out['seed'])}, not seed {seed}")
    if str(out["version"]) != scene_version(scene):
        raise StateMismatch("the end state is of an older version of the scene")
    return out


def state_problem(path: Path, scene: Scene, seed: int) -> str | None:
    """Why the state file cannot start a round of this scene and seed, or None when it can."""
    try:
        load(path, scene, seed)
    except StateMismatch as e:
        return str(e)
    return None


def apply(state: dict, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the saved state into `data`. Nothing settles and nothing moves."""
    for k in ARRAYS:
        target = getattr(data, k)
        if target.shape != state[k].shape:
            raise StateMismatch(f"the end state has {k} of shape {state[k].shape}, not {target.shape}")
        target[:] = state[k]
    data.time = float(state["time"])
    mujoco.mj_forward(model, data)
