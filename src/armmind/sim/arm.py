"""Moves the gripper to targets in a straight line. IK finds joint angles for each point of the path."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import mujoco
import numpy as np

from armmind.sim.world import TCP_OFFSET, World
from armmind.types import CLOSED_ON_NOTHING_M, Action, ActionResult, Status

WORKSPACE_LO = np.array([0.25, -0.40, 0.01])
WORKSPACE_HI = np.array([0.75, 0.40, 0.50])
READY = np.array([0.30, 0.0, 0.40])
SPEED = 0.25  # m/s along the straight distance
TURN_SPEED = 90.0  # degrees per second for a change of yaw
YAW_LIMIT = 90.0  # degrees
TICK_STEPS = 5  # physics steps per control tick: the path gets a new IK target every 10 ms
PATH_ITERS = 50  # IK iterations per path point, warm-started from the last point
CHECK_YAW_STEP = 30.0  # degrees: the reach check turns the yaw in steps this size or smaller
SETTLE_S = 0.3
GRIP_S = 0.6
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
REACH_TOL = 0.02
MAX_WAIT_S = 3.0
FRAME_DT = 1 / 30


def clamp_target(xyz) -> np.ndarray:
    """The point that move_to goes to: the target, moved into the workspace box."""
    return np.clip(np.asarray(xyz, dtype=float), WORKSPACE_LO, WORKSPACE_HI)


def move_duration(start, target, yaw0: float, yaw: float) -> float:
    """The time of the motion of a move_to in seconds, without the settle time after it."""
    dist = float(np.linalg.norm(np.asarray(target, dtype=float) - np.asarray(start, dtype=float)))
    return max(0.4, dist / SPEED, abs(yaw - yaw0) / TURN_SPEED)


class Arm:
    def __init__(self, world: World, on_frame: Callable[[], None] | None = None):
        self.w, self.on_frame = world, on_frame
        self._ik = mujoco.MjData(world.model)  # scratch data: IK never moves the real arm
        self._yaw = 0.0  # the commanded yaw in degrees
        self._next_frame = 0.0

    def solve_ik(self, target, yaw: float = 0.0, start=None, iters: int = 300) -> np.ndarray | None:
        """Joint angles that put the TCP at target with this yaw. Starts from `start`, or else the current pose."""
        m, d, hand = self.w.model, self._ik, self.w.hand_id
        d.qpos[:] = self.w.data.qpos
        if start is not None:
            d.qpos[:7] = start
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        want = np.empty(4)
        mujoco.mju_mat2Quat(want, (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.w.rot0).ravel())
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        here, turn = np.empty(4), np.empty(3)
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rot = d.xmat[hand].reshape(3, 3)
            p = d.xpos[hand] + rot @ TCP_OFFSET
            err_p = target - p
            # The axis-angle turn from here to the wanted orientation. Unlike a sum of cross products,
            # it does not shrink to zero near a half turn.
            mujoco.mju_mat2Quat(here, rot.ravel())
            mujoco.mju_subQuat(turn, want, here)  # in the hand frame
            err_r = rot @ turn
            if np.linalg.norm(err_p) < 1e-3 and np.linalg.norm(err_r) < 1e-2:
                return d.qpos[:7].copy()
            mujoco.mj_jac(m, d, jp, jr, p, hand)
            jac = np.vstack([jp, jr])[:, :7]
            err = np.concatenate([err_p, err_r])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            q, lo, hi = d.qpos[:7], m.jnt_range[:7, 0], m.jnt_range[:7, 1]
            stuck = ((q <= lo) & (dq < 0)) | ((q >= hi) & (dq > 0))
            if stuck.any():  # a joint at its limit cannot help: solve again with the other joints
                jac[:, stuck] = 0.0
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            d.qpos[:7] = np.clip(q + dq, lo, hi)
        return None

    def reachable(self, target, yaw: float) -> bool:
        """True when IK finds the target with this yaw, from the current pose or from home. Nothing moves."""
        return self._reachable(np.asarray(target, dtype=float), yaw)

    def _reachable(self, target, yaw: float) -> bool:
        """True when IK finds the target with this yaw, from the current pose or else from home."""
        return (self._walk(target, self._yaw, yaw, None) is not None
                or self._walk(target, 0.0, yaw, self.w.model.key_qpos[0][:7]) is not None)

    def _walk(self, target, yaw_from: float, yaw_to: float, start) -> np.ndarray | None:
        """Go to the target at yaw_from first, then turn to yaw_to in small steps, each one warm-started
        from the last. One big turn can fail where small ones work."""
        steps = max(1, int(np.ceil(abs(yaw_to - yaw_from) / CHECK_YAW_STEP)))
        q = start
        for k in range(steps + 1):
            q = self.solve_ik(target, yaw_from + k / steps * (yaw_to - yaw_from), q)
            if q is None:
                return None
        return q

    def reset_ready(self) -> None:
        q = self.solve_ik(READY)
        if q is None:
            raise RuntimeError("no IK solution for the ready pose")
        d = self.w.data
        d.qpos[:7] = q
        d.ctrl[:7] = q
        d.ctrl[7] = GRIP_OPEN
        self._yaw = 0.0
        mujoco.mj_forward(self.w.model, d)
        self._next_frame = self.w.time
        self._run(SETTLE_S)

    @property
    def yaw(self) -> float:
        """The commanded yaw in degrees."""
        return self._yaw

    def resume(self, yaw: float) -> None:
        """Go on from a loaded state: the commanded yaw of that state, and the frame clock from now."""
        self._yaw = yaw
        self._next_frame = self.w.time

    def held(self) -> tuple[str, ...]:
        """The scene objects that the gripper holds now: it is closed, the width is not about 0, and a part of
        the gripper touches the object. An open gripper, or a push with the closed empty fingers, holds nothing.
        A mesh object often touches only one finger and the palm, so one finger is enough."""
        if self.w.data.ctrl[7] != GRIP_CLOSED or self.w.gripper_width() < CLOSED_ON_NOTHING_M:
            return ()
        names = {o.name for o in self.w.objects}
        return tuple(n for n in self.w.gripper_contacts() if n in names)

    def act(self, action: Action) -> ActionResult:
        before = self.held()
        return replace(self._act(action), held_before=before, held_after=self.held())

    def _act(self, action: Action) -> ActionResult:
        if action.type == "move_to":
            return self._move_to(action)
        if action.type in ("open", "close"):
            self.w.data.ctrl[7] = GRIP_CLOSED if action.type == "close" else GRIP_OPEN
            self._run(GRIP_S)
            return self._result(action, "reached")
        if action.type == "wait":
            self._run(min(max(action.seconds, 0.0), MAX_WAIT_S))
            return self._result(action, "reached")
        raise ValueError(f"unknown action type {action.type!r}")

    def _move_to(self, action: Action) -> ActionResult:
        want = np.asarray(action.xyz, dtype=float)
        target = clamp_target(want)
        notes = ["target clamped to the workspace"] if not np.allclose(target, want) else []
        yaw = self._yaw if action.yaw is None else float(np.clip(action.yaw, -YAW_LIMIT, YAW_LIMIT))
        if action.yaw is not None and yaw != action.yaw:
            notes.append(f"yaw clamped to {yaw:g}")
        note = ", ".join(notes)
        if not self._reachable(target, yaw):
            why = "no joint solution for this point"
            return self._result(action, "unreachable", f"{note}, and {why}" if note else why)
        start, yaw0, q = self.w.tcp(), self._yaw, self.w.data.ctrl[:7].copy()
        duration = move_duration(start, target, yaw0, yaw)
        ticks = max(1, round(duration / (self.w.dt * TICK_STEPS)))
        last_yaw = yaw0
        for i in range(1, ticks + 1):
            a = i / ticks
            a = a * a * (3 - 2 * a)  # smoothstep: start and stop gently
            q_next = self.solve_ik(start + a * (target - start), yaw0 + a * (yaw - yaw0), q, PATH_ITERS)
            if q_next is None:  # hold the last good point
                self._yaw = last_yaw
                self._run(SETTLE_S)
                return self._result(action, "unreachable", ", ".join([*notes, "path leaves the reachable area"]))
            q, last_yaw = q_next, yaw0 + a * (yaw - yaw0)
            self.w.data.ctrl[:7] = q
            for _ in range(TICK_STEPS):
                self._step()
        self._yaw = yaw
        self._run(SETTLE_S)
        reached = np.linalg.norm(self.w.tcp() - target) <= REACH_TOL
        return self._result(action, "reached" if reached else "blocked", note)

    def _result(self, action: Action, status: Status, note: str = "") -> ActionResult:
        p = self.w.tcp()
        return ActionResult(action, status, (round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)),
                            round(self.w.gripper_width(), 3), tuple(self.w.gripper_contacts()), note,
                            round(self.w.gripper_yaw(), 1))

    def _step(self) -> None:
        self.w.step()
        if self.on_frame is not None and self.w.time >= self._next_frame:
            self._next_frame = self.w.time + FRAME_DT
            self.on_frame()

    def _run(self, seconds: float) -> None:
        for _ in range(round(seconds / self.w.dt)):
            self._step()
