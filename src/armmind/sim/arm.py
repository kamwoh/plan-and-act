"""Moves the gripper to targets in a straight line. IK finds joint angles for each point of the path."""
from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from armmind.sim.world import TCP_OFFSET, World
from armmind.types import Action, ActionResult, Status

WORKSPACE_LO = np.array([0.25, -0.40, 0.01])
WORKSPACE_HI = np.array([0.75, 0.40, 0.50])
READY = np.array([0.30, 0.0, 0.40])
SPEED = 0.25  # m/s along the straight distance
TURN_SPEED = 90.0  # degrees per second for a change of yaw
YAW_LIMIT = 90.0  # degrees
TICK_STEPS = 5  # physics steps per control tick: the path gets a new IK target every 10 ms
PATH_ITERS = 50  # IK iterations per path point, warm-started from the last point
SETTLE_S = 0.3
GRIP_S = 0.6
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
REACH_TOL = 0.02
MAX_WAIT_S = 3.0
FRAME_DT = 1 / 30


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
        want = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.w.rot0
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rot = d.xmat[hand].reshape(3, 3)
            p = d.xpos[hand] + rot @ TCP_OFFSET
            err_p = target - p
            err_r = 0.5 * sum(np.cross(rot[:, i], want[:, i]) for i in range(3))
            if np.linalg.norm(err_p) < 1e-3 and np.linalg.norm(err_r) < 1e-2:
                return d.qpos[:7].copy()
            mujoco.mj_jac(m, d, jp, jr, p, hand)
            jac = np.vstack([jp, jr])[:, :7]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), np.concatenate([err_p, err_r]))
            d.qpos[:7] = np.clip(d.qpos[:7] + dq, m.jnt_range[:7, 0], m.jnt_range[:7, 1])
        return None

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

    def act(self, action: Action) -> ActionResult:
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
        target = np.clip(want, WORKSPACE_LO, WORKSPACE_HI)
        notes = ["target clamped to the workspace"] if not np.allclose(target, want) else []
        yaw = self._yaw if action.yaw is None else float(np.clip(action.yaw, -YAW_LIMIT, YAW_LIMIT))
        if action.yaw is not None and yaw != action.yaw:
            notes.append(f"yaw clamped to {yaw:g}")
        note = ", ".join(notes)
        if self.solve_ik(target, yaw) is None:
            why = "no joint solution for this point"
            return self._result(action, "unreachable", f"{note}, and {why}" if note else why)
        start, yaw0, q = self.w.tcp(), self._yaw, self.w.data.ctrl[:7].copy()
        duration = max(0.4, float(np.linalg.norm(target - start)) / SPEED, abs(yaw - yaw0) / TURN_SPEED)
        ticks = max(1, round(duration / (self.w.dt * TICK_STEPS)))
        for i in range(1, ticks + 1):
            a = i / ticks
            a = a * a * (3 - 2 * a)  # smoothstep: start and stop gently
            q_next = self.solve_ik(start + a * (target - start), yaw0 + a * (yaw - yaw0), q, PATH_ITERS)
            if q_next is None:  # hold the last good point
                self._yaw = yaw0 + a * (yaw - yaw0)
                self._run(SETTLE_S)
                return self._result(action, "unreachable", "path leaves the reachable area")
            q = q_next
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
