"""Moves the gripper to targets in a straight line. IK finds joint angles for each point of the path."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import mujoco
import numpy as np

from plan_and_act.sim.world import World
from plan_and_act.types import CLOSED_ON_NOTHING_M, Action, ActionResult, Status

WORKSPACE_LO = np.array([0.25, -0.40, 0.01])
WORKSPACE_HI = np.array([0.75, 0.40, 0.50])
READY = np.array([0.30, 0.0, 0.40])  # the start point of the Panda; Robot.ready of each arm
SPEED = 0.25  # m/s along the straight distance
TURN_SPEED = 90.0  # degrees per second for a change of yaw
YAW_LIMIT = 90.0  # degrees
TICK_STEPS = 5  # physics steps per control tick: the path gets a new IK target every 10 ms
PATH_ITERS = 50  # IK iterations per path point, warm-started from the last point
CHECK_YAW_STEP = 30.0  # degrees: the reach check turns the yaw in steps this size or smaller
SETTLE_S = 0.3
GRIP_S = 0.6
REACH_TOL = 0.02
MAX_WAIT_S = 3.0
FRAME_DT = 1 / 30
READY_TRIES = 300  # random IK starts for the ready pose
READY_KEEP = 12  # the ready pose solutions to compare
DOWN = np.array([0.0, 0.0, -1.0])
CLEARANCE = 0.03  # m: the arm links of a ready pose stay this high above the table, when a solution can
PROBE_DX, PROBE_Z = 0.15, 0.03  # the probe point: this far out from the ready point, this high above the table
_READY_Q: dict[str, np.ndarray] = {}  # robot name -> the arm joints at the ready pose


def clamp_target(xyz) -> np.ndarray:
    """The point that move_to goes to: the target, moved into the workspace box."""
    return np.clip(np.asarray(xyz, dtype=float), WORKSPACE_LO, WORKSPACE_HI)


def _below(model: mujoco.MjModel, body: int, top: int) -> bool:
    """True when body is top or a body below it."""
    while body > 0 and body != top:
        body = model.body_parentid[body]
    return body == top


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
        self._start = self._ready_q()  # the IK start when the start from the current pose fails

    def solve_ik(self, target, yaw: float = 0.0, start=None, iters: int = 300) -> np.ndarray | None:
        """Joint angles that put the TCP at target with this yaw. Starts from `start`, or else the current pose.
        The gripper points straight down. An arm with a max tilt tilts the gripper when it cannot point down."""
        q = self._ik_pass(target, yaw, start, iters, tilt=False)
        if q is None and self.w.robot.max_tilt > 0:
            q = self._ik_pass(target, yaw, start, iters, tilt=True)
        return q

    def _ik_pass(self, target, yaw: float, start, iters: int, tilt: bool) -> np.ndarray | None:
        """Damped least squares. With tilt, the position and the yaw come first, and the solution can tilt
        up to the max tilt of the robot."""
        m, d, site, p = self.w.model, self._ik, self.w.tcp_id, self.w.parts
        d.qpos[:] = self.w.data.qpos
        if start is not None:
            d.qpos[p.arm_qpos] = start
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        want = np.empty(4)
        mujoco.mju_mat2Quat(want, (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.w.rot0).ravel())
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        here, turn = np.empty(4), np.empty(3)
        max_tilt = np.radians(self.w.robot.max_tilt)
        close_dir = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.w.rot0[:, 1]  # wanted closing axis
        for _ in range(iters):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            rot = d.site_xmat[site].reshape(3, 3)
            err_p = target - d.site_xpos[site]
            # The axis-angle turn from here to the wanted orientation. Unlike a sum of cross products,
            # it does not shrink to zero near a half turn.
            if not tilt:
                mujoco.mju_mat2Quat(here, rot.ravel())
                mujoco.mju_subQuat(turn, want, here)  # in the tcp frame
                err_r = rot @ turn
                done = np.linalg.norm(err_r) < 1e-2
            else:
                # Two errors that do not mix: the yaw of the closing axis seen from above, about world z, and
                # the tilt, the turn that points the gripper axis straight down.
                yh, wh = rot[:2, 1], close_dir[:2]
                yaw_err = float(np.arctan2(yh[0] * wh[1] - yh[1] * wh[0], yh @ wh))
                turn_down = np.cross(rot[:, 2], DOWN)
                err_r = np.array([turn_down[0], turn_down[1], yaw_err])
                done = abs(yaw_err) < 1e-2 and np.arccos(np.clip(-rot[2, 2], -1.0, 1.0)) <= max_tilt
            if np.linalg.norm(err_p) < 1e-3 and done:
                return d.qpos[p.arm_qpos].copy()
            mujoco.mj_jacSite(m, d, jp, jr, site)
            jac = np.vstack([jp, jr])[:, p.arm_dof]
            err = np.concatenate([err_p, err_r])
            q, lo, hi = d.qpos[p.arm_qpos], p.lo, p.hi
            dq = self._step_dq(jac, err, tilt)
            stuck = ((q <= lo) & (dq < 0)) | ((q >= hi) & (dq > 0))
            if stuck.any():  # a joint at its limit cannot help: solve again with the other joints
                jac[:, stuck] = 0.0
                dq = self._step_dq(jac, err, tilt)
            d.qpos[p.arm_qpos] = np.clip(q + dq, lo, hi)
        return None

    @staticmethod
    def _step_dq(jac: np.ndarray, err: np.ndarray, tilt: bool) -> np.ndarray:
        """One damped least squares step. With tilt, the position and the yaw (rows 0, 1, 2 and 5) come first,
        and the tilt (rows 3 and 4) moves only the joints that they leave free."""
        if not tilt:
            return jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
        main, side = jac[[0, 1, 2, 5]], jac[[3, 4]]
        pinv = main.T @ np.linalg.inv(main @ main.T + 1e-4 * np.eye(4))
        dq = pinv @ err[[0, 1, 2, 5]]
        free = np.eye(jac.shape[1]) - pinv @ main
        ns = side @ free
        return dq + free @ ns.T @ np.linalg.solve(ns @ ns.T + 1e-3 * np.eye(2), err[[3, 4]] - side @ dq)

    def reachable(self, target, yaw: float) -> bool:
        """True when IK finds the target with this yaw, from the current pose or from home. Nothing moves."""
        return self._reachable(np.asarray(target, dtype=float), yaw)

    def _reachable(self, target, yaw: float) -> bool:
        """True when IK finds the target with this yaw, from the current pose or else from home."""
        return (self._walk(target, self._yaw, yaw, None) is not None
                or self._walk(target, 0.0, yaw, self._start) is not None)

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

    def _ready_q(self) -> np.ndarray:
        """The IK start when the start from the current pose fails. For the Panda: its home. For another arm:
        the joints of the ready pose that are farthest from a singular pose, from home and random starts. The
        home of many arms is folded up, far from a gripper that points down. Found once for each robot, with a
        fixed seed, so the same robot always gets the same joints."""
        name, p, robot = self.w.robot.name, self.w.parts, self.w.robot
        if name not in _READY_Q:
            ready = np.asarray(robot.ready)
            if robot.ik_from_home:
                if self.solve_ik(ready, start=p.home) is None:
                    raise RuntimeError(f"no IK solution for the ready pose of the {robot.title}")
                _READY_Q[name] = p.home
                return p.home
            rng = np.random.default_rng(0)
            lo, hi = np.maximum(p.lo, -np.pi), np.minimum(p.hi, np.pi)
            found = [q for start in (p.home, *(rng.uniform(lo, hi) for _ in range(READY_TRIES)))
                     if (q := self.solve_ik(ready, start=start)) is not None][:READY_KEEP]
            if not found:
                raise RuntimeError(f"no IK solution for the ready pose of the {robot.title}")
            # The arm must reach down to the table from the ready pose, with no link on the table.
            probe = np.array([ready[0] + PROBE_DX, ready[1], PROBE_Z])
            score = {i: self._reach_down(q, probe) for i, q in enumerate(found)}
            clear = [i for i in score if score[i] >= CLEARANCE]
            best = max(clear, key=lambda i: self._manipulability(found[i])) if clear else max(score, key=score.get)
            _READY_Q[name] = found[best]
        return _READY_Q[name]

    def _reach_down(self, q: np.ndarray, probe: np.ndarray) -> float:
        """The clearance of the arm links at q and at the joints that reach the probe from q, the lower one.
        -inf when IK does not reach the probe from q."""
        down = self.solve_ik(probe, start=q)
        return -np.inf if down is None else min(self._clearance(q), self._clearance(down))

    def _clearance(self, q: np.ndarray) -> float:
        """The height above the table of the lowest arm link at q, from the shoulder on and without the
        gripper. An elbow that points down hits the table when the gripper goes down."""
        m, d, p = self.w.model, self._ik, self.w.parts
        d.qpos[:] = self.w.data.qpos
        d.qpos[p.arm_qpos] = q
        mujoco.mj_kinematics(m, d)
        shoulder = m.jnt_bodyid[next(j for j in range(m.njnt) if m.jnt_qposadr[j] == p.arm_qpos[1])]
        low = np.inf
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if b in p.gripper_bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]) or not _below(m, b, shoulder):
                continue
            centre, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
            rot = d.geom_xmat[g].reshape(3, 3)
            low = min(low, float((d.geom_xpos[g] + rot @ centre)[2] - np.abs(rot[2]) @ half))
        return low

    def _manipulability(self, q: np.ndarray) -> float:
        """How far the arm is from a singular pose at q: the volume of its tcp velocities for unit joint speeds."""
        m, d, p = self.w.model, self._ik, self.w.parts
        d.qpos[:] = self.w.data.qpos
        d.qpos[p.arm_qpos] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, self.w.tcp_id)
        jac = np.vstack([jp, jr])[:, p.arm_dof]
        return float(np.sqrt(max(np.linalg.det(jac @ jac.T), 0.0)))

    def reset_ready(self) -> None:
        p = self.w.parts
        q = self.solve_ik(np.asarray(self.w.robot.ready), start=self._start)
        if q is None:
            raise RuntimeError(f"no IK solution for the ready pose of the {self.w.robot.title}")
        d = self.w.data
        d.qpos[p.arm_qpos] = q
        d.ctrl[p.arm_ctrl] = q
        d.ctrl[p.grip_ctrl] = p.grip_open
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
        if not self.w.gripper_closed() or self.w.gripper_width() < CLOSED_ON_NOTHING_M:
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
            p = self.w.parts
            self.w.data.ctrl[p.grip_ctrl] = p.grip_closed if action.type == "close" else p.grip_open
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
        start, yaw0, q = self.w.tcp(), self._yaw, self.w.data.ctrl[self.w.parts.arm_ctrl].copy()
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
            self.w.data.ctrl[self.w.parts.arm_ctrl] = q
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
