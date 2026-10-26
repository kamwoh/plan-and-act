"""Camera images and a coordinate grid drawn on the table."""
from __future__ import annotations

import io

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from plan_and_act.sim.world import World

GRID_X = [round(0.2 + 0.1 * i, 1) for i in range(7)]  # 0.2 .. 0.8
GRID_Y = [round(-0.4 + 0.1 * i, 1) for i in range(9)]  # -0.4 .. 0.4
LINE = (30, 30, 30)
TEXT = (0, 0, 0)


def _label(v: float) -> str:
    return f"{v + 0.0:.1f}"  # + 0.0 turns -0.0 into 0.0


class Cameras:
    """Use one instance on one thread only. The EGL context belongs to the thread that made it."""

    def __init__(self, world: World, width: int = 640, height: int = 480):
        self.world, self.w, self.h = world, width, height
        self._r = mujoco.Renderer(world.model, height, width)
        # The grid only suits the default cameras. On a scene's own camera it would cross the fixed items.
        self._no_grid = set(world.scene.camera_overrides)

    def render(self, cam: str, shadows: bool = True) -> np.ndarray:
        self._r.update_scene(self.world.data, camera=cam)
        # Set the flag on every call, because the scene keeps it. No shadows is about 5 times faster.
        self._r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows)
        return self._r.render().copy()

    def render_free(self, view: dict, shadows: bool = True) -> np.ndarray:
        """An image from a free camera: {"lookat": [x, y, z], "distance", "azimuth", "elevation"} in metres and
        degrees, as in the MuJoCo viewer. For the page only: Claude always gets the fixed cameras."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = view["lookat"]
        cam.distance, cam.azimuth, cam.elevation = view["distance"], view["azimuth"], view["elevation"]
        self._r.update_scene(self.world.data, camera=cam)
        self._r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows)
        return self._r.render().copy()

    def free_view(self, cam: str = "front") -> dict:
        """The free camera pose that shows what a fixed camera shows: it looks at the point where the camera
        axis meets the table, or 1 m ahead when the axis does not meet it."""
        m, d = self.world.model, self.world.data
        cid = m.camera(cam).id
        pos, fwd = d.cam_xpos[cid], -d.cam_xmat[cid].reshape(3, 3)[:, 2]
        dist = -pos[2] / fwd[2] if fwd[2] < -1e-3 else 1.0
        lookat = pos + fwd * dist
        return {"lookat": [round(float(v), 4) for v in lookat], "distance": round(float(dist), 4),
                "azimuth": round(float(np.degrees(np.arctan2(fwd[1], fwd[0]))), 2),
                "elevation": round(float(np.degrees(np.arcsin(np.clip(fwd[2], -1.0, 1.0)))), 2)}

    def project(self, cam: str, points) -> np.ndarray:
        m, d = self.world.model, self.world.data
        cid = m.camera(cam).id
        rot = d.cam_xmat[cid].reshape(3, 3)
        v = (np.asarray(points, dtype=float) - d.cam_xpos[cid]) @ rot  # rows are rot.T @ (p - pos)
        depth = -v[:, 2]  # a MuJoCo camera looks along its -z axis
        f = (self.h / 2) / np.tan(np.radians(m.cam_fovy[cid]) / 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.stack([self.w / 2 + f * v[:, 0] / depth, self.h / 2 - f * v[:, 1] / depth], axis=1)
        out[depth <= 1e-6] = np.nan
        return out

    def with_grid(self, cam: str, shadows: bool = True) -> Image.Image:
        img = Image.fromarray(self.render(cam, shadows))
        if cam in self._no_grid:
            return img
        draw = ImageDraw.Draw(img)
        for x in GRID_X:
            a, b = self.project(cam, [[x, GRID_Y[0], 0], [x, GRID_Y[-1], 0]])
            draw.line([tuple(a), tuple(b)], fill=LINE, width=1)
            draw.text((a[0] - 34, a[1] - 6), f"x={_label(x)}", fill=TEXT)
        for y in GRID_Y:
            a, b = self.project(cam, [[GRID_X[0], y, 0], [GRID_X[-1], y, 0]])
            draw.line([tuple(a), tuple(b)], fill=LINE, width=1)
            draw.text((a[0] - 14, a[1] - 14), f"y={_label(y)}", fill=TEXT)
        return img

    def close(self) -> None:
        self._r.close()  # without this, Python prints EGL errors at exit


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def jpeg(arr: np.ndarray, quality: int = 70) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
