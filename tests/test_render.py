import numpy as np
import pytest

from armmind.sim.render import GRID_X, GRID_Y, Cameras, jpeg, png
from armmind.sim.world import World

LAYOUT = {"red_cube": (0.62, -0.25), "blue_cube": (0.45, 0.15), "green_cube": (0.62, 0.05)}

pytestmark = pytest.mark.assets


@pytest.fixture
def cams():
    w = World()
    w.reset(LAYOUT)
    c = Cameras(w)
    yield c
    c.close()


@pytest.mark.parametrize("cam", ["front", "top"])
def test_render_shape(cams, cam):
    img = cams.render(cam)
    assert img.shape == (480, 640, 3)
    assert img.dtype == np.uint8


@pytest.mark.parametrize("cam", ["front", "top"])
def test_projection_hits_red_cube(cams, cam):
    top_face = cams.world.cube_pos("red_cube") + [0, 0, 0.02]
    (u, v), = cams.project(cam, [top_face])
    r, g, b = cams.render(cam)[int(v), int(u)].astype(int)
    assert r > 150 and r > g + 80 and r > b + 80


@pytest.mark.parametrize("cam", ["front", "top"])
def test_grid_corners_inside_image(cams, cam):
    corners = [[x, y, 0] for x in (GRID_X[0], GRID_X[-1]) for y in (GRID_Y[0], GRID_Y[-1])]
    px = cams.project(cam, corners)
    assert np.all(px[:, 0] >= 10) and np.all(px[:, 0] <= 630)
    assert np.all(px[:, 1] >= 10) and np.all(px[:, 1] <= 470)


def test_projection_behind_camera_is_nan(cams):
    px = cams.project("top", [[0.5, 0.0, 5.0]])
    assert np.isnan(px).all()


def test_grid_image_and_encoders(cams):
    img = cams.with_grid("top")
    assert img.size == (640, 480)
    assert png(img)[:8] == b"\x89PNG\r\n\x1a\n"
    assert jpeg(cams.render("front"))[:2] == b"\xff\xd8"


@pytest.mark.parametrize("cam", ["front", "top"])
def test_render_without_shadows(cams, cam):
    plain, lit = cams.render(cam, shadows=False), cams.render(cam)
    assert plain.shape == lit.shape == (480, 640, 3)
    assert (plain != lit).any()
    assert (cams.render(cam) == lit).all()  # the flag does not stick


def test_with_grid_passes_shadows_through(cams):
    assert np.any(np.asarray(cams.with_grid("top", shadows=False)) != np.asarray(cams.with_grid("top")))
