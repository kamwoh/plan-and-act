"""Hard-coded plans that read the true cube positions. They prove that the arm and the tasks work."""
from __future__ import annotations

from armmind.sim.world import World
from armmind.types import Action


def scripted_plan(task: str, world: World) -> list[Action]:
    red, blue = world.cube_pos("red_cube"), world.cube_pos("blue_cube")

    def above(p, z: float) -> Action:
        return Action("move_to", (float(p[0]), float(p[1]), z))

    grasp = [Action("open"), above(red, 0.15), above(red, 0.02), Action("close")]
    plans = {
        "reach": [Action("close"), above(red, 0.15), above(red, 0.04)],
        "pick": [*grasp, above(red, 0.20)],
        "stack": [*grasp, above(red, 0.15), above(blue, 0.15), above(blue, 0.065), Action("open"),
                  above(blue, 0.15)],
    }
    return plans[task]
