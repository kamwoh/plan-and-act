import itertools

import numpy as np
import pytest

from armmind.policy.scripted import scripted_plan
from armmind.sim.arm import Arm
from armmind.sim.world import CUBES, World
from armmind.tasks import TASKS, random_layout


def test_layout_is_repeatable_and_spread_out():
    for seed in range(20):
        a, b = random_layout(seed), random_layout(seed)
        assert a == b
        assert set(a) == set(CUBES)
        for p, q in itertools.combinations(a.values(), 2):
            assert np.hypot(p[0] - q[0], p[1] - q[1]) >= 0.10
        for x, y in a.values():
            assert 0.40 <= x <= 0.65 and -0.25 <= y <= 0.25
    assert random_layout(0) != random_layout(1)


@pytest.mark.assets
@pytest.mark.parametrize("name", sorted(TASKS))
def test_no_task_is_solved_at_start(name):
    w = World()
    w.reset(TASKS[name].layout(0))
    Arm(w).reset_ready()
    assert not TASKS[name].success(w)


@pytest.mark.assets
@pytest.mark.parametrize("name,seed", [(n, s) for n in sorted(TASKS) for s in range(3)])
def test_scripted_plan_solves_task(name, seed):
    task = TASKS[name]
    w = World()
    w.reset(task.layout(seed))
    arm = Arm(w)
    arm.reset_ready()
    for action in scripted_plan(name, w):
        arm.act(action)
    assert task.success(w)
