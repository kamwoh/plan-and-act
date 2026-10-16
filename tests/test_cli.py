import pytest

from armmind.cli import main


def test_serve_refuses_ps1_mind_port(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--port", "8765"])
    assert exc.value.code == 2
    assert "8765" in capsys.readouterr().err


@pytest.mark.assets
def test_scripted_defaults_to_cubes(capsys):
    assert main(["scripted", "--task", "reach", "--seed", "0"]) == 0
    assert capsys.readouterr().out.strip().endswith("success")


def test_scripted_refuses_a_task_of_another_scene(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["scripted", "--scene", "cubes", "--task", "nope"])
    assert exc.value.code == 2
    assert "nope" in capsys.readouterr().err


@pytest.mark.assets
def test_scripted_uses_the_task_start_joints(capsys):
    assert main(["scripted", "--scene", "drawer", "--task", "close_drawer", "--seed", "0"]) == 0
    assert capsys.readouterr().out.strip().endswith("success")
