import pytest

from plan_and_act.cli import main


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


def test_serve_uses_the_demo_backend_only_with_the_env_var(monkeypatch, tmp_path):
    import uvicorn

    from plan_and_act import cli
    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: apps.append(app))
    monkeypatch.setenv("PLAN_AND_ACT_FAKE_LLM", "1")
    assert cli.main(["serve", "--port", "8788", "--runs", str(tmp_path / "runs")]) == 0
    monkeypatch.delenv("PLAN_AND_ACT_FAKE_LLM")
    assert cli.main(["serve", "--port", "8788", "--runs", str(tmp_path / "runs")]) == 0
    assert apps[0].state.runner.backend.name == "demo"
    assert apps[1].state.runner.backend.name != "demo"


def test_demo_mode_keeps_its_lessons_out_of_the_real_folder(monkeypatch, tmp_path):
    import tempfile

    import uvicorn

    from plan_and_act import cli
    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, host, port: apps.append(app))
    monkeypatch.setenv("PLAN_AND_ACT_FAKE_LLM", "1")
    assert cli.main(["serve", "--port", "8788", "--runs", str(tmp_path / "runs")]) == 0
    root = apps[0].state.runner.lessons.root
    assert root != tmp_path / "lessons" and str(root).startswith(tempfile.gettempdir())
