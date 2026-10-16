import pytest

from armmind.cli import main


def test_serve_refuses_ps1_mind_port(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["serve", "--port", "8765"])
    assert exc.value.code == 2
    assert "8765" in capsys.readouterr().err
