import contextlib
import io
import pytest

from ndel import cli


def test_cli_help_runs() -> None:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    assert "ndel" in stdout.getvalue()
