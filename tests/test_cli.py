from __future__ import annotations

import importlib.metadata
import subprocess
from shutil import which

import pytest

ENTRY_POINTS = {
    "stim": "stim-ai",
    "stim-serve": "stim-serve",
    "stim-systemone": "stim-systemone",
}


def run_console_script(command: str, *args: str) -> subprocess.CompletedProcess[str]:
    executable = which(command)
    assert executable is not None, f"{command} is not installed"
    return subprocess.run(
        [executable, *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("command", ENTRY_POINTS)
def test_console_script_has_help(command: str) -> None:
    result = run_console_script(command, "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert command in result.stdout


@pytest.mark.parametrize(("command", "distribution"), ENTRY_POINTS.items())
def test_console_script_reports_version(command: str, distribution: str) -> None:
    result = run_console_script(command, "--version")
    assert result.returncode == 0
    assert result.stdout.strip() == f"{command} {importlib.metadata.version(distribution)}"
