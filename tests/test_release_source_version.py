"""Keep software pre-release metadata consistent without publishing anything."""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import tomllib

import pytest

import sonumio
from sonumio import api
from sonumio.cli import main


def test_python_package_installed_version_matches_source():
    project_root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((project_root / "pyproject.toml").read_text("utf-8"))
    assert data["project"]["name"] == "sonumio"
    assert data["project"]["version"] == "0.0.1"
    assert data["project"]["license"] == "Apache-2.0"
    assert version("sonumio") == sonumio.__version__ == "0.0.1"


def test_local_api_reports_matching_software_version():
    assert api.app.version == sonumio.__version__
    assert api.app.title == "Sonumio experimental API"


def test_console_script_exposes_first_source_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "sonumio 0.0.1"
