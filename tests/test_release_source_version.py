"""Keep every software-version surface synchronized without publishing."""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path
import re
import tomllib

import pytest

import sonumio
from sonumio import api
from sonumio.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def expected_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    assert project["name"] == "sonumio"
    assert project["license"] == "Apache-2.0"
    candidate = project["version"]
    assert isinstance(candidate, str)
    assert re.fullmatch(r"(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})", candidate)
    return candidate


def test_python_package_installed_version_matches_source(expected_version):
    assert version("sonumio") == sonumio.__version__ == expected_version


def test_local_api_reports_matching_software_version(expected_version):
    assert api.app.version == sonumio.__version__ == expected_version
    assert api.app.title == "Sonumio experimental API"


def test_console_script_exposes_source_version(capsys, expected_version):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"sonumio {expected_version}"


def test_private_web_package_matches_source_version(expected_version):
    package = json.loads((ROOT / "web" / "package.json").read_text("utf-8"))
    assert package["name"] == "@sonumio/web"
    assert package["version"] == expected_version
    assert package["private"] is True
