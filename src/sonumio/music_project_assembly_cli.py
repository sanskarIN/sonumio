"""Assemble native projects and their compatible automation locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_assembly import read_assembly
from .music_assembly_cli import _bindings, _distinct
from .music_project import read_project, encode_project
from .music_project_assembly import (
    ProjectAssemblyPlan, ProjectPin, read_project_assembly,
    encode_project_assembly, assemble_projects,
)
from .music_publish import publish_music_artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("pin", "inspect", "assemble"):
        command = sub.add_parser(action)
        command.add_argument("recipe", type=Path)
        command.add_argument("--source", action="append", required=True, help="Local alias=project.sonumio")
        command.add_argument("--expected-recipe-sha256")
        if action == "pin":
            command.add_argument("--title", required=True)
        if action != "inspect":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--expected-output-sha256")
    args = parser.parse_args(argv)
    try:
        paths = _bindings(args.source)
        if args.action != "inspect":
            _distinct((args.recipe, *paths.values()), args.output)
        if args.action == "pin":
            score_recipe = read_assembly(args.recipe, expected_sha256=args.expected_recipe_sha256)
            if set(paths) != {pin.alias for pin in score_recipe.sources}:
                raise GenerationError("Project source bindings differ from score recipe")
            projects = {alias: read_project(path) for alias, path in paths.items()}
            plan = ProjectAssemblyPlan(args.title, score_recipe,
                tuple(ProjectPin(alias, projects[alias].sha256) for alias in sorted(projects)))
        else:
            plan = read_project_assembly(args.recipe, expected_sha256=args.expected_recipe_sha256)
            pins = {pin.alias: pin.project_sha256 for pin in plan.projects}
            if set(paths) != set(pins):
                raise GenerationError("Project source bindings differ from recipe")
            projects = {alias: read_project(path, expected_sha256=pins[alias])
                        for alias, path in paths.items()}
        project, report = assemble_projects(plan, projects)
        if args.action != "inspect":
            data = encode_project_assembly(plan) if args.action == "pin" else encode_project(project)
            receipt = publish_music_artifact(args.output, data,
                kind="score" if args.action == "pin" else "project",
                overwrite=args.overwrite, expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=receipt.sha256, bytes_written=receipt.bytes_written,
                          output_kind="project_assembly_recipe" if args.action=="pin" else "project")
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio project assembly: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
