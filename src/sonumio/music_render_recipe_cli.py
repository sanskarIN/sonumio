"""Inspect, render, archive and verify complete local music recipes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_project import read_project, encode_project
from .music_project_cli import _distinct, _output
from .music_performance import read_performance
from .music_rack import read_rack
from .music_render_recipe import RenderRecipe, read_recipe, encode_recipe, describe_recipe, render_recipe
from .music_render_bundle import create_bundle, verify_bundle, read_bundle
from .music_publish import publish_music_artifact
from .atomic_cli_wav import publish_generated_wav


def _pin(value, expected, label):
    if expected is not None and (value is None or value.sha256 != expected):
        raise GenerationError(label + " digest mismatch")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    create=sub.add_parser("create")
    create.add_argument("project",type=Path)
    create.add_argument("--expected-project-sha256")
    create.add_argument("--performance",type=Path)
    create.add_argument("--expected-performance-sha256")
    create.add_argument("--rack",type=Path)
    create.add_argument("--expected-rack-sha256")
    _output(create)
    for action in ("inspect","render","bundle"):
        cmd=sub.add_parser(action)
        cmd.add_argument("recipe",type=Path)
        cmd.add_argument("--expected-recipe-sha256")
        if action!="inspect":
            _output(cmd)
    for action in ("verify","extract-wav","extract-project","extract-recipe"):
        cmd=sub.add_parser(action)
        cmd.add_argument("bundle",type=Path)
        cmd.add_argument("--expected-recipe-sha256")
        cmd.add_argument("--replay",action="store_true")
        if action!="verify":
            _output(cmd)
    args=parser.parse_args(argv)
    try:
        if args.action=="create":
            _distinct((args.project,args.performance,args.rack),args.output)
            project=read_project(args.project,expected_sha256=args.expected_project_sha256)
            performance=read_performance(args.performance) if args.performance is not None else None
            rack=read_rack(args.rack) if args.rack is not None else None
            _pin(performance,args.expected_performance_sha256,"Performance")
            _pin(rack,args.expected_rack_sha256,"Rack")
            recipe=RenderRecipe(project,performance,rack)
            report=describe_recipe(recipe)
            publication=publish_music_artifact(args.output,encode_recipe(recipe),kind="render_recipe",
                overwrite=args.overwrite,expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=publication.sha256,bytes_written=publication.bytes_written)
        elif args.action in ("inspect","render","bundle"):
            recipe=read_recipe(args.recipe,expected_sha256=args.expected_recipe_sha256)
            report=describe_recipe(recipe)
            if args.action!="inspect":
                _distinct((args.recipe,),args.output)
                if args.action=="render":
                    wav,report=render_recipe(recipe)
                    publication=publish_generated_wav(args.output,wav,overwrite=args.overwrite,
                        expected_sha256=args.expected_output_sha256)
                    report={"render_receipt":report,"output_sha256":publication.wav_sha256,
                            "bytes_written":publication.bytes_written}
                else:
                    data=create_bundle(recipe)
                    _,_,_,report=verify_bundle(data)
                    publication=publish_music_artifact(args.output,data,kind="render_bundle",
                        overwrite=args.overwrite,expected_sha256=args.expected_output_sha256)
                    report.update(output_sha256=publication.sha256,bytes_written=publication.bytes_written)
        else:
            recipe,wav,_,report=verify_bundle(read_bundle(args.bundle),replay=args.replay,
                expected_recipe_sha256=args.expected_recipe_sha256)
            if args.action!="verify":
                _distinct((args.bundle,),args.output)
                if args.action=="extract-wav":
                    publication=publish_generated_wav(args.output,wav,overwrite=args.overwrite,
                        expected_sha256=args.expected_output_sha256)
                    digest=publication.wav_sha256
                else:
                    kind="project" if args.action=="extract-project" else "render_recipe"
                    data=encode_project(recipe.project) if kind=="project" else encode_recipe(recipe)
                    publication=publish_music_artifact(args.output,data,kind=kind,overwrite=args.overwrite,
                        expected_sha256=args.expected_output_sha256)
                    digest=publication.sha256
                report.update(output_sha256=digest,bytes_written=publication.bytes_written)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio render recipe: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
