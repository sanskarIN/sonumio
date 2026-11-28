"""Standalone local CLI for music project documents and verified revisions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .errors import GenerationError
from .music_automation import read_automation, encode_automation
from .music_score_io import read_score, encode_score
from .music_project import (
    create_project, revise_project, describe_project, verify_revision,
    encode_project, read_project, render_project,
)
from .music_publish import publish_music_artifact
from .atomic_cli_wav import publish_generated_wav


def _project(parser,name="project"):
    parser.add_argument(name,type=Path)
    parser.add_argument("--expected-"+name+"-sha256")


def _score_options(parser,*,required):
    parser.add_argument("--score",type=Path,required=required)
    parser.add_argument("--expected-score-sha256")
    parser.add_argument("--automation",type=Path)
    parser.add_argument("--expected-automation-sha256")


def _output(parser):
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")


def _distinct(inputs,output):
    for path in inputs:
        if path is None:
            continue
        if path.resolve()==output.resolve() or (output.exists() and os.path.samefile(path,output)):
            raise GenerationError("Music project output cannot overwrite an input")


def _load_score(args):
    if args.score is None:
        if args.expected_score_sha256 is not None:
            raise GenerationError("A score pin requires a score input")
        return None
    return read_score(args.score,expected_sha256=args.expected_score_sha256)


def _load_automation(args):
    if args.automation is None:
        if args.expected_automation_sha256 is not None:
            raise GenerationError("An automation pin requires an automation input")
        return None
    plan=read_automation(args.automation)
    if args.expected_automation_sha256 is not None and plan.sha256!=args.expected_automation_sha256:
        raise GenerationError("Project automation input digest mismatch")
    return plan


def _publish(args,project,report):
    receipt=publish_music_artifact(args.output,encode_project(project),kind="project",
        overwrite=args.overwrite,expected_sha256=args.expected_output_sha256)
    report.update({"output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written})
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    create=sub.add_parser("create",help="Embed a verified score and automation plan")
    create.add_argument("--title",required=True)
    _score_options(create,required=True)
    _output(create)
    inspect=sub.add_parser("inspect",help="Verify and describe a project")
    _project(inspect)
    revise=sub.add_parser("revise",help="Create a content-linked project revision")
    _project(revise)
    revise.add_argument("--title")
    _score_options(revise,required=False)
    _output(revise)
    parent=sub.add_parser("verify-parent",help="Verify a revision against its supplied parent")
    _project(parent,"parent")
    _project(parent,"child")
    render=sub.add_parser("render",help="Render the embedded score and automation")
    _project(render)
    _output(render)
    for name in ("extract-score","extract-automation"):
        extract=sub.add_parser(name,help="Extract one canonical embedded document")
        _project(extract)
        _output(extract)
    args=parser.parse_args(argv)
    try:
        if args.command=="create":
            _distinct((args.score,args.automation),args.output)
            project=create_project(args.title,_load_score(args),_load_automation(args))
            report=_publish(args,project,describe_project(project))
        elif args.command=="verify-parent":
            previous=read_project(args.parent,expected_sha256=args.expected_parent_sha256)
            child=read_project(args.child,expected_sha256=args.expected_child_sha256)
            report=verify_revision(previous,child)
        else:
            project=read_project(args.project,expected_sha256=args.expected_project_sha256)
            if args.command=="inspect":
                report=describe_project(project)
            elif args.command=="revise":
                _distinct((args.project,args.score,args.automation),args.output)
                revised=revise_project(project,title=args.title,score=_load_score(args),
                                       automation=_load_automation(args))
                report=describe_project(revised)
                report["revision_verification"]=verify_revision(project,revised)
                _publish(args,revised,report)
            elif args.command=="render":
                _distinct((args.project,),args.output)
                wav,report=render_project(project)
                receipt=publish_generated_wav(args.output,wav,overwrite=args.overwrite,
                    expected_sha256=args.expected_output_sha256)
                report.update({"output_sha256":receipt.wav_sha256,"bytes_written":receipt.bytes_written})
            else:
                _distinct((args.project,),args.output)
                kind="score" if args.command=="extract-score" else "automation"
                data=encode_score(project.score) if kind=="score" else encode_automation(project.automation)
                receipt=publish_music_artifact(args.output,data,kind=kind,overwrite=args.overwrite,
                    expected_sha256=args.expected_output_sha256)
                report={"project_sha256":project.sha256,"extracted":kind,
                        "score_sha256":project.score_sha256,"automation_sha256":project.automation_sha256,
                        "output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written}
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio music project: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
