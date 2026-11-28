"""Create and apply score-pinned percussion groove plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_groove import (
    DRUMS,STYLES,MAX_BYTES,DrumRule,GroovePlan,encode_groove,decode_groove,generate_groove,
)
from .music_groove_solver import GrooveConstraints
from .music_project_cli import _distinct,_output
from .music_publish import publish_music_artifact
from .music_score_io import read_score,encode_score
from .arrangement import arrangement_digest


def _read(path,expected):
    with open(path,"rb") as stream:
        plan=decode_groove(stream.read(MAX_BYTES+1))
    if expected is not None and expected!=plan.sha256:
        raise GenerationError("Groove plan SHA256 mismatch")
    return plan


def _parse_rule(text,style):
    fields=text.split(":")
    if len(fields)!=2 or fields[0] not in DRUMS or not fields[1].isascii() or not fields[1].isdigit():
        raise GenerationError("Expected groove rule track:hit-count")
    track,count=fields
    hits=int(count)
    if track=="kick":
        return DrumRule(track,GrooveConstraints(hits,mandatory=(0,) if hits else ()),
                        style=style,velocity=.65,length_ticks=4)
    if track=="snare":
        return DrumRule(track,GrooveConstraints(hits,mandatory=(4,12) if hits>=2 else ()),
                        style=style,velocity=.5,length_ticks=3)
    return DrumRule(track,GrooveConstraints(hits),style=style,velocity=.28,pan=.2)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    for name in ("plan","inspect","apply"):
        command=sub.add_parser(name)
        command.add_argument("score",type=Path)
        command.add_argument("--expected-score-sha256")
        if name=="plan":
            command.add_argument("--first-bar",type=int,default=0)
            command.add_argument("--bars",type=int)
            command.add_argument("--seed",type=int,default=0)
            command.add_argument("--rule",action="append",required=True)
            command.add_argument("--style",choices=STYLES,default="balanced")
            command.add_argument("--include-intro",action="store_true")
            command.add_argument("--include-break",action="store_true")
            command.add_argument("--alternate-bars",action="store_true")
        else:
            command.add_argument("groove",type=Path)
            command.add_argument("--expected-groove-sha256")
        if name=="apply":
            _output(command)
    args=parser.parse_args(argv)
    try:
        score=read_score(args.score,expected_sha256=args.expected_score_sha256)
        if args.action=="plan":
            rules=tuple(sorted((_parse_rule(value,args.style) for value in args.rule),
                               key=lambda rule:DRUMS.index(rule.track)))
            plan=GroovePlan(arrangement_digest(score),args.first_bar,
                score.plan.bars-args.first_bar if args.bars is None else args.bars,args.seed,rules,
                not args.include_intro,not args.include_break,args.alternate_bars)
            generate_groove(score,plan)
            print(encode_groove(plan).decode("ascii"))
            return 0
        plan=_read(args.groove,args.expected_groove_sha256)
        result,report=generate_groove(score,plan)
        if args.action=="apply":
            _distinct((args.score,args.groove),args.output)
            receipt=publish_music_artifact(args.output,encode_score(result),kind="score",
                overwrite=args.overwrite,expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=receipt.sha256,bytes_written=receipt.bytes_written)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio groove: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
