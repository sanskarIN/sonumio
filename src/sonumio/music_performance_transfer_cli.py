"""Inspect or publish an event-identity performance transfer or merge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_project_cli import _distinct, _output
from .music_publish import publish_music_artifact
from .music_score_io import read_score
from .music_performance import read_performance, encode_performance
from .music_performance_transfer import (
    MATCH_MODES, REMOVAL_MODES, CLOCK_MODES,
    TransferPolicy, transfer_performance, merge_performances,
)


def _input(parser, name):
    parser.add_argument(name, type=Path)
    parser.add_argument("--expected-"+name.replace("_","-")+"-sha256")


def _score(args, name):
    return read_score(getattr(args,name),expected_sha256=getattr(args,"expected_"+name+"_sha256"))


def _performance(args,name):
    result=read_performance(getattr(args,name))
    pin=getattr(args,"expected_"+name+"_sha256")
    if pin is not None and result.sha256!=pin:
        raise GenerationError("Performance SHA256 mismatch: "+name)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    for action in ("preview-transfer","transfer","preview-merge","merge"):
        command=sub.add_parser(action)
        if action.endswith("transfer"):
            names=("source","target","performance")
            command.add_argument("--match",choices=MATCH_MODES,default="exact")
            command.add_argument("--missing",choices=REMOVAL_MODES,default="reject")
            command.add_argument("--clock",choices=CLOCK_MODES,default="identical")
        else:
            names=("score","base","left","right")
        for name in names:
            _input(command,name)
        if not action.startswith("preview"):
            _output(command)
    args=parser.parse_args(argv)
    try:
        if args.action.endswith("transfer"):
            source,target=_score(args,"source"),_score(args,"target")
            original=_performance(args,"performance")
            result,report=transfer_performance(source,target,original,
                TransferPolicy(args.match,args.missing,args.clock))
            inputs=(args.source,args.target,args.performance)
        else:
            score=_score(args,"score")
            result,report=merge_performances(score,*(_performance(args,name) for name in ("base","left","right")))
            inputs=(args.score,args.base,args.left,args.right)
        if result is None:
            print(json.dumps(report,sort_keys=True,allow_nan=False))
            return 3
        if not args.action.startswith("preview"):
            _distinct(inputs,args.output)
            receipt=publish_music_artifact(args.output,encode_performance(result),kind="score",
                overwrite=args.overwrite,expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=receipt.sha256,bytes_written=receipt.bytes_written)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,OSError,TypeError,RuntimeError) as exc:
        print(f"Sonumio performance transfer: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
