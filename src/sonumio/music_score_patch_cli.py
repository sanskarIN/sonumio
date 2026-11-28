"""Local diff, apply, reverse and verify commands for immutable score patches."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from .errors import GenerationError
from .music_score_io import read_score,encode_score
from .music_score_patch import (
    MAX_CHAIN,make_patch,apply_with_receipt,inspect_patch,chain_patches,read_patch,encode_patch,merge_scores,
)
from .music_publish import publish_music_artifact

COMMANDS=("music-score-diff","music-patch-inspect","music-patch-apply",
          "music-patch-reverse","music-patch-chain","music-score-merge")


def _score_arg(parser,name):
    parser.add_argument(name,type=Path)
    parser.add_argument("--expected-"+name+"-sha256")


def _patch_arg(parser):
    parser.add_argument("patch",type=Path)
    parser.add_argument("--expected-patch-sha256")


def _output(parser):
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")


def register_score_patch_commands(sub):
    diff=sub.add_parser("music-score-diff",help="Compute a reversible structural score patch")
    _score_arg(diff,"source")
    _score_arg(diff,"target")
    _output(diff)
    inspect=sub.add_parser("music-patch-inspect",help="Inspect a patch without applying it")
    _patch_arg(inspect)
    apply=sub.add_parser("music-patch-apply",help="Apply and verify a score patch")
    _score_arg(apply,"source")
    _patch_arg(apply)
    _output(apply)
    reverse=sub.add_parser("music-patch-reverse",help="Reverse a patch after validating its source")
    _score_arg(reverse,"source")
    _patch_arg(reverse)
    _output(reverse)
    chain=sub.add_parser("music-patch-chain",help="Verify and collapse a series of score patches")
    _score_arg(chain,"source")
    chain.add_argument("patches",type=Path,nargs="+")
    _output(chain)
    merge=sub.add_parser("music-score-merge",help="Merge independent track edits against a common score")
    for name in ("base","left","right"):
        _score_arg(merge,name)
    _output(merge)


def _different(paths,target):
    for path in paths:
        if path.resolve()==target.resolve() or (target.exists() and os.path.samefile(path,target)):
            raise GenerationError("Score patch output cannot replace an input")


def _publish(args,data,kind,report):
    result=publish_music_artifact(args.output,data,kind=kind,overwrite=args.overwrite,
                                  expected_sha256=args.expected_output_sha256)
    report.update({"output_sha256":result.sha256,"bytes_written":result.bytes_written})
    return report


def run_score_patch_command(args):
    try:
        if args.command=="music-patch-inspect":
            report=inspect_patch(read_patch(args.patch,expected_sha256=args.expected_patch_sha256))
        elif args.command=="music-score-diff":
            _different((args.source,args.target),args.output)
            source=read_score(args.source,expected_sha256=args.expected_source_sha256)
            target=read_score(args.target,expected_sha256=args.expected_target_sha256)
            patch=make_patch(source,target)
            _,report=apply_with_receipt(source,patch)
            _publish(args,encode_patch(patch),"patch",report)
        elif args.command=="music-score-merge":
            _different((args.base,args.left,args.right),args.output)
            base=read_score(args.base,expected_sha256=args.expected_base_sha256)
            left=read_score(args.left,expected_sha256=args.expected_left_sha256)
            right=read_score(args.right,expected_sha256=args.expected_right_sha256)
            result,report=merge_scores(base,left,right)
            if result is None:
                print(json.dumps(report,sort_keys=True,allow_nan=False))
                return 2
            _publish(args,encode_score(result),"score",report)
        elif args.command=="music-patch-chain":
            if not 1<=len(args.patches)<=MAX_CHAIN:
                raise GenerationError("Score patch chain exceeds length budget")
            _different((args.source,*args.patches),args.output)
            source=read_score(args.source,expected_sha256=args.expected_source_sha256)
            patches=tuple(read_patch(path) for path in args.patches)
            patch,report=chain_patches(source,patches)
            _publish(args,encode_patch(patch),"patch",report)
        else:
            _different((args.source,args.patch),args.output)
            source=read_score(args.source,expected_sha256=args.expected_source_sha256)
            patch=read_patch(args.patch,expected_sha256=args.expected_patch_sha256)
            target,report=apply_with_receipt(source,patch)
            if args.command=="music-patch-reverse":
                reverse=patch.reversed()
                _,report=apply_with_receipt(target,reverse)
                _publish(args,encode_patch(reverse),"patch",report)
            else:
                _publish(args,encode_score(target),"score",report)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio score patch: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0
