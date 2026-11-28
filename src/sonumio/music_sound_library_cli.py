"""Create, inspect, merge and transfer reusable original sound libraries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .arrangement import TRACKS
from .errors import GenerationError
from .music_project_cli import _distinct,_output
from .music_publish import publish_music_artifact
from .music_score_io import read_score
from .music_patch_bank import read_patch_bank,encode_patch_bank
from .music_drum_bank import read_drum_bank,encode_drum_bank
from .music_sound_library import (
    read_library,encode_library,original_library,library_from_banks,rename_entry,merge_libraries,
)
from .music_sound_assignment import (
    make_assignment,read_assignment,encode_assignment,compile_assignment,transfer_assignment,transfer_library,
)


def _library(parser,label="library"):
    parser.add_argument(label,type=Path)
    parser.add_argument("--expected-"+label+"-sha256")


def _score(parser,label="score"):
    parser.add_argument(label,type=Path)
    parser.add_argument("--expected-"+label+"-sha256")


def _assignment(parser):
    parser.add_argument("assignment",type=Path)
    parser.add_argument("--expected-assignment-sha256")


def _pairs(values,allowed=None):
    result={}
    for value in values:
        key,separator,item=value.partition("=")
        if not separator or not key or not item or key in result or (allowed is not None and key not in allowed):
            raise GenerationError("Expected unique name=value pairs")
        result[key]=item
    return result


def _publish(args,data,kind,inputs):
    _distinct(inputs,args.output)
    receipt=publish_music_artifact(args.output,data,kind=kind,overwrite=args.overwrite,
                                   expected_sha256=args.expected_output_sha256)
    return {"output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written}


def _describe(library):
    return {"library_sha256":library.sha256,"entries":[
        {"name":entry.name,"kind":entry.kind,"timbre_sha256":entry.timbre_sha256}
        for entry in library.entries],"entry_count":len(library.entries)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    builtin=sub.add_parser("original")
    _output(builtin)
    inspect=sub.add_parser("inspect")
    _library(inspect)
    capture=sub.add_parser("capture")
    _score(capture)
    capture.add_argument("--pitched",type=Path)
    capture.add_argument("--expected-pitched-sha256")
    capture.add_argument("--drums",type=Path)
    capture.add_argument("--expected-drums-sha256")
    capture.add_argument("--prefix",default="custom")
    _output(capture)
    rename=sub.add_parser("rename")
    _library(rename)
    rename.add_argument("old")
    rename.add_argument("new")
    _output(rename)
    merge=sub.add_parser("merge")
    _library(merge,"left")
    _library(merge,"right")
    _output(merge)
    assign=sub.add_parser("assign")
    _score(assign)
    _library(assign)
    assign.add_argument("--track",action="append",required=True)
    _output(assign)
    for name in ("inspect-assignment","extract-pitched","extract-drums"):
        command=sub.add_parser(name)
        _score(command)
        _library(command)
        _assignment(command)
        if name!="inspect-assignment":
            _output(command)
    transfer=sub.add_parser("transfer-score")
    _score(transfer,"source")
    _score(transfer,"target")
    _library(transfer)
    _assignment(transfer)
    _output(transfer)
    revise=sub.add_parser("transfer-library")
    _score(revise)
    _library(revise,"source")
    _library(revise,"target")
    _assignment(revise)
    revise.add_argument("--rename",action="append",default=[])
    _output(revise)
    args=parser.parse_args(argv)
    try:
        if args.action=="original":
            library=original_library()
            report=_describe(library)
            report.update(_publish(args,encode_library(library),"sound_library",()))
        elif args.action=="capture":
            score=read_score(args.score,expected_sha256=args.expected_score_sha256)
            if args.pitched is None and args.expected_pitched_sha256 is not None:
                raise GenerationError("Pitched pin requires a patch bank")
            if args.drums is None and args.expected_drums_sha256 is not None:
                raise GenerationError("Drum pin requires a drum bank")
            pitched=read_patch_bank(args.pitched,expected_sha256=args.expected_pitched_sha256) if args.pitched else None
            drums=read_drum_bank(args.drums,expected_sha256=args.expected_drums_sha256) if args.drums else None
            library=library_from_banks(score,pitched,drums,prefix=args.prefix)
            report=_describe(library)
            report.update(_publish(args,encode_library(library),"sound_library",(args.score,args.pitched,args.drums)))
        elif args.action=="merge":
            left=read_library(args.left,expected_sha256=args.expected_left_sha256)
            right=read_library(args.right,expected_sha256=args.expected_right_sha256)
            result,report=merge_libraries(left,right)
            if result is None:
                print(json.dumps(report,sort_keys=True,allow_nan=False))
                return 2
            report.update(_publish(args,encode_library(result),"sound_library",(args.left,args.right)))
        elif args.action=="transfer-score":
            old=read_score(args.source,expected_sha256=args.expected_source_sha256)
            new=read_score(args.target,expected_sha256=args.expected_target_sha256)
            library=read_library(args.library,expected_sha256=args.expected_library_sha256)
            plan=read_assignment(args.assignment,expected_sha256=args.expected_assignment_sha256)
            result,report=transfer_assignment(old,new,library,plan)
            report.update(_publish(args,encode_assignment(result),"sound_assignment",
                (args.source,args.target,args.library,args.assignment)))
        elif args.action=="transfer-library":
            score=read_score(args.score,expected_sha256=args.expected_score_sha256)
            old=read_library(args.source,expected_sha256=args.expected_source_sha256)
            new=read_library(args.target,expected_sha256=args.expected_target_sha256)
            plan=read_assignment(args.assignment,expected_sha256=args.expected_assignment_sha256)
            result,report=transfer_library(score,old,new,plan,rename_map=_pairs(args.rename))
            report.update(_publish(args,encode_assignment(result),"sound_assignment",
                (args.score,args.source,args.target,args.assignment)))
        else:
            library=read_library(args.library,expected_sha256=args.expected_library_sha256)
            if args.action=="inspect":
                report=_describe(library)
            elif args.action=="rename":
                result=rename_entry(library,args.old,args.new)
                report=_describe(result)
                report.update(_publish(args,encode_library(result),"sound_library",(args.library,)))
            else:
                score=read_score(args.score,expected_sha256=args.expected_score_sha256)
                if args.action=="assign":
                    plan=make_assignment(score,library,_pairs(args.track,TRACKS))
                    _,_,report=compile_assignment(score,library,plan)
                    report.update(_publish(args,encode_assignment(plan),"sound_assignment",(args.score,args.library)))
                else:
                    plan=read_assignment(args.assignment,expected_sha256=args.expected_assignment_sha256)
                    pitched,drums,report=compile_assignment(score,library,plan)
                    if args.action!="inspect-assignment":
                        data=encode_patch_bank(pitched) if args.action=="extract-pitched" else encode_drum_bank(drums)
                        kind="pitched_bank" if args.action=="extract-pitched" else "drum_bank"
                        report.update(_publish(args,data,kind,(args.score,args.library,args.assignment)))
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio sound library: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
