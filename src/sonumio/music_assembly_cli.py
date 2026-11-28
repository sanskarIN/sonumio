"""Local CLI for pinning and replaying arrangement assembly recipes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .errors import GenerationError
from .music_assembly import (
    MAX_RECIPE_BYTES,MAX_SOURCES,decode_draft,encode_assembly,read_assembly,assemble,
)
from .music_score_io import read_score,encode_score
from .music_publish import publish_music_artifact


def _bindings(values):
    if not 1<=len(values)<=MAX_SOURCES:
        raise GenerationError("Expected 1–8 assembly source bindings")
    result={}
    for binding in values:
        alias,separator,path=binding.partition("=")
        if not separator or not alias or not path or alias in result:
            raise GenerationError("Assembly source bindings must use unique alias=path pairs")
        result[alias]=Path(path)
    return result


def _distinct(paths,target):
    for path in paths:
        if path.resolve()==target.resolve() or (target.exists() and os.path.samefile(path,target)):
            raise GenerationError("Assembly output cannot overwrite its inputs")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("pin","inspect","assemble"):
        command=sub.add_parser(name)
        command.add_argument("recipe",type=Path)
        command.add_argument("--source",action="append",required=True,help="Local alias=score.json binding")
        if name!="pin":
            command.add_argument("--expected-recipe-sha256")
        if name!="inspect":
            command.add_argument("--output",type=Path,required=True)
            command.add_argument("--overwrite",action="store_true")
            command.add_argument("--expected-output-sha256")
    args=parser.parse_args(argv)
    try:
        paths=_bindings(args.source)
        if args.command!="inspect":
            _distinct((args.recipe,*paths.values()),args.output)
        if args.command=="pin":
            with args.recipe.open("rb") as stream:
                recipe=decode_draft(stream.read(MAX_RECIPE_BYTES+1))
        else:
            recipe=read_assembly(args.recipe,expected_sha256=args.expected_recipe_sha256)
        pins={pin.alias:pin.score_sha256 for pin in recipe.sources}
        if set(paths)!=set(pins):
            raise GenerationError("Assembly source bindings differ from recipe")
        sources={alias:read_score(path,expected_sha256=pins[alias]) for alias,path in paths.items()}
        score,report=assemble(recipe,sources)
        if args.command!="inspect":
            data=encode_assembly(recipe) if args.command=="pin" else encode_score(score)
            # The atomic JSON publisher applies its outer 1 MiB cap.
            # Recipe encoding has already enforced the tighter 64 KiB cap.
            receipt=publish_music_artifact(args.output,data,kind="score",overwrite=args.overwrite,
                                            expected_sha256=args.expected_output_sha256)
            report.update({"output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written,
                           "output_kind":"assembly_recipe" if args.command=="pin" else "score"})
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio music assembly: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
