"""Inspect or apply a pinned constrained melody transformation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_automation_cli import _distinct_destination
from .music_melody_variation import read_melody_plan, vary_melody
from .music_publish import publish_music_artifact
from .music_score_io import read_score, encode_score


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    for action in ("inspect","apply"):
        command=sub.add_parser(action)
        command.add_argument("score",type=Path)
        command.add_argument("variation",type=Path)
        command.add_argument("--expected-score-sha256")
        command.add_argument("--expected-variation-sha256")
        if action=="apply":
            command.add_argument("--output",type=Path,required=True)
            command.add_argument("--overwrite",action="store_true")
            command.add_argument("--expected-output-sha256")
    args=parser.parse_args(argv)
    try:
        if args.action=="apply":
            _distinct_destination((args.score,args.variation),args.output)
        score=read_score(args.score,expected_sha256=args.expected_score_sha256)
        plan=read_melody_plan(args.variation)
        if args.expected_variation_sha256 is not None and plan.sha256!=args.expected_variation_sha256:
            raise GenerationError("Variation SHA256 mismatch")
        result,report=vary_melody(score,plan)
        if args.action=="apply":
            receipt=publish_music_artifact(args.output,encode_score(result),kind="score",
                                           overwrite=args.overwrite,
                                           expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=receipt.sha256,bytes_written=receipt.bytes_written)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,RuntimeError,OSError) as exc:
        print(f"Sonumio melodic variation: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
