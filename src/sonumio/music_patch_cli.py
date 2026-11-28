"""Inspect and render original instrument patches, or emit preset documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .arrangement import PITCHED
from .errors import GenerationError
from .music_patch_bank import read_patch_bank,encode_patch_bank,describe_patch_bank,generate_patched_music
from .music_patch_presets import NAMES,preset_bank
from .music_automation import read_automation
from .music_project_cli import _distinct,_output
from .music_score_io import read_score
from .atomic_cli_wav import publish_generated_wav


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    for name in ("presets","inspect","render"):
        command=sub.add_parser(name)
        command.add_argument("score",type=Path)
        command.add_argument("--expected-score-sha256")
        if name=="presets":
            command.add_argument("--assign",action="append",required=True,help="track=preset")
        else:
            command.add_argument("patches",type=Path)
            command.add_argument("--expected-patches-sha256")
        if name=="render":
            command.add_argument("--automation",type=Path)
            command.add_argument("--expected-automation-sha256")
            _output(command)
    args=parser.parse_args(argv)
    try:
        score=read_score(args.score,expected_sha256=args.expected_score_sha256)
        if args.action=="presets":
            assignments={}
            for value in args.assign:
                track,separator,name=value.partition("=")
                if not separator or track not in PITCHED or name not in NAMES or track in assignments:
                    raise GenerationError("Expected unique pitched track=preset assignments")
                assignments[track]=name
            bank=preset_bank(score,assignments)
            describe_patch_bank(score,bank)
            print(encode_patch_bank(bank).decode("ascii"))
            return 0
        bank=read_patch_bank(args.patches,expected_sha256=args.expected_patches_sha256)
        report=describe_patch_bank(score,bank)
        if args.action=="render":
            _distinct((args.score,args.patches,args.automation),args.output)
            mix=read_automation(args.automation) if args.automation is not None else None
            if args.expected_automation_sha256 is not None and (mix is None or mix.sha256!=args.expected_automation_sha256):
                raise GenerationError("Automation input digest mismatch")
            wav,report=generate_patched_music(score,bank,mix)
            receipt=publish_generated_wav(args.output,wav,overwrite=args.overwrite,
                expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=receipt.wav_sha256,bytes_written=receipt.bytes_written)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,OSError,TypeError,RuntimeError) as exc:
        print(f"Sonumio instrument patch: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
