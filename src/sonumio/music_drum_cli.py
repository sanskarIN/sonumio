"""Local percussion patch inspection and combined patch rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_drum_bank import read_drum_bank,encode_drum_bank,describe_combined_banks,generate_combined_music
from .music_drum_presets import PRESETS,preset_drum_bank
from .music_patch_bank import read_patch_bank
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
            command.add_argument("--assign",action="append",required=True,help="drum=preset")
        else:
            command.add_argument("drums",type=Path)
            command.add_argument("--expected-drums-sha256")
            command.add_argument("--pitched",type=Path)
            command.add_argument("--expected-pitched-sha256")
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
                track,separator,preset=value.partition("=")
                if not separator or track not in PRESETS or preset not in PRESETS[track] or track in assignments:
                    raise GenerationError("Expected unique drum=preset assignments")
                assignments[track]=preset
            drums=preset_drum_bank(score,assignments)
            describe_combined_banks(score,drums)
            print(encode_drum_bank(drums).decode("ascii"))
            return 0
        drums=read_drum_bank(args.drums,expected_sha256=args.expected_drums_sha256)
        pitched=read_patch_bank(args.pitched,expected_sha256=args.expected_pitched_sha256) if args.pitched is not None else None
        if args.expected_pitched_sha256 is not None and pitched is None:
            raise GenerationError("Pitched digest pin requires a patch bank")
        report=describe_combined_banks(score,drums,pitched)
        if args.action=="render":
            _distinct((args.score,args.drums,args.pitched,args.automation),args.output)
            mix=read_automation(args.automation) if args.automation is not None else None
            if args.expected_automation_sha256 is not None and (mix is None or mix.sha256!=args.expected_automation_sha256):
                raise GenerationError("Automation digest mismatch")
            wav,report=generate_combined_music(score,drums,pitched,mix)
            receipt=publish_generated_wav(args.output,wav,overwrite=args.overwrite,
                expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=receipt.wav_sha256,bytes_written=receipt.bytes_written)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,OSError,TypeError,RuntimeError) as exc:
        print(f"Sonumio percussion patch: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
