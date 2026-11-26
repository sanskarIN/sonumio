"""Standalone original instrumental music CLI using atomic WAV publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .music import MusicPlan, STYLES, MODES, generate_music
from .atomic_cli_wav import publish_generated_wav


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style",choices=STYLES,default="calm")
    parser.add_argument("--root",type=int,default=60,help="MIDI tonic (48–71)")
    parser.add_argument("--mode",choices=tuple(MODES),default="major")
    parser.add_argument("--bpm",type=int,default=100)
    parser.add_argument("--bars",type=int,default=4)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")
    args=parser.parse_args(argv)
    try:
        plan=MusicPlan(args.style,args.root,args.mode,args.bpm,args.bars,args.seed)
        audio,report=generate_music(plan)
        receipt=publish_generated_wav(args.output,audio,overwrite=args.overwrite,
                                      expected_sha256=args.expected_output_sha256)
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"Music generation failed: {exc}\n")
    report["bytes_written"]=receipt.bytes_written
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
