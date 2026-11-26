"""Verified local soundtrack rendering from two original score documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arrangement import TRACKS
from .music_mix import MixPlan
from .music_score_io import read_score
from .nature_scene_io import read_scene
from .soundtrack import Placement,Ducking,SoundtrackPlan,generate_soundtrack
from .atomic_cli_wav import publish_generated_wav


def _placement(parser,prefix,default_gain):
    parser.add_argument("--"+prefix+"-source-start",type=int,default=0)
    parser.add_argument("--"+prefix+"-destination-start",type=int,default=0)
    parser.add_argument("--"+prefix+"-frames",type=int)
    parser.add_argument("--"+prefix+"-gain-db",type=float,default=default_gain)
    parser.add_argument("--"+prefix+"-fade-in",type=int,default=480)
    parser.add_argument("--"+prefix+"-fade-out",type=int,default=1200)


def _read_placement(args,prefix):
    return Placement(*(getattr(args,prefix+"_"+name) for name in
                       ("source_start","destination_start","frames","gain_db","fade_in","fade_out")))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score",type=Path)
    parser.add_argument("scene",type=Path)
    parser.add_argument("--expected-score-sha256")
    parser.add_argument("--expected-scene-sha256")
    parser.add_argument("--frames",type=int,required=True,help="Output frames at 24000 Hz")
    _placement(parser,"music",-3.)
    _placement(parser,"ambience",-6.)
    parser.add_argument("--mute",choices=TRACKS,action="append",default=[])
    parser.add_argument("--solo",choices=TRACKS,action="append",default=[])
    parser.add_argument("--duck",action="store_true")
    parser.add_argument("--duck-threshold-dbfs",type=float,default=-30.)
    parser.add_argument("--duck-ratio",type=float,default=3.)
    parser.add_argument("--duck-max-db",type=float,default=9.)
    parser.add_argument("--duck-attack",type=float,default=.05)
    parser.add_argument("--duck-release",type=float,default=.3)
    parser.add_argument("--peak-ceiling",type=float,default=.9)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")
    args=parser.parse_args(argv)
    try:
        score=read_score(args.score,expected_sha256=args.expected_score_sha256)
        scene=read_scene(args.scene,expected_sha256=args.expected_scene_sha256)
        plan=SoundtrackPlan(
            args.frames,_read_placement(args,"music"),_read_placement(args,"ambience"),
            MixPlan(mute=tuple(args.mute),solo=tuple(args.solo)),
            Ducking(args.duck,args.duck_threshold_dbfs,args.duck_ratio,args.duck_max_db,
                    args.duck_attack,args.duck_release),args.peak_ceiling)
        data,report=generate_soundtrack(score,scene,plan)
        receipt=publish_generated_wav(args.output,data,overwrite=args.overwrite,
                                      expected_sha256=args.expected_output_sha256)
        report.update({"output_sha256":receipt.wav_sha256,"bytes_written":receipt.bytes_written})
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Soundtrack failed: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
