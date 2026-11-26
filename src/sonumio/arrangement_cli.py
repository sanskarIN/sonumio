"""Standalone arrangement CLI: compose, inspect, render or export MIDI.

Every command produces at most one atomic file. The score checksum is
distinct from the serialized JSON or WAV byte checksum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .arrangement import ArrangementPlan, TRACKS, compose_arrangement, arrangement_digest
from .music import STYLES, MODES
from .music_score_io import encode_score, read_score
from .music_midi import encode_midi
from .music_mix import MixPlan, generate_arrangement_audio
from .music_analysis import analyze_score
from .music_publish import publish_music_artifact
from .atomic_cli_wav import publish_generated_wav


def _output(parser):
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")


def _input(parser):
    parser.add_argument("score",type=Path)
    parser.add_argument("--expected-score-sha256")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    compose=sub.add_parser("compose",help="Generate a structured symbolic score")
    compose.add_argument("--style",choices=STYLES,default="cinematic")
    compose.add_argument("--root",type=int,default=60)
    compose.add_argument("--mode",choices=tuple(MODES),default="minor")
    compose.add_argument("--bpm",type=int,default=96)
    compose.add_argument("--bars",type=int,default=16)
    compose.add_argument("--seed",type=int,default=0)
    compose.add_argument("--swing",type=float,default=0)
    _output(compose)
    inspect=sub.add_parser("inspect",help="Verify and analyze a symbolic score")
    _input(inspect)
    render=sub.add_parser("render",help="Synthesize a stereo WAV from a verified score")
    _input(render)
    _output(render)
    render.add_argument("--mute",action="append",choices=TRACKS,default=[])
    render.add_argument("--solo",action="append",choices=TRACKS,default=[])
    render.add_argument("--room",type=float,default=.15)
    render.add_argument("--peak-ceiling",type=float,default=.9)
    midi=sub.add_parser("midi",help="Export score as standard MIDI type 1")
    _input(midi)
    _output(midi)
    args=parser.parse_args(argv)
    try:
        if args.command=="compose":
            plan=ArrangementPlan(args.style,args.root,args.mode,args.bpm,args.bars,args.seed,args.swing)
            score=compose_arrangement(plan)
            data=encode_score(score)
            receipt=publish_music_artifact(args.output,data,kind="score",overwrite=args.overwrite,
                                           expected_sha256=args.expected_output_sha256)
            report=analyze_score(score)
            report.update({"output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written})
        else:
            score=read_score(args.score,expected_sha256=args.expected_score_sha256)
            if args.command=="inspect":
                report=analyze_score(score)
            elif args.command=="midi":
                receipt=publish_music_artifact(args.output,encode_midi(score),kind="midi",
                                               overwrite=args.overwrite,
                                               expected_sha256=args.expected_output_sha256)
                report={"score_sha256":arrangement_digest(score),"output_sha256":receipt.sha256,
                        "bytes_written":receipt.bytes_written,"format":"MIDI type 1"}
            else:
                mix=MixPlan(mute=tuple(args.mute),solo=tuple(args.solo),
                            room=args.room,peak_ceiling=args.peak_ceiling)
                data,report=generate_arrangement_audio(score,mix)
                receipt=publish_generated_wav(args.output,data,overwrite=args.overwrite,
                                              expected_sha256=args.expected_output_sha256)
                report.update({"output_sha256":receipt.wav_sha256,"bytes_written":receipt.bytes_written})
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Arrangement failed: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
