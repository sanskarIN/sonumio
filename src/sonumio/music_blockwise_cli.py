"""Local CLI entry point for bounded-block arrangement rendering."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .arrangement import TRACKS
from .atomic_cli_wav import publish_generated_wav
from .errors import GenerationError
from .music_blockwise import DEFAULT_BLOCK, generate_blockwise_music
from .music_mix import MixPlan
from .music_score_io import read_score


def register_blockwise_music_command(sub):
    parser = sub.add_parser("music-render-blockwise", help="Render a score without full-song float stems")
    parser.add_argument("score", type=Path)
    parser.add_argument("--expected-score-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--block-frames", type=int, default=DEFAULT_BLOCK)
    parser.add_argument("--mute", action="append", choices=TRACKS, default=[])
    parser.add_argument("--solo", action="append", choices=TRACKS, default=[])
    parser.add_argument("--room", type=float, default=.15)
    parser.add_argument("--peak-ceiling", type=float, default=.9)
    return parser


def run_blockwise_music_command(args):
    source = args.score
    target = args.output
    if source.resolve() == target.resolve() or (
        target.exists() and source.exists() and os.path.samefile(source, target)
    ):
        raise GenerationError("Music output cannot overwrite the input score")
    score = read_score(source, expected_sha256=args.expected_score_sha256)
    mix = MixPlan(mute=tuple(args.mute), solo=tuple(args.solo),
                  room=args.room, peak_ceiling=args.peak_ceiling)
    wav, report = generate_blockwise_music(score, mix, block_frames=args.block_frames)
    receipt = publish_generated_wav(target, wav, overwrite=args.overwrite,
                                    expected_sha256=args.expected_output_sha256)
    report.update({"output_sha256": receipt.wav_sha256,
                   "bytes_written": receipt.bytes_written})
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0
