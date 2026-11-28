"""Render a verified score with a bounded local mixer automation document."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .atomic_cli_wav import publish_generated_wav
from .errors import GenerationError
from .music_automation import generate_automated_music, read_automation
from .music_blockwise_automation import generate_blockwise_automated_music
from .music_score_io import read_score


def register_automation_command(sub):
    command = sub.add_parser("music-automate", help="Render score-timed gain and balance automation")
    command.add_argument("score", type=Path)
    command.add_argument("automation", type=Path)
    command.add_argument("--expected-score-sha256")
    command.add_argument("--expected-automation-sha256")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--expected-output-sha256")
    command.add_argument("--block-frames",type=int,help="Use bounded two-pass rendering (256–16384 frames)")
    return command


def _distinct_destination(inputs, output):
    for source in inputs:
        if source.resolve() == output.resolve():
            raise GenerationError("Music input and output must differ")
        if output.exists() and os.path.samefile(source, output):
            raise GenerationError("Music input and output share file identity")


def run_automation_command(args):
    try:
        _distinct_destination((args.score, args.automation), args.output)
        score = read_score(args.score, expected_sha256=args.expected_score_sha256)
        plan = read_automation(args.automation)
        if (args.expected_automation_sha256 is not None
                and plan.sha256 != args.expected_automation_sha256):
            raise GenerationError("Automation plan SHA256 mismatch")
        if args.block_frames is None:
            wav, report = generate_automated_music(score, plan)
        else:
            wav, report = generate_blockwise_automated_music(score, plan, block_frames=args.block_frames)
        receipt = publish_generated_wav(args.output, wav, overwrite=args.overwrite,
                                        expected_sha256=args.expected_output_sha256)
        report.update({"output_sha256": receipt.wav_sha256,
                       "bytes_written": receipt.bytes_written})
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio music automation: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    register_automation_command(parser.add_subparsers(dest="command", required=True))
    return run_automation_command(parser.parse_args(["music-automate", *(sys.argv[1:] if argv is None else argv)]))


if __name__ == "__main__":
    raise SystemExit(main())
