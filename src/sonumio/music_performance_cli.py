"""Inspect, generate, and render local sample-clock performance documents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .arrangement import TRACKS
from .atomic_cli_wav import publish_generated_wav
from .errors import GenerationError
from .music_automation import read_automation
from .music_automation_cli import _distinct_destination
from .music_humanize import HumanizePlan, TrackFeel, humanize
from .music_performance import (
    describe_performance, encode_performance, read_performance, generate_performance_music,
)
from .music_score_io import read_score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "render", "humanize"):
        cmd = commands.add_parser(name)
        cmd.add_argument("score", type=Path)
        cmd.add_argument("--expected-score-sha256")
        if name == "humanize":
            cmd.add_argument("--seed", type=int, required=True)
            cmd.add_argument("--track", choices=TRACKS, action="append", required=True)
            cmd.add_argument("--timing-frames", type=int, default=0)
            cmd.add_argument("--shortening-frames", type=int, default=0)
            cmd.add_argument("--velocity-variation", type=float, default=0.)
            cmd.add_argument("--pan-variation", type=float, default=0.)
        else:
            cmd.add_argument("performance", type=Path)
            cmd.add_argument("--expected-performance-sha256")
        if name == "render":
            cmd.add_argument("--automation", type=Path)
            cmd.add_argument("--expected-automation-sha256")
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--overwrite", action="store_true")
            cmd.add_argument("--expected-output-sha256")
    args = parser.parse_args(argv)
    try:
        score = read_score(args.score, expected_sha256=args.expected_score_sha256)
        if args.command == "humanize":
            if len(set(args.track)) != len(args.track):
                raise GenerationError("Duplicate humanization track")
            plan = HumanizePlan(args.seed, tuple(TrackFeel(
                track, args.timing_frames, args.shortening_frames,
                args.velocity_variation, args.pan_variation)
                for track in TRACKS if track in args.track))
            performance, _ = humanize(score, plan)
            # Emit only canonical JSON. The caller chooses how to save it.
            print(encode_performance(performance).decode("ascii"))
            return 0
        performance = read_performance(args.performance)
        if (args.expected_performance_sha256 is not None
                and performance.sha256 != args.expected_performance_sha256):
            raise GenerationError("Performance SHA256 mismatch")
        if args.command == "inspect":
            report = describe_performance(score, performance)
        else:
            paths = [args.score, args.performance]
            automation = None
            if args.automation is not None:
                paths.append(args.automation)
                automation = read_automation(args.automation)
            if (args.expected_automation_sha256 is not None
                    and (automation is None or automation.sha256 != args.expected_automation_sha256)):
                raise GenerationError("Automation SHA256 mismatch")
            _distinct_destination(paths, args.output)
            wav, report = generate_performance_music(score, performance, automation)
            receipt = publish_generated_wav(args.output, wav, overwrite=args.overwrite,
                                            expected_sha256=args.expected_output_sha256)
            report.update({"output_sha256": receipt.wav_sha256,
                           "bytes_written": receipt.bytes_written})
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio performance: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
