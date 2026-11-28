"""Local music rack inspection and verified WAV publication."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .atomic_cli_wav import publish_generated_wav
from .errors import GenerationError
from .music_automation_cli import _distinct_destination
from .music_rack import read_rack, describe_rack, generate_rack_music
from .music_score_io import read_score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "render"):
        command = sub.add_parser(name)
        command.add_argument("score", type=Path)
        command.add_argument("rack", type=Path)
        command.add_argument("--expected-score-sha256")
        command.add_argument("--expected-rack-sha256")
        if name == "render":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--overwrite", action="store_true")
            command.add_argument("--expected-output-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            _distinct_destination((args.score, args.rack), args.output)
        score = read_score(args.score, expected_sha256=args.expected_score_sha256)
        plan = read_rack(args.rack)
        if args.expected_rack_sha256 is not None and args.expected_rack_sha256 != plan.sha256:
            raise GenerationError("Music rack SHA256 mismatch")
        if args.command == "inspect":
            report = describe_rack(score, plan)
        else:
            wav, report = generate_rack_music(score, plan)
            receipt = publish_generated_wav(args.output, wav, overwrite=args.overwrite,
                                            expected_sha256=args.expected_output_sha256)
            report.update({"output_sha256": receipt.wav_sha256,
                           "bytes_written": receipt.bytes_written})
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio music rack: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
