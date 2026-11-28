"""Publish an original local plucked-string score as digest-verified PCM WAV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .plucked_score_io import load_plucked_score, export_plucked_score


def register_plucked_command(sub):
    command = sub.add_parser(
        "plucked-render", help="Render strict original plucked-string score to WAV",
    )
    command.add_argument("--score", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--expected-output-sha256")
    command.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32), default=16)
    command.add_argument("--dither-seed", type=int)
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--create-parent", action="store_true")
    return command


def run_plucked_command(args):
    try:
        score = load_plucked_score(args.score)
        report = export_plucked_score(
            score, args.output, score_path=args.score,
            expected_output_sha256=args.expected_output_sha256,
            output_bits=args.output_bits, dither_seed=args.dither_seed,
            overwrite=args.overwrite, create_parent=args.create_parent,
        )
        payload = json.dumps(report.as_dict(), sort_keys=True, allow_nan=False)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio plucked string: {exc}", file=sys.stderr)
        return 2
    print(payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_plucked_command(sub)
    return run_plucked_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
