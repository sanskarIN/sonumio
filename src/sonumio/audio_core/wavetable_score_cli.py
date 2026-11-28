"""Render an original mathematical wavetable score into an atomic local WAV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .wavetable_score_io import (
    load_wavetable_score, export_wavetable_score,
)


def register_wavetable_command(sub):
    command = sub.add_parser(
        "wavetable-render",
        help="Render a strict original Fourier note score to verified PCM WAV",
    )
    command.add_argument("--score", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--expected-output-sha256")
    command.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32), default=16)
    command.add_argument("--dither-seed", type=int)
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--create-parent", action="store_true")
    return command


def run_wavetable_command(args):
    try:
        score = load_wavetable_score(args.score)
        report = export_wavetable_score(
            score, args.output,
            score_path=args.score,
            expected_output_sha256=args.expected_output_sha256,
            output_bits=args.output_bits,
            dither_seed=args.dither_seed,
            overwrite=args.overwrite,
            create_parent=args.create_parent,
        )
        payload = json.dumps(report.as_dict(), sort_keys=True, allow_nan=False)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio wavetable: {exc}", file=sys.stderr)
        return 2
    print(payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_wavetable_command(sub)
    return run_wavetable_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
