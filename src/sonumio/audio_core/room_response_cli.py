"""Render a static room from a declarative plan and verified local WAV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .room_response_io import ensure_distinct, export_room_plan, load_room_plan


def register_room_command(sub):
    command = sub.add_parser("room-response", help="Render a static geometric room response")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--source-sha256")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--expected-output-sha256")
    command.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32), default=16)
    command.add_argument("--dither-seed", type=int)
    command.add_argument("--packet-frames", type=int, default=8192)
    command.add_argument("--engine", choices=("matrix", "partitioned"), default="matrix",
                         help="Explicit FIR backend (default preserves legacy replay)")
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--create-parent", action="store_true")
    return command


def run_room_command(args):
    try:
        ensure_distinct(args.plan, args.output)
        result = export_room_plan(
            load_room_plan(args.plan), args.source, args.output,
            expected_source_sha256=args.source_sha256,
            expected_output_sha256=args.expected_output_sha256,
            output_bits=args.output_bits, dither_seed=args.dither_seed,
            packet_frames=args.packet_frames, overwrite=args.overwrite,
            create_parent=args.create_parent, engine=args.engine,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio room response: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_room_command(sub)
    return run_room_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
