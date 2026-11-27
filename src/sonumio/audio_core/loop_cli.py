"""Local CLI for inspecting and constructing circular PCM loops.

Run as python -m sonumio.audio_core.loop_cli. The parser can also be
registered by the primary command-line entry point without duplicating
argument or publication logic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .loop_analysis import LoopSearchPlan
from .loop_render import LoopRenderPlan, LAWS
from .loop_wav import search_wav_loop, export_wav_loop


def _common(parser):
    parser.add_argument("input", type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--stop-frame", type=int, required=True,
                        help="Exclusive source stop frame")
    parser.add_argument("--blend-frames", type=int, default=0,
                        help="Consumes this many source frames from one output cycle")
    parser.add_argument("--packet-frames", type=int, default=8192)
    parser.add_argument("--expected-sha256")


def _search_options(parser):
    parser.add_argument("--start-radius", type=int, default=128)
    parser.add_argument("--stop-radius", type=int, default=128)
    parser.add_argument("--context-frames", type=int, default=256)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--refine-count", type=int, default=4)
    parser.add_argument("--results", type=int, default=8)
    parser.add_argument("--minimum-period", type=int, default=2)
    parser.add_argument("--maximum-period", type=int)


def register_loop_commands(sub):
    search = sub.add_parser("loop-search", help="Rank local circular PCM endpoints")
    _common(search)
    _search_options(search)
    render = sub.add_parser("loop-render", help="Export a circular PCM loop")
    _common(render)
    _search_options(render)
    render.add_argument("--auto-search", action="store_true")
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--law", choices=LAWS, default="cosine")
    render.add_argument("--repeats", type=int, default=1)
    render.add_argument("--peak-ceiling", type=float)
    render.add_argument("--output-bits", type=int, choices=(8,16,24,32))
    render.add_argument("--dither-seed", type=int)
    render.add_argument("--expected-output-sha256")
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--create-parent", action="store_true")


def _plan(args):
    return LoopSearchPlan(
        args.start_frame, args.stop_frame, start_radius=args.start_radius,
        stop_radius=args.stop_radius, context_frames=args.context_frames,
        stride=args.stride, refine_count=args.refine_count,
        results=args.results, minimum_period=args.minimum_period,
        maximum_period=args.maximum_period, blend_frames=args.blend_frames,
    )


def run_loop_command(args):
    try:
        if args.command == "loop-search":
            result = search_wav_loop(
                args.input, _plan(args), expected_sha256=args.expected_sha256,
                packet_frames=args.packet_frames,
            )
        else:
            if args.auto_search:
                settings = dict(search=_plan(args), law=args.law,
                                repeats=args.repeats, peak_ceiling=args.peak_ceiling)
            else:
                settings = dict(plan=LoopRenderPlan(
                    args.start_frame, args.stop_frame, args.blend_frames,
                    args.law, args.repeats, args.peak_ceiling,
                ))
            result = export_wav_loop(
                args.input, args.output, **settings,
                expected_sha256=args.expected_sha256,
                expected_output_sha256=args.expected_output_sha256,
                packet_frames=args.packet_frames, output_bits=args.output_bits,
                dither_seed=args.dither_seed, overwrite=args.overwrite,
                create_parent=args.create_parent,
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio circular PCM loop: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_loop_commands(sub)
    return run_loop_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
