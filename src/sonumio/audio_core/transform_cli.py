"""CLI adapter for verified offline duration and pitch transformations."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from .time_scale import METHODS, TimeScaleSettings
from .wav_transform import WavTransformConfig, transform_wav_file


def add_transform_parser(subparsers):
    command = subparsers.add_parser(
        "wav-transform",
        help="Change canonical WAV duration or pitch using local NumPy DSP",
    )
    command.add_argument("input", type=Path)
    command.add_argument("--output", type=Path, required=True)
    operation = command.add_mutually_exclusive_group(required=True)
    operation.add_argument("--ratio", type=float,
                           help="Output/input duration ratio, 0.5 to 2")
    operation.add_argument("--target-frames", type=int)
    operation.add_argument("--semitones", type=float,
                           help="Pitch transposition, -12 to +12; duration retained")
    command.add_argument("--method", choices=METHODS,
                         help="Defaults to WSOLA for duration, phase vocoder for pitch")
    command.add_argument("--window-frames", type=int, default=1024)
    command.add_argument("--search-frames", type=int,
                         help="WSOLA search radius; defaults to min(128, window/2)")
    command.add_argument("--fft-hop", type=int)
    command.add_argument("--lobes", type=int, default=24)
    command.add_argument("--peak-ceiling", type=float,
                         help="Optional linear peak attenuation cap; otherwise reject overrange")
    command.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32))
    command.add_argument("--dither-seed", type=int)
    command.add_argument("--packet-frames", type=int, default=8192)
    command.add_argument("--expected-sha256")
    command.add_argument("--expected-output-sha256")
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--create-parent", action="store_true")
    return command


def run_transform_command(args):
    try:
        method = args.method or ("phase_vocoder" if args.semitones is not None else "wsola")
        radius = min(128, args.window_frames // 2) if args.search_frames is None else args.search_frames
        settings = TimeScaleSettings(
            method=method, window_frames=args.window_frames,
            search_frames=radius, fft_hop=args.fft_hop,
            peak_ceiling=args.peak_ceiling,
        )
        config = WavTransformConfig(
            ratio=args.ratio, target_frames=args.target_frames,
            semitones=args.semitones, time_scale=settings, lobes=args.lobes,
            output_bits=args.output_bits, dither_seed=args.dither_seed,
            packet_frames=args.packet_frames,
        )
        report = transform_wav_file(
            args.input, args.output, config=config,
            expected_source_sha256=args.expected_sha256,
            expected_output_sha256=args.expected_output_sha256,
            overwrite=args.overwrite, create_parent=args.create_parent,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio WAV transform: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(), sort_keys=True, allow_nan=False))
    return 0
