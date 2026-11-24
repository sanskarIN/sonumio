"""Standalone developer CLI for the verified streaming spectrum inspector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .stream_spectrum import SpectrumConfig, inspect_wav_spectrum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure one-sided block spectrum of verified canonical PCM WAV",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--fft-frames", type=int, default=1024)
    parser.add_argument("--packet-frames", type=int, default=8192)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--include-bins", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = inspect_wav_spectrum(
            args.input, config=SpectrumConfig(args.fft_frames),
            packet_frames=args.packet_frames,
            expected_sha256=args.expected_sha256,
        )
        # Build the entire strict JSON representation before emitting it.
        payload = json.dumps(
            report.as_dict(include_bins=args.include_bins),
            sort_keys=True, allow_nan=False,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio WAV spectrum: {exc}", file=sys.stderr)
        return 2
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
