"""Verified local WAV front-end for rhythmic and monophonic pitch analysis.

No source WAV rewrite, network request, model loading or arbitrary file
selection happens here. The existing canonical packet reader verifies all
source bytes before invoking a bounded offline feature extractor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .wave_snapshot import read_verified_pcm
from .rhythm import RhythmSettings, analyze_rhythm
from .yin_pitch import PitchSettings, track_pitch


def register_audio_analysis_commands(sub):
    beat = sub.add_parser(
        "wav-rhythm", help="Inspect verified PCM onset and tempo hypotheses",
    )
    beat.add_argument("input", type=Path)
    beat.add_argument("--expected-sha256")
    beat.add_argument("--packet-frames", type=int, default=8192)
    beat.add_argument("--fft-frames", type=int, default=512)
    beat.add_argument("--hop-frames", type=int, default=128)
    beat.add_argument("--min-bpm", type=float, default=50.)
    beat.add_argument("--max-bpm", type=float, default=220.)
    beat.add_argument("--min-strength", type=float, default=.15)
    beat.add_argument("--prominence", type=float, default=.08)
    beat.add_argument("--min-spacing-seconds", type=float, default=.08)
    beat.add_argument("--summary-only", action="store_true",
                      help="Suppress per-onset and estimated beat-frame lists")
    pitch = sub.add_parser(
        "wav-pitch", help="Inspect verified PCM monophonic F0 hypotheses",
    )
    pitch.add_argument("input", type=Path)
    pitch.add_argument("--expected-sha256")
    pitch.add_argument("--packet-frames", type=int, default=8192)
    pitch.add_argument("--frame-frames", type=int, default=1024)
    pitch.add_argument("--hop-frames", type=int, default=256)
    pitch.add_argument("--min-hz", type=float, default=65.)
    pitch.add_argument("--max-hz", type=float, default=1400.)
    pitch.add_argument("--yin-threshold", type=float, default=.15)
    pitch.add_argument("--silence-dbfs", type=float, default=-60.)
    pitch.add_argument("--summary-only", action="store_true",
                       help="Suppress individual voiced/unvoiced pitch frames")
    return beat, pitch


def _analyze_rhythm(args, audio):
    options = RhythmSettings(
        n_fft=args.fft_frames,
        hop_frames=args.hop_frames,
        min_bpm=args.min_bpm,
        max_bpm=args.max_bpm,
        min_strength=args.min_strength,
        prominence=args.prominence,
        min_spacing_seconds=args.min_spacing_seconds,
    )
    report = analyze_rhythm(audio, options)
    result = report.as_dict()
    result["onset_count"] = len(report.onsets)
    result["tempo_candidate_count"] = len(report.tempos)
    if args.summary_only:
        result.pop("onsets")
        result.pop("beat_frames")
    return result


def _analyze_pitch(args, audio):
    options = PitchSettings(
        frame_frames=args.frame_frames,
        hop_frames=args.hop_frames,
        minimum_hz=args.min_hz,
        maximum_hz=args.max_hz,
        yin_threshold=args.yin_threshold,
        silence_dbfs=args.silence_dbfs,
    )
    report = track_pitch(audio, options)
    result = report.as_dict()
    if args.summary_only:
        result.pop("frames")
    return result


def run_audio_analysis_command(args) -> int:
    if args.command not in ("wav-rhythm", "wav-pitch"):
        raise ValueError("Unknown audio inspection command")
    try:
        snapshot = read_verified_pcm(
            args.input, expected_sha256=args.expected_sha256,
            packet_frames=args.packet_frames,
        )
        if args.command == "wav-rhythm":
            analysis = _analyze_rhythm(args, snapshot.audio)
        else:
            analysis = _analyze_pitch(args, snapshot.audio)
        document = {
            "protocol": "sonumio-local-audio-analysis-v1",
            "command": args.command,
            "source": snapshot.as_dict(),
            "analysis": analysis,
            "source_modified": False,
            "model_inference": False,
        }
        # Serialize completely before touching stdout: invalid data do not
        # leave an incomplete JSON object in a consumption pipeline.
        payload = json.dumps(document, sort_keys=True, allow_nan=False)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio local analysis: {exc}", file=sys.stderr)
        return 2
    print(payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_audio_analysis_commands(sub)
    return run_audio_analysis_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
