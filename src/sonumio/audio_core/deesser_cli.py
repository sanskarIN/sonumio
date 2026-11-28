"""Verified local WAV sibilance-band dynamics and atomic publication.

Input PCM is read through Sonumio's bounded, checksum-aware snapshot
facility. The complete output is encoded and published atomically,
with optional SHA-256 pinning and no implicit source overwrite.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .dither import QuantizationSettings
from .errors import AudioContractError, AudioFormatError, AudioLimitError
from .hashing import digest_audio, strict_sha256
from .room_response_io import ensure_distinct
from .sibilance_dynamics import DeEsserSettings, deess
from .wave_snapshot import read_verified_pcm
from .waveio import encode_wav_bytes


MAX_DEESS_OUTPUT_PCM = 24 * 1024 * 1024


def export_deessed_wav(
    source,
    destination,
    *,
    expected_source_sha256=None,
    expected_output_sha256=None,
    settings: DeEsserSettings = DeEsserSettings(),
    output_bits=16,
    dither_seed=None,
    packet_frames=8192,
    overwrite=False,
    create_parent=False,
):
    """Publish an entirely verified source-clock PCM WAV with provenance."""
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(settings, DeEsserSettings):
        raise AudioContractError("Expected DeEsserSettings")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Invalid de-esser publication flags")
    if expected_source_sha256 is not None:
        strict_sha256(expected_source_sha256)
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    target = Path(destination)
    if target.suffix.lower() != ".wav":
        raise AudioFormatError("De-esser output must have .wav extension")
    ensure_distinct(source, target)
    quantization = QuantizationSettings(
        bits=output_bits, dither=dither_seed is not None, seed=dither_seed,
    )
    snapshot = read_verified_pcm(
        source, expected_sha256=expected_source_sha256,
        packet_frames=packet_frames,
    )
    audio = snapshot.audio
    if audio.frames * audio.channels * 4 > MAX_DEESS_OUTPUT_PCM:
        raise AudioLimitError("De-esser output exceeds offline PCM budget")
    result = deess(audio, settings)
    peak = float(np.max(np.abs(result.audio.samples)))
    if peak > 1:
        raise AudioContractError("De-esser output exceeds PCM [-1,+1]; lower source level")
    payload = encode_wav_bytes(result.audio, settings=quantization)
    ensure_distinct(source, target)
    published = publish_generated_wav(
        target, payload, overwrite=overwrite, create_parent=create_parent,
        expected_sha256=expected_output_sha256,
    )
    return {
        "protocol": "sonumio-verified-deesser-wav-v1",
        "source": {**snapshot.as_dict(),
                   "source_pcm_sha256": digest_audio(audio)},
        "output_sha256": published.wav_sha256,
        "output_pcm_sha256": digest_audio(result.audio),
        "output_bytes": published.bytes_written,
        "output_bits": output_bits,
        "output_channels": result.audio.channels,
        "output_frames": result.audio.frames,
        "sample_rate": result.audio.sample_rate,
        "output_peak": peak,
        "dither_seed": dither_seed,
        "effect": result.report.as_dict(),
        "source_modified": False,
        "model_inference": False,
    }


def register_deesser_command(sub):
    command = sub.add_parser(
        "wav-deess", help="Render verified WAV through non-neural sibilance dynamics",
    )
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--source-sha256")
    command.add_argument("--expected-output-sha256")
    command.add_argument("--fft-frames", type=int, default=1024)
    command.add_argument("--hop-frames", type=int, default=256)
    command.add_argument("--band-low-hz", type=float, default=3500.)
    command.add_argument("--band-high-hz", type=float, default=7800.)
    command.add_argument("--threshold-dbfs", type=float, default=-28.)
    command.add_argument("--ratio", type=float, default=4.)
    command.add_argument("--attack-seconds", type=float, default=.001)
    command.add_argument("--release-seconds", type=float, default=.05)
    command.add_argument("--maximum-reduction-db", type=float, default=18.)
    command.add_argument("--wet", type=float, default=1.)
    command.add_argument("--unlink-channels", action="store_true")
    command.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32), default=16)
    command.add_argument("--dither-seed", type=int)
    command.add_argument("--packet-frames", type=int, default=8192)
    command.add_argument("--overwrite", action="store_true")
    command.add_argument("--create-parent", action="store_true")
    return command


def run_deesser_command(args):
    try:
        options = DeEsserSettings(
            n_fft=args.fft_frames, hop_frames=args.hop_frames,
            band_low_hz=args.band_low_hz,
            band_high_hz=args.band_high_hz,
            threshold_dbfs=args.threshold_dbfs,
            ratio=args.ratio, attack_seconds=args.attack_seconds,
            release_seconds=args.release_seconds,
            maximum_reduction_db=args.maximum_reduction_db,
            wet=args.wet, link_channels=not args.unlink_channels,
        )
        report = export_deessed_wav(
            args.source, args.output,
            expected_source_sha256=args.source_sha256,
            expected_output_sha256=args.expected_output_sha256,
            settings=options, output_bits=args.output_bits,
            dither_seed=args.dither_seed, packet_frames=args.packet_frames,
            overwrite=args.overwrite, create_parent=args.create_parent,
        )
        receipt = json.dumps(report, sort_keys=True, allow_nan=False)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio de-esser: {exc}", file=sys.stderr)
        return 2
    print(receipt)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_deesser_command(sub)
    return run_deesser_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
