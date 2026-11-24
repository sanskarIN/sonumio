"""Reproducible, source-only procedural demo with explicit audio provenance.

The preset does not use neural inference, external recordings or model
weights. Run with python -m sonumio.demo --output outputs/forest-storm.wav.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from . import __version__
from .atomic_cli_wav import publish_generated_wav
from .audio_core.stream_quality import inspect_wav_quality
from .service import generate_audio

PROMPT = "Rain, thunder, wind and birds in a forest"
SECONDS = 12.0
SEED = 42
ENGINE = "procedural-sfx-v0"


def create_demo(
    destination: str | Path, *, overwrite: bool = False,
    expected_sha256: str | None = None,
) -> dict:
    """Render and check the fixed preset before atomic local publication."""
    generated = generate_audio(PROMPT, SECONDS, model_id=ENGINE, seed=SEED)
    audio = generated.result
    if (
        audio.engine_id != ENGINE or audio.seed != SEED
        or audio.sample_rate != 24000 or audio.samples.shape != (288000,)
        or not np.isfinite(audio.samples).all()
        or float(np.max(np.abs(audio.samples))) > .921
    ):
        raise RuntimeError("Unexpected procedural demo audio contract")
    digest = hashlib.sha256(generated.wav_bytes).hexdigest()
    # The publisher verifies RIFF geometry and bytes, and refuses to replace
    # an existing file unless explicitly authorized. Its trusted SHA check
    # happens before creating a parent directory.
    published = publish_generated_wav(
        destination, generated.wav_bytes,
        overwrite=overwrite, expected_sha256=digest if expected_sha256 is None else expected_sha256,
    )
    # Inspect the actual published file, not only the in-memory waveform.
    # If this subsequent read fails, the already-published WAV may remain;
    # no verified report is returned in that case.
    checked = inspect_wav_quality(
        published.path, expected_sha256=published.wav_sha256,
        include_quiet_intervals=True, include_continuity=True,
    )
    if checked.info.frames != 288000 or checked.info.channels != 1 or (
        checked.info.sample_rate != 24000 or checked.info.bits_per_sample != 16
    ):
        raise RuntimeError("Unexpected published demo WAV geometry")
    return {
        "protocol": "sonumio-procedural-demo-v1",
        "software_version": __version__,
        "numpy_version": np.__version__,
        "prompt": PROMPT,
        "seed": SEED,
        "duration_seconds": SECONDS,
        "engine_id": ENGINE,
        "original_model_inference": False,
        "third_party_recordings_used": False,
        "wav_sha256": published.wav_sha256,
        "bytes_written": published.bytes_written,
        "inspection": checked.as_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate and verify the source-only Sonumio procedural demo",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/forest-storm.wav"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    try:
        report = create_demo(
            args.output, overwrite=args.overwrite,
            expected_sha256=args.expected_sha256,
        )
        payload = json.dumps(report, sort_keys=True, allow_nan=False)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio demo: {exc}", file=sys.stderr)
        return 2
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
