"""Verified PCM WAV preview of original symbolic scores using Sonumio audio_core.

This module adapts the immutable MIDI workstation score to the EXISTING
finite-Fourier wavetable engine. It does not add another synthesizer, claim
model training, import commercial samples, or execute MIDI SysEx messages.
WAV publication is intentionally separate from editing or saving a score.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import math

from .contracts import MidiIntegrityError, MidiLimitError, MidiWorkstationError, integer, sha256
from .publish import _distinct
from .score import Score
from .wavetable_bridge import WavetableBridgeSettings, render_score_wavetable

MAX_PREVIEW_FLOAT_BYTES = 24 * 1024 * 1024
MAX_PREVIEW_WAV_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ScorePreviewOptions:
    bridge: WavetableBridgeSettings = WavetableBridgeSettings()
    output_bits: int = 16
    dither_seed: int | None = None
    overwrite: bool = False
    create_parent: bool = False
    expected_output_sha256: str | None = None

    def __post_init__(self):
        if not isinstance(self.bridge, WavetableBridgeSettings):
            raise MidiWorkstationError("Preview requires WavetableBridgeSettings")
        if type(self.output_bits) is not int or self.output_bits not in (8, 16, 24, 32):
            raise MidiWorkstationError("PCM preview requires 8/16/24/32-bit integer samples")
        if self.dither_seed is not None:
            integer(self.dither_seed, "preview dither seed", 0, 2**32 - 1)
        if type(self.overwrite) is not bool or type(self.create_parent) is not bool:
            raise MidiWorkstationError("Preview publication flags must be bool")
        if self.expected_output_sha256 is not None:
            sha256(self.expected_output_sha256, "expected preview WAV digest")


@dataclass(frozen=True, slots=True)
class ScorePreviewReport:
    source_score_sha256: str
    wav_sha256: str
    pcm_float_sha256: str
    output_frames: int
    output_channels: int
    sample_rate: int
    output_bits: int
    bytes_written: int
    peak: float
    rms: float
    rendered_notes: int
    skipped_percussion_notes: int
    peak_polyphony: int
    dither_seed: int | None
    protocol: str = "sonumio-score-preview-wav-v1"

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "original_mathematical_synthesis": True,
            "trained_model_inference": False,
            "external_samples_used": False,
            "source_score_modified": False,
            "perceptual_quality_established": False,
            "remote_publication": False,
        }


def export_score_preview(
    score: Score, destination: str | Path,
    options: ScorePreviewOptions = ScorePreviewOptions(), *,
    source_path: str | Path | None = None,
) -> ScorePreviewReport:
    """Render and durably publish one hash-checked PCM preview.

    The memory budget is enforced before any synthesis allocation, and the
    score/source destination distinction is checked both before and after
    rendering. The underlying generated-WAV publisher writes atomically and
    rejects unapproved replacement or output symlink targets.
    """
    if not isinstance(score, Score) or not isinstance(options, ScorePreviewOptions):
        raise MidiWorkstationError("Expected immutable Score and preview options")
    output = Path(destination)
    if output.suffix.lower() != ".wav":
        raise MidiWorkstationError("Preview output must be a .wav file")
    _distinct(source_path, output)

    # Validate buffer budget at the exact declared score/sample clock before
    # running the engine; avoid expensive synthesis with an unpublishable WAV.
    from .transport import TransportSettings, build_schedule
    transport = TransportSettings(
        sample_rate=options.bridge.sample_rate,
        tail_frames=options.bridge.release_frames,
        include_muted=options.bridge.include_muted,
    )
    schedule = build_schedule(score, transport)
    float_bytes = schedule.output_frames * options.bridge.channels * 4
    if float_bytes > MAX_PREVIEW_FLOAT_BYTES:
        raise MidiLimitError("Score preview exceeds offline float32 PCM budget")
    expected_wav_bytes = 44 + (schedule.output_frames * options.bridge.channels
                               * (options.output_bits // 8))
    # Account for an optional RIFF pad byte at odd payload length.
    if expected_wav_bytes + 1 > MAX_PREVIEW_WAV_BYTES:
        raise MidiLimitError("Score preview exceeds offline WAV publication budget")

    try:
        from sonumio.audio_core.dither import QuantizationSettings
        from sonumio.audio_core.hashing import digest_audio
        from sonumio.audio_core.waveio import encode_wav_bytes
        from sonumio.audio_core.errors import AudioContractError, AudioProcessingError
        from sonumio.atomic_cli_wav import publish_generated_wav
    except ImportError as exc:
        raise MidiWorkstationError("Sonumio audio_core is required for PCM preview") from exc

    try:
        quantization = QuantizationSettings(
            bits=options.output_bits,
            dither=options.dither_seed is not None,
            seed=options.dither_seed,
        )
        rendered = render_score_wavetable(score, options.bridge)
        audio = rendered.audio
        if audio.frames != schedule.output_frames:
            raise MidiIntegrityError("Wavetable render disagrees with score clock")
        if (audio.sample_rate != options.bridge.sample_rate
                or audio.channels != options.bridge.channels):
            raise MidiIntegrityError("Rendered audio channel/rate contract mismatch")
        peak = rendered.report.output_peak
        if not math.isfinite(peak) or peak > 1.:
            raise MidiWorkstationError(
                "Preview peak exceeds PCM [-1,+1]; reduce master gain"
            )
        pcm_float_sha256 = digest_audio(audio)
        encoded = encode_wav_bytes(audio, settings=quantization)
        _distinct(source_path, output)
        published = publish_generated_wav(
            output, encoded, overwrite=options.overwrite,
            create_parent=options.create_parent,
            expected_sha256=options.expected_output_sha256,
        )
    except (AudioContractError, AudioProcessingError) as exc:
        raise MidiWorkstationError(f"Score preview audio-core contract: {exc}") from exc

    return ScorePreviewReport(
        source_score_sha256=score.sha256,
        wav_sha256=published.wav_sha256,
        pcm_float_sha256=pcm_float_sha256,
        output_frames=audio.frames,
        output_channels=audio.channels,
        sample_rate=audio.sample_rate,
        output_bits=options.output_bits,
        bytes_written=published.bytes_written,
        peak=rendered.report.output_peak,
        rms=rendered.report.output_rms,
        rendered_notes=rendered.report.rendered_notes,
        skipped_percussion_notes=rendered.report.skipped_percussion_notes,
        peak_polyphony=rendered.report.peak_polyphony,
        dither_seed=options.dither_seed,
    )
