"""Canonical local score plans and verified original wavetable WAV publication.

Score JSON contains only finite numeric synthesis parameters and note events;
it cannot choose source files, plugins, training corpora, model weights,
output destinations, or network endpoints. Output path is a CLI argument.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import os
from pathlib import Path
import stat

import numpy as np

from .dither import QuantizationSettings
from .errors import AudioContractError, AudioFormatError, AudioLimitError
from .hashing import canonical_json_bytes, digest_audio, parse_canonical_json, strict_sha256
from .waveio import encode_wav_bytes
from .wavetable import (
    WavetablePlan, WavetableNote, VoiceEnvelope, render_wavetable,
)

PROTOCOL = "sonumio-original-wavetable-score-v1"
RECEIPT = "sonumio-verified-wavetable-render-v1"
MAX_SCORE_BYTES = 250_000
MAX_OFFLINE_PCM_BYTES = 24 * 1024 * 1024


def _shape(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise AudioContractError(f"Invalid {name} fields")
    return value


@dataclass(frozen=True, slots=True)
class WavetableScore:
    plan: WavetablePlan
    notes: tuple[WavetableNote, ...]

    def __post_init__(self):
        if not isinstance(self.plan, WavetablePlan):
            raise AudioContractError("Score requires a WavetablePlan")
        if type(self.notes) is not tuple or len(self.notes) > 2048:
            raise AudioContractError("Score needs bounded note tuple")
        if any(not isinstance(n, WavetableNote) for n in self.notes):
            raise AudioContractError("Score notes must use WavetableNote")

    def to_bytes(self):
        data = {
            "protocol": PROTOCOL,
            "render": asdict(self.plan),
            "notes": [asdict(note) for note in self.notes],
        }
        return canonical_json_bytes(data, max_bytes=MAX_SCORE_BYTES)

    @property
    def sha256(self):
        return hashlib.sha256(self.to_bytes()).hexdigest()


def parse_wavetable_score(raw: bytes) -> WavetableScore:
    data = parse_canonical_json(raw, max_bytes=MAX_SCORE_BYTES)
    _shape(data, ("protocol", "render", "notes"), "wavetable score")
    if data["protocol"] != PROTOCOL:
        raise AudioFormatError("Unsupported wavetable score protocol")
    spec = _shape(data["render"], (
        "sample_rate", "frames", "channels", "waveform", "table_size",
        "max_harmonics", "master_gain", "envelope",
    ), "render")
    env = _shape(spec["envelope"], (
        "attack_frames", "decay_frames", "sustain_level", "release_frames",
    ), "voice envelope")
    plan = WavetablePlan(
        sample_rate=spec["sample_rate"],
        frames=spec["frames"],
        channels=spec["channels"],
        waveform=spec["waveform"],
        table_size=spec["table_size"],
        max_harmonics=spec["max_harmonics"],
        master_gain=spec["master_gain"],
        envelope=VoiceEnvelope(**env),
    )
    notes = data["notes"]
    if type(notes) is not list or len(notes) > 2048:
        raise AudioContractError("Wavetable score needs bounded note array")
    parsed = tuple(
        WavetableNote(**_shape(item, (
            "start_frame", "gate_frames", "midi_note", "velocity",
            "pan", "initial_phase",
        ), "score note"))
        for item in notes
    )
    return WavetableScore(plan, parsed)


def load_wavetable_score(path):
    """Read one bounded, ordinary UTF8 JSON descriptor without following symlinks."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise AudioFormatError("Expected ordinary wavetable score file")
    flags = (
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    )
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise AudioFormatError("Expected regular score file descriptor")
        if before.st_size > MAX_SCORE_BYTES:
            raise AudioLimitError("Wavetable score exceeds byte budget")
        raw = stream.read(MAX_SCORE_BYTES + 1)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise AudioFormatError("Wavetable score changed during read")
    return parse_wavetable_score(raw)


def _ensure_distinct(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise AudioFormatError("Score and WAV output paths must differ")
    if target.exists() and source.exists():
        try:
            if os.path.samefile(source, target):
                raise AudioFormatError("Score and WAV share file identity")
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class WavetableExport:
    score_sha256: str
    output_sha256: str
    output_pcm_sha256: str
    output_bytes: int
    output_bits: int
    output_frames: int
    output_channels: int
    sample_rate: int
    notes: int
    maximum_polyphony: int
    peak: float
    rms: float
    dither_seed: int | None

    def as_dict(self):
        return {
            "protocol": RECEIPT,
            **asdict(self),
            "source_type": "original_mathematical_synthesis",
            "model_inference": False,
            "binary_model_weights_included": False,
            "source_recording_used": False,
        }


def export_wavetable_score(
    score: WavetableScore, destination, *,
    score_path=None,
    expected_output_sha256=None,
    output_bits=16,
    dither_seed=None,
    overwrite=False,
    create_parent=False,
) -> WavetableExport:
    """Bound rendering, verify SHA before publication and report its digest."""
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(score, WavetableScore):
        raise AudioContractError("Expected WavetableScore")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Invalid WAV overwrite or parent flags")
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    settings = QuantizationSettings(
        bits=output_bits, dither=dither_seed is not None, seed=dither_seed,
    )
    target = Path(destination)
    if target.suffix.lower() != ".wav":
        raise AudioFormatError("Wavetable output must end in .wav")
    if score_path is not None:
        _ensure_distinct(score_path, target)
    if score.plan.frames * score.plan.channels * 4 > MAX_OFFLINE_PCM_BYTES:
        raise AudioLimitError("Score output exceeds offline PCM memory budget")
    rendered = render_wavetable(score.plan, score.notes)
    audio = rendered.audio
    peak = float(np.max(np.abs(audio.samples)))
    if peak > 1.:
        raise AudioContractError("Score output exceeds PCM [-1,+1]; lower master_gain")
    encoded = encode_wav_bytes(audio, settings=settings)
    if score_path is not None:
        _ensure_distinct(score_path, target)
    published = publish_generated_wav(
        target, encoded,
        overwrite=overwrite, create_parent=create_parent,
        expected_sha256=expected_output_sha256,
    )
    return WavetableExport(
        score.sha256,
        published.wav_sha256,
        digest_audio(audio),
        published.bytes_written,
        output_bits, audio.frames, audio.channels, audio.sample_rate,
        rendered.report.notes, rendered.report.maximum_polyphony,
        peak, rendered.report.rms, dither_seed,
    )
