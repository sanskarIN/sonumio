"""Strict original plucked-string scores and atomic local WAV output.

The score schema only selects bounded mathematical damped-delay synthesis
parameters and sample-clock notes. It does not choose output paths,
external audio assets, executable plugins, or trained model weights.
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
from .hashing import canonical_json_bytes, parse_canonical_json, digest_audio, strict_sha256
from .plucked_string import (
    PluckedStringPlan, PluckNote, StringVoice, render_plucked_strings,
)
from .waveio import encode_wav_bytes

PROTOCOL = "sonumio-original-plucked-score-v1"
RECEIPT = "sonumio-verified-plucked-wav-v1"
MAX_SCORE_BYTES = 250_000
MAX_OFFLINE_PCM_BYTES = 24 * 1024 * 1024


def _fields(data, expected, name):
    if type(data) is not dict or set(data) != set(expected):
        raise AudioContractError(f"Unexpected {name} fields")
    return data


@dataclass(frozen=True, slots=True)
class PluckedScore:
    plan: PluckedStringPlan
    notes: tuple[PluckNote, ...]

    def __post_init__(self):
        if not isinstance(self.plan, PluckedStringPlan):
            raise AudioContractError("Expected PluckedStringPlan")
        if (type(self.notes) is not tuple or len(self.notes) > 1024
                or any(not isinstance(n, PluckNote) for n in self.notes)):
            raise AudioContractError("Expected bounded PluckNote tuple")

    def to_bytes(self):
        return canonical_json_bytes({
            "protocol": PROTOCOL,
            "plan": asdict(self.plan),
            "notes": [asdict(n) for n in self.notes],
        }, max_bytes=MAX_SCORE_BYTES)

    @property
    def sha256(self):
        return hashlib.sha256(self.to_bytes()).hexdigest()


def parse_plucked_score(raw: bytes) -> PluckedScore:
    data = parse_canonical_json(raw, max_bytes=MAX_SCORE_BYTES)
    _fields(data, ("protocol", "plan", "notes"), "plucked score")
    if data["protocol"] != PROTOCOL:
        raise AudioFormatError("Unsupported plucked-score protocol")
    settings = _fields(data["plan"], (
        "sample_rate", "frames", "channels", "gain", "voice",
    ), "plucked plan")
    voice = StringVoice(**_fields(settings["voice"], (
        "damping", "brightness", "pluck_position", "attack_frames",
        "release_frames", "excitation_smoothing",
    ), "string voice"))
    plan = PluckedStringPlan(
        settings["sample_rate"], settings["frames"], settings["channels"],
        settings["gain"], voice,
    )
    notes = data["notes"]
    if type(notes) is not list or len(notes) > 1024:
        raise AudioContractError("Plucked score needs bounded note array")
    return PluckedScore(plan, tuple(
        PluckNote(**_fields(n, (
            "start_frame", "duration_frames", "midi_note", "velocity",
            "pan", "seed",
        ), "pluck note"))
        for n in notes
    ))


def load_plucked_score(path):
    file = Path(path)
    if not file.is_file() or file.is_symlink():
        raise AudioFormatError("Expected ordinary plucked-score descriptor")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | (
        getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    )
    with os.fdopen(os.open(file, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise AudioFormatError("Plucked score must be regular file")
        if before.st_size > MAX_SCORE_BYTES:
            raise AudioLimitError("Plucked score exceeds size bound")
        raw = stream.read(MAX_SCORE_BYTES + 1)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise AudioFormatError("Plucked score changed during read")
    return parse_plucked_score(raw)


def _distinct(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise AudioFormatError("Plucked score and output must differ")
    if target.exists():
        try:
            if os.path.samefile(source, target):
                raise AudioFormatError("Plucked score and WAV share file identity")
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class PluckedExport:
    score_sha256: str
    output_sha256: str
    output_pcm_sha256: str
    output_bytes: int
    output_bits: int
    frames: int
    channels: int
    sample_rate: int
    note_count: int
    maximum_polyphony: int
    peak: float
    rms: float
    dither_seed: int | None

    def as_dict(self):
        return {
            "protocol": RECEIPT,
            **asdict(self),
            "source_type": "original_plucked_string_synthesis",
            "model_inference": False,
            "source_recording_used": False,
            "physical_accuracy_certified": False,
        }


def export_plucked_score(
    score: PluckedScore,
    destination,
    *,
    score_path=None,
    expected_output_sha256=None,
    output_bits=16,
    dither_seed=None,
    overwrite=False,
    create_parent=False,
):
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(score, PluckedScore):
        raise AudioContractError("Expected PluckedScore")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Invalid plucked output publication flags")
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    settings = QuantizationSettings(
        bits=output_bits, dither=dither_seed is not None, seed=dither_seed,
    )
    target = Path(destination)
    if target.suffix.lower() != ".wav":
        raise AudioFormatError("Plucked output must end in .wav")
    if score_path is not None:
        _distinct(score_path, target)
    if score.plan.frames * score.plan.channels * 4 > MAX_OFFLINE_PCM_BYTES:
        raise AudioLimitError("Plucked PCM output exceeds memory budget")
    output = render_plucked_strings(score.plan, score.notes)
    audio = output.audio
    peak = float(np.max(np.abs(audio.samples)))
    if peak > 1:
        raise AudioContractError("Plucked output exceeds normalized PCM; reduce gain")
    encoded = encode_wav_bytes(audio, settings=settings)
    if score_path is not None:
        _distinct(score_path, target)
    pub = publish_generated_wav(
        target, encoded, overwrite=overwrite, create_parent=create_parent,
        expected_sha256=expected_output_sha256,
    )
    return PluckedExport(
        score.sha256, pub.wav_sha256, digest_audio(audio),
        pub.bytes_written, output_bits, audio.frames, audio.channels,
        audio.sample_rate, output.report.notes, output.report.maximum_polyphony,
        peak, output.report.rms, dither_seed,
    )
