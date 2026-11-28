"""Strict room plans and verified local WAV rendering.

Plans contain geometry and DSP parameters only. Source and destination
paths are supplied explicitly by the caller. No network, plugins, models,
or external files are selected by a plan.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from pathlib import Path
import stat

import numpy as np

from .contracts import AudioSpec, AudioLayout, MAX_CHANNELS
from .dither import QuantizationSettings
from .errors import AudioContractError, AudioFormatError, AudioLimitError
from .hashing import canonical_json_bytes, digest_audio, parse_canonical_json, strict_sha256
from .room_response import (
    Point3D, ShoeboxRoom, RoomConfig, build_room_response, room_convolver,
)
from .wave_snapshot import read_verified_pcm
from .waveio import encode_wav_bytes

MAX_PLAN_BYTES = 16_384
MAX_OUTPUT_PCM_BYTES = 24 * 1024 * 1024
PROTOCOL = "sonumio-static-room-plan-v1"


def _fields(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise AudioContractError(f"Unexpected {label} fields")
    return value


def _point(value):
    _fields(value, ("x", "y", "z"), "point")
    return Point3D(**value)


def _point_list(value):
    if type(value) is not list or not 1 <= len(value) <= MAX_CHANNELS:
        raise AudioContractError("Expected bounded position array")
    return tuple(_point(v) for v in value)


@dataclass(frozen=True, slots=True)
class RoomPlan:
    room: ShoeboxRoom
    sources: tuple[Point3D, ...]
    receivers: tuple[Point3D, ...]
    config: RoomConfig
    include_tail: bool = True
    output_gain: float = 1.

    def __post_init__(self):
        from .contracts import finite_float
        if not isinstance(self.room, ShoeboxRoom) or not isinstance(self.config, RoomConfig):
            raise AudioContractError("Room plan needs room and render configuration")
        for group in (self.sources, self.receivers):
            if type(group) is not tuple or not 1 <= len(group) <= MAX_CHANNELS:
                raise AudioContractError("Invalid room plan positions")
            for point in group:
                self.room.validate_point(point)
        if type(self.include_tail) is not bool:
            raise AudioContractError("include_tail must be boolean")
        object.__setattr__(self, "output_gain", finite_float(
            self.output_gain, "output gain", minimum=0, maximum=1,
        ))

    def to_bytes(self):
        data = asdict(self)
        # JSON arrays, never Python tuples or implementation-specific objects.
        data["room"]["reflections"] = list(self.room.reflections)
        data["sources"] = [asdict(p) for p in self.sources]
        data["receivers"] = [asdict(p) for p in self.receivers]
        return canonical_json_bytes({"protocol": PROTOCOL, **data})

    @property
    def sha256(self):
        return hashlib.sha256(self.to_bytes()).hexdigest()


def parse_room_plan(raw):
    obj = parse_canonical_json(raw, max_bytes=MAX_PLAN_BYTES)
    _fields(obj, ("protocol", "room", "sources", "receivers", "config",
                  "include_tail", "output_gain"), "room plan")
    if obj["protocol"] != PROTOCOL:
        raise AudioFormatError("Unsupported room plan protocol")
    room = _fields(obj["room"], ("width", "depth", "height", "reflections"), "room")
    if type(room["reflections"]) is not list:
        raise AudioContractError("Wall reflections must be a JSON array")
    config = _fields(obj["config"], (
        "sample_rate", "reflection_order", "speed_of_sound", "reference_distance",
        "minimum_distance", "include_direct", "max_seconds", "block_frames",
    ), "room config")
    return RoomPlan(
        ShoeboxRoom(room["width"], room["depth"], room["height"],
                    tuple(room["reflections"])),
        _point_list(obj["sources"]), _point_list(obj["receivers"]),
        RoomConfig(**config), obj["include_tail"], obj["output_gain"],
    )


def load_room_plan(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AudioFormatError("Expected ordinary room plan file")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(os.open(path, flags), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise AudioFormatError("Expected regular room plan descriptor")
        if before.st_size > MAX_PLAN_BYTES:
            raise AudioLimitError("Room plan exceeds byte limit")
        raw = stream.read(MAX_PLAN_BYTES + 1)
        after = os.fstat(stream.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise AudioFormatError("Room plan changed during read")
    return parse_room_plan(raw)


def ensure_distinct(source, target):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise AudioFormatError("Room input and output must differ")
    if target.exists():
        try:
            same = os.path.samefile(source, target)
        except FileNotFoundError:
            same = False
        if same:
            raise AudioFormatError("Room input and output share file identity")


@dataclass(frozen=True, slots=True)
class RoomExport:
    plan_sha256: str
    source: dict
    output_sha256: str
    output_pcm_sha256: str
    output_bytes: int
    output_bits: int
    output_frames: int
    output_channels: int
    impulse_frames: int
    path_count: int
    peak: float
    dither_seed: int | None
    engine: str = "matrix"

    def as_dict(self):
        data = asdict(self)
        # Preserve the original receipt shape for existing matrix consumers.
        protocol = "sonumio-verified-room-wav-v1"
        if self.engine == "matrix":
            data.pop("engine")
        else:
            protocol = "sonumio-verified-room-wav-v2"
        return {"protocol": protocol, **data, "model_inference": False}


def export_room_plan(plan, source, destination, *, expected_source_sha256=None,
                     expected_output_sha256=None, output_bits=16,
                     dither_seed=None, packet_frames=8192,
                     overwrite=False, create_parent=False, engine="matrix"):
    """Verify source bytes, render bounded PCM, publish one complete WAV."""
    from ..atomic_cli_wav import publish_generated_wav
    if not isinstance(plan, RoomPlan):
        raise AudioContractError("Expected RoomPlan")
    convolver = room_convolver(engine)
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Invalid room publication flags")
    if expected_output_sha256 is not None:
        strict_sha256(expected_output_sha256)
    if expected_source_sha256 is not None:
        strict_sha256(expected_source_sha256)
    settings = QuantizationSettings(
        bits=output_bits, dither=dither_seed is not None, seed=dither_seed,
    )
    target = Path(destination)
    if target.suffix.lower() != ".wav":
        raise AudioFormatError("Room output must end in .wav")
    ensure_distinct(source, target)
    snapshot = read_verified_pcm(
        source, expected_sha256=expected_source_sha256, packet_frames=packet_frames,
    )
    audio = snapshot.audio
    if audio.channels != len(plan.sources) or audio.sample_rate != plan.config.sample_rate:
        raise AudioContractError("Input PCM geometry differs from room plan")
    response = build_room_response(plan.room, plan.sources, plan.receivers, plan.config)
    frames = audio.frames + (response.impulse.geometry.taps - 1 if plan.include_tail else 0)
    layout = AudioLayout.for_channels(len(plan.receivers))
    spec = AudioSpec(audio.sample_rate, layout, frames)
    if spec.byte_count > MAX_OUTPUT_PCM_BYTES:
        raise AudioLimitError("Room output exceeds offline PCM budget")
    rendered = convolver(audio, response.impulse, include_tail=plan.include_tail,
                               output_layout=layout)
    # Explicit attenuation only; do not silently clip or normalize.
    if plan.output_gain != 1:
        rendered = rendered.copy_with(
            rendered.samples * plan.output_gain, source_note="room output attenuation",
        )
    peak = float(np.max(np.abs(rendered.samples)))
    if peak > 1:
        raise AudioContractError("Room output exceeds normalized PCM; reduce output_gain")
    encoded = encode_wav_bytes(rendered, settings=settings)
    ensure_distinct(source, target)
    publication = publish_generated_wav(
        target, encoded, overwrite=overwrite, create_parent=create_parent,
        expected_sha256=expected_output_sha256,
    )
    return RoomExport(
        plan.sha256, {**snapshot.as_dict(), "source_pcm_sha256": digest_audio(audio)},
        publication.wav_sha256, digest_audio(rendered), publication.bytes_written,
        output_bits, rendered.frames, rendered.channels, response.impulse.geometry.taps,
        len(response.paths), peak, dither_seed, engine,
    )
