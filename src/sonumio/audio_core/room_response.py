"""Deterministic static shoebox-room impulse responses.

This is a finite-order specular image-source approximation. It models
geometric delay and frequency-independent wall reflection amplitude.
It does not model diffraction, scattering, measured reverberation, HRTFs,
air absorption, or pressure at the boundaries. All coordinates use metres.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math

import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, MAX_CHANNELS, check_rate, finite_float,
)
from .errors import AudioContractError, AudioLimitError
from .matrix_convolution import (
    ConvolutionGeometry, ImpulseMatrix, MAX_TAPS, convolve_matrix,
)

MAX_ORDER = 4
MAX_ROOM_LENGTH = 100.0
MAX_PATHS = 32_000


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float

    def __post_init__(self):
        for name in ("x", "y", "z"):
            object.__setattr__(self, name, finite_float(
                getattr(self, name), name, minimum=0, maximum=MAX_ROOM_LENGTH,
            ))

    @property
    def xyz(self):
        return self.x, self.y, self.z


@dataclass(frozen=True, slots=True)
class ShoeboxRoom:
    width: float
    depth: float
    height: float
    # Amplitude reflection coefficients; order is x-, x+, y-, y+, z-, z+.
    reflections: tuple[float, ...] = (.5, .5, .5, .5, .5, .5)

    def __post_init__(self):
        for name in ("width", "depth", "height"):
            object.__setattr__(self, name, finite_float(
                getattr(self, name), name, minimum=.5, maximum=MAX_ROOM_LENGTH,
            ))
        if type(self.reflections) is not tuple or len(self.reflections) != 6:
            raise AudioContractError("Room needs six wall reflection coefficients")
        object.__setattr__(self, "reflections", tuple(
            finite_float(v, "wall reflection", minimum=0, maximum=.99)
            for v in self.reflections
        ))

    @property
    def dimensions(self):
        return self.width, self.depth, self.height

    def validate_point(self, point):
        if not isinstance(point, Point3D) or not all(
            0 < p < length for p, length in zip(point.xyz, self.dimensions)
        ):
            raise AudioContractError("Acoustic point must be strictly inside room")


@dataclass(frozen=True, slots=True)
class RoomConfig:
    sample_rate: int = 24000
    reflection_order: int = 2
    speed_of_sound: float = 343.
    reference_distance: float = 1.
    minimum_distance: float = .25
    include_direct: bool = True
    max_seconds: float = 2.
    block_frames: int = 1024

    def __post_init__(self):
        check_rate(self.sample_rate)
        _integer(self.reflection_order, "reflection order", 0, MAX_ORDER)
        for name, lo, hi in (
            ("speed_of_sound", 300., 380.),
            ("reference_distance", .01, 10.),
            ("minimum_distance", .01, 10.),
            ("max_seconds", .001, 10.),
        ):
            object.__setattr__(self, name, finite_float(
                getattr(self, name), name, minimum=lo, maximum=hi,
            ))
        if type(self.include_direct) is not bool:
            raise AudioContractError("include_direct must be boolean")
        ConvolutionGeometry(1, 1, 1, self.block_frames)


@dataclass(frozen=True, slots=True)
class ReflectionPath:
    receiver: int
    source: int
    image_index: tuple[int, int, int]
    wall_hits: tuple[int, int, int, int, int, int]
    distance_metres: float
    arrival_frames: float
    amplitude: float

    @property
    def order(self):
        return sum(self.wall_hits)

    @property
    def direct(self):
        return self.order == 0


@dataclass(frozen=True, slots=True)
class RoomResponse:
    impulse: ImpulseMatrix
    paths: tuple[ReflectionPath, ...]
    config: RoomConfig

    @property
    def direct_paths(self):
        return tuple(p for p in self.paths if p.direct)


def _axis_image(position, length, index):
    if index % 2:
        image = (index + 1) * length - position
    else:
        image = index * length + position
    magnitude = abs(index)
    if index < 0:
        return image, (magnitude + 1) // 2, magnitude // 2
    return image, magnitude // 2, (magnitude + 1) // 2


def _indices(order):
    # Deterministic order first, then lexicographic image coordinates.
    candidates = [
        v for v in product(range(-order, order + 1), repeat=3)
        if sum(abs(i) for i in v) <= order
    ]
    return tuple(sorted(candidates, key=lambda v: (sum(map(abs, v)), v)))


def _image(room, source, index):
    positions = []
    hits = []
    for position, length, axis in zip(source.xyz, room.dimensions, index):
        image, lower, upper = _axis_image(position, length, axis)
        positions.append(image)
        hits.extend((lower, upper))
    return tuple(positions), tuple(hits)


def _points(room, values, name):
    if type(values) is not tuple or not 1 <= len(values) <= MAX_CHANNELS:
        raise AudioContractError(f"Expected bounded {name} tuple")
    for point in values:
        room.validate_point(point)
    return values


def enumerate_paths(
    room: ShoeboxRoom, sources: tuple[Point3D, ...],
    receivers: tuple[Point3D, ...], config: RoomConfig = RoomConfig(),
) -> tuple[ReflectionPath, ...]:
    """Enumerate finite-order image paths before allocating impulse samples."""
    if not isinstance(room, ShoeboxRoom) or not isinstance(config, RoomConfig):
        raise AudioContractError("Expected ShoeboxRoom and RoomConfig")
    _points(room, sources, "sources")
    _points(room, receivers, "receivers")
    indices = _indices(config.reflection_order)
    if len(sources) * len(receivers) * len(indices) > MAX_PATHS:
        raise AudioLimitError("Room response path count exceeds bound")
    paths = []
    for source_index, source in enumerate(sources):
        for index in indices:
            if index == (0, 0, 0) and not config.include_direct:
                continue
            image, hits = _image(room, source, index)
            reflection = math.prod(
                coefficient ** count
                for coefficient, count in zip(room.reflections, hits)
            )
            if reflection == 0:
                continue
            for receiver_index, receiver in enumerate(receivers):
                distance = math.dist(image, receiver.xyz)
                seconds = distance / config.speed_of_sound
                if seconds > config.max_seconds:
                    continue
                arrival = seconds * config.sample_rate
                amplitude = (
                    reflection * config.reference_distance /
                    max(config.minimum_distance, distance)
                )
                paths.append(ReflectionPath(
                    receiver_index, source_index, index, hits,
                    distance, arrival, amplitude,
                ))
    return tuple(paths)


def build_room_response(
    room: ShoeboxRoom, sources: tuple[Point3D, ...],
    receivers: tuple[Point3D, ...], config: RoomConfig = RoomConfig(),
) -> RoomResponse:
    """Deposit every path into two adjacent integer taps.

    The interpolation preserves each path's coefficient sum and fractional
    centroid. It is a simple linear fractional delay, not a band-limited
    fractional-delay filter. Arrival cutoff is applied before deposition;
    the final interpolation tap can be less than one sample beyond it.
    """
    paths = enumerate_paths(room, sources, receivers, config)
    taps = max(1, max((math.ceil(p.arrival_frames) + 1 for p in paths), default=1))
    if taps > MAX_TAPS:
        raise AudioLimitError("Room impulse exceeds maximum tap count")
    geometry = ConvolutionGeometry(
        len(sources), len(receivers), taps, config.block_frames,
    )
    kernel = np.zeros((geometry.outputs, geometry.inputs, taps), dtype=np.float64)
    for path in paths:
        start = math.floor(path.arrival_frames)
        fraction = path.arrival_frames - start
        channel = kernel[path.receiver, path.source]
        channel[start] += path.amplitude * (1 - fraction)
        if fraction:
            channel[start + 1] += path.amplitude * fraction
    impulse = ImpulseMatrix(
        kernel, config.sample_rate, block_frames=config.block_frames,
    )
    return RoomResponse(impulse, paths, config)


def render_room(
    audio: AudioBuffer, room: ShoeboxRoom,
    sources: tuple[Point3D, ...], receivers: tuple[Point3D, ...],
    config: RoomConfig = RoomConfig(), *, include_tail: bool = True,
    output_layout: AudioLayout | None = None,
) -> AudioBuffer:
    """Render source channels into receiver channels on the source clock."""
    if not isinstance(audio, AudioBuffer) or not isinstance(config, RoomConfig):
        raise AudioContractError("Expected AudioBuffer and RoomConfig")
    if type(include_tail) is not bool:
        raise AudioContractError("include_tail must be boolean")
    if audio.sample_rate != config.sample_rate:
        raise AudioContractError("Room sample rate must match source")
    if type(sources) is not tuple or audio.channels != len(sources):
        raise AudioContractError("Each source channel needs exactly one position")
    if output_layout is not None and (
        not isinstance(output_layout, AudioLayout)
        or type(receivers) is not tuple
        or output_layout.channels != len(receivers)
    ):
        raise AudioContractError("Receiver layout does not match receiver positions")
    response = build_room_response(room, sources, receivers, config)
    return convolve_matrix(
        audio, response.impulse, include_tail=include_tail,
        output_layout=output_layout,
    )
