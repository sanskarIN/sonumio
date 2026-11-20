"""Defensive, immutable channel-first audio buffers and allocation limits.

This code never trusts NumPy shape metadata from untrusted callers without
checking dtype, finite samples, size, channel count and sample rate. The
immutable storage is a copy, not merely a read-only view into mutable input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import numpy as np

from .errors import AudioContractError, AudioLimitError

MIN_RATE = 8_000
MAX_RATE = 192_000
MAX_CHANNELS = 8
MAX_SECONDS = 120.0
MAX_BYTES = 128 * 1024 * 1024
LayoutName = Literal["mono", "stereo", "surround", "discrete"]


def check_rate(rate: int) -> int:
    if type(rate) is not int or not MIN_RATE <= rate <= MAX_RATE:
        raise AudioContractError("sample_rate must be integer in [8000, 192000]")
    return rate


def finite_float(value: float, name: str, *, minimum: float | None = None,
                 maximum: float | None = None) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise AudioContractError(name + " must be a finite number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise AudioContractError(name + " below minimum")
    if maximum is not None and result > maximum:
        raise AudioContractError(name + " above maximum")
    return result


def validate_allocation(channels: int, frames: int, *,
                        itemsize: int = 4, max_bytes: int = MAX_BYTES) -> int:
    for name, value in (("channels", channels), ("frames", frames),
                        ("itemsize", itemsize), ("max_bytes", max_bytes)):
        if type(value) is not int or value <= 0:
            raise AudioContractError(name + " must be a positive integer")
    if channels > MAX_CHANNELS:
        raise AudioLimitError("Too many audio channels")
    if frames > max_bytes // channels // itemsize:
        raise AudioLimitError("Audio allocation exceeds configured memory bound")
    return channels * frames * itemsize


def _check_metadata(value: np.ndarray, *, allow_float64: bool) -> None:
    if not isinstance(value, np.ndarray):
        raise AudioContractError("Expected a NumPy PCM array")
    if value.ndim not in (1, 2) or value.size == 0:
        raise AudioContractError("Audio must be nonempty 1-D or 2-D")
    if value.dtype not in ((np.dtype("float32"), np.dtype("float64"))
                           if allow_float64 else (np.dtype("float32"),)):
        raise AudioContractError("PCM samples must use an allowed floating dtype")
    channels, frames = (1, value.shape[0]) if value.ndim == 1 else value.shape
    validate_allocation(int(channels), int(frames),
                        itemsize=int(value.dtype.itemsize))


def check_audio(samples: np.ndarray, *, allow_float64: bool = True,
                allow_overrange: bool = True) -> np.ndarray:
    """Return original float32/float64 array after strict validation.

    This function does not copy. Callers that retain buffers must use
    `make_buffer` to separate ownership from mutable input.
    """
    _check_metadata(samples, allow_float64=allow_float64)
    if not np.isfinite(samples).all():
        raise AudioContractError("PCM audio contains NaN or infinite samples")
    if not allow_overrange and np.any(np.abs(samples) > 1.0):
        raise AudioContractError("PCM input exceeds normalized [-1,+1] range")
    return samples


def as_channel_first(samples: np.ndarray, *,
                     channels_last: bool = False, copy: bool = True,
                     float64: bool = False) -> np.ndarray:
    """Create a canonical C-contiguous (channels, frames) float PCM array.

    A mono 1-D input is interpreted as one channel. A 2-D array is
    channel-first unless channels_last=True is explicit. There is no
    automatic guessed transpose; ambiguous 2-D inputs fail the channel
    limit rather than silently interleaving audio.
    """
    if not isinstance(samples, np.ndarray) or samples.ndim not in (1, 2):
        raise AudioContractError("Audio array must have one or two dimensions")
    if channels_last and samples.ndim == 1:
        channels_last = False
    if samples.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise AudioContractError("Floating PCM dtype is required")
    samples = samples.T if channels_last else samples
    check_audio(samples)
    dtype = np.float64 if float64 else np.float32
    out = np.array(samples, dtype=dtype, copy=copy, order="C")
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if not out.flags.c_contiguous:
        out = np.ascontiguousarray(out)
    return out


@dataclass(frozen=True, slots=True)
class AudioLayout:
    """Explicit channel topology (never inferred from descriptive name)."""

    channels: int
    name: LayoutName = "discrete"
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.channels) is not int or not 1 <= self.channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid channel count")
        if self.name not in ("mono", "stereo", "surround", "discrete"):
            raise AudioContractError("Unknown channel topology")
        if self.name == "mono" and self.channels != 1:
            raise AudioContractError("Mono requires one channel")
        if self.name == "stereo" and self.channels != 2:
            raise AudioContractError("Stereo requires two channels")
        if self.name == "surround" and self.channels < 3:
            raise AudioContractError("Surround requires at least three channels")
        if not isinstance(self.labels, tuple) or (
            self.labels and len(self.labels) != self.channels
        ):
            raise AudioContractError("Channel labels must match channel count")
        if any(not isinstance(v, str) or not v or len(v) > 32
               for v in self.labels):
            raise AudioContractError("Invalid channel label")
        if len(set(self.labels)) != len(self.labels):
            raise AudioContractError("Duplicate channel label")

    @classmethod
    def for_channels(cls, channels: int) -> "AudioLayout":
        if channels == 1:
            return cls(1, "mono", ("M",))
        if channels == 2:
            return cls(2, "stereo", ("L", "R"))
        return cls(channels, "discrete")


@dataclass(frozen=True, slots=True)
class AudioSpec:
    """Portable sample geometry and explicit memory/duration budget."""

    sample_rate: int
    layout: AudioLayout
    frames: int
    dtype: str = "float32"

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if not isinstance(self.layout, AudioLayout):
            raise AudioContractError("Expected AudioLayout")
        if self.dtype not in ("float32", "float64"):
            raise AudioContractError("Unsupported in-memory PCM dtype")
        itemsize = 8 if self.dtype == "float64" else 4
        validate_allocation(self.layout.channels, self.frames, itemsize=itemsize)
        if self.frames > int(self.sample_rate * MAX_SECONDS):
            raise AudioLimitError("Audio exceeds maximum in-memory duration")

    @property
    def channels(self) -> int:
        return self.layout.channels

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate

    @property
    def byte_count(self) -> int:
        return self.channels * self.frames * (8 if self.dtype == "float64" else 4)

    def same_clock(self, other: "AudioSpec") -> bool:
        return isinstance(other, AudioSpec) and self.sample_rate == other.sample_rate

    def same_shape(self, other: "AudioSpec") -> bool:
        return self.same_clock(other) and self.frames == other.frames and (
            self.channels == other.channels
        )

    def with_frames(self, frames: int) -> "AudioSpec":
        return AudioSpec(self.sample_rate, self.layout, frames, self.dtype)


@dataclass(frozen=True, slots=True, eq=False)
class AudioBuffer:
    """Owns read-only finite PCM data, always channel-first and float32.

    The buffer cannot be written through normal NumPy access. Its data
    pointer is private to this container because construction copies the
    caller's input. Use `mutable_copy` for effects/rendering.
    """

    samples: np.ndarray
    spec: AudioSpec
    source_note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.spec, AudioSpec):
            raise AudioContractError("AudioBuffer needs an AudioSpec")
        if self.spec.dtype != "float32":
            raise AudioContractError("AudioBuffer canonical dtype is float32")
        data = as_channel_first(self.samples, copy=True)
        if data.shape != (self.spec.channels, self.spec.frames):
            raise AudioContractError("Audio shape disagrees with explicit AudioSpec")
        if not isinstance(self.source_note, str) or len(self.source_note) > 500:
            raise AudioContractError("Invalid source note")
        # A writable owner array with write=False can have that flag reset
        # by its consumer. Immutable bytes backing prevents that bypass.
        frozen = np.frombuffer(data.tobytes(order="C"), dtype=np.float32)
        frozen = frozen.reshape(data.shape)
        object.__setattr__(self, "samples", frozen)

    @property
    def sample_rate(self) -> int:
        return self.spec.sample_rate

    @property
    def channels(self) -> int:
        return self.spec.channels

    @property
    def frames(self) -> int:
        return self.spec.frames

    @property
    def seconds(self) -> float:
        return self.spec.seconds

    def mutable_copy(self, *, dtype: str = "float32") -> np.ndarray:
        if dtype not in ("float32", "float64"):
            raise AudioContractError("Unsupported working buffer dtype")
        return np.array(self.samples, dtype=dtype, copy=True, order="C")

    def copy_with(self, samples: np.ndarray, *,
                  source_note: str | None = None) -> "AudioBuffer":
        array = as_channel_first(samples)
        spec = self.spec.with_frames(array.shape[1])
        if array.shape[0] != self.channels:
            raise AudioContractError("Audio channel count changed unexpectedly")
        return AudioBuffer(array, spec, self.source_note if source_note is None
                           else source_note)

    def sliced(self, start: int, stop: int) -> "AudioBuffer":
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.frames):
            raise AudioContractError("Invalid bounded frame slice")
        return AudioBuffer(self.samples[:, start:stop],
                           self.spec.with_frames(stop - start), self.source_note)


def make_buffer(samples: np.ndarray, sample_rate: int, *,
                channels_last: bool = False,
                layout: AudioLayout | None = None,
                source_note: str = "") -> AudioBuffer:
    check_rate(sample_rate)
    array = as_channel_first(samples, channels_last=channels_last)
    if array.shape[1] > int(sample_rate * MAX_SECONDS):
        raise AudioLimitError("Audio exceeds maximum in-memory duration")
    actual_layout = layout if layout is not None else AudioLayout.for_channels(
        array.shape[0]
    )
    if actual_layout.channels != array.shape[0]:
        raise AudioContractError("Declared layout does not match PCM channels")
    spec = AudioSpec(sample_rate, actual_layout, array.shape[1])
    return AudioBuffer(array, spec, source_note)
