"""Bounded multichannel FIR convolution on a fixed FFT block clock.

The matrix is indexed [receiver, source, tap]. Unlike a channel-wise
reverb, each receiver can have a different response to every source.
Processing is causal, with no circular wrap or implicit normalization.
Streaming calls may split input arbitrarily; the FFT clock does not change.
This is an offline reference, not a real-time latency guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, MAX_BYTES, MAX_CHANNELS,
    check_audio, check_rate, finite_float, make_buffer,
)
from .errors import AudioContractError, AudioLimitError

MAX_TAPS = 262_144
MAX_BLOCK = 16_384
MIN_BLOCK = 64
MAX_WORK_BYTES = 192 * 1024 * 1024
MAX_COMPLEX_PRODUCTS = 200_000_000
MAX_ABS_SAMPLE = 1_000_000.0


def _integer(value: int, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be an integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class ConvolutionGeometry:
    """Resource envelope for a fixed linear-convolution block."""

    inputs: int
    outputs: int
    taps: int
    block: int = 1024

    def __post_init__(self) -> None:
        _integer(self.inputs, "input channels", 1, MAX_CHANNELS)
        _integer(self.outputs, "output channels", 1, MAX_CHANNELS)
        _integer(self.taps, "impulse taps", 1, MAX_TAPS)
        _integer(self.block, "block frames", MIN_BLOCK, MAX_BLOCK)
        if self.block & (self.block - 1):
            raise AudioContractError("Convolution block must be a power of two")
        if self.estimated_work_bytes > MAX_WORK_BYTES:
            raise AudioLimitError("Convolution geometry exceeds work memory")

    @property
    def fft_size(self) -> int:
        return 1 << (self.block + self.taps - 2).bit_length()

    @property
    def bins(self) -> int:
        return self.fft_size // 2 + 1

    @property
    def estimated_work_bytes(self) -> int:
        # Include immutable kernel, temporary FFT input/outputs, spectra,
        # pending input, overlap and scratch, with conservative headroom.
        matrix = self.inputs * self.outputs
        kernel = matrix * self.taps * 8
        spectra = (matrix + 2 * self.inputs + 3 * self.outputs + 4) * self.bins * 16
        time = (2 * self.inputs + 3 * self.outputs + 4) * self.fft_size * 8
        state = (self.inputs + self.outputs) * (self.block + self.taps) * 8
        return 2 * kernel + spectra + time + state

    def validate_work(self, total_input: int) -> None:
        blocks = (total_input + self.block - 1) // self.block
        if blocks * self.inputs * self.outputs * self.bins > MAX_COMPLEX_PRODUCTS:
            raise AudioLimitError("Convolution exceeds cumulative spectral work limit")


@dataclass(frozen=True, slots=True, eq=False)
class ImpulseMatrix:
    """Immutable finite floating-point impulse coefficients on one clock."""

    coefficients: np.ndarray
    sample_rate: int
    block_frames: int = 1024
    normalization: str = "none"

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        x = self.coefficients
        if (not isinstance(x, np.ndarray) or x.ndim != 3
                or x.dtype not in (np.float32, np.float64)):
            raise AudioContractError("Impulse must be a floating receiver/source/tap matrix")
        outputs, inputs, taps = x.shape
        geometry = ConvolutionGeometry(inputs, outputs, taps, self.block_frames)
        if type(self.normalization) is not str or self.normalization not in (
            "none", "receiver_peak", "receiver_energy",
        ):
            raise AudioContractError("Unknown impulse normalization")
        if not np.isfinite(x).all() or np.max(np.abs(x)) > MAX_ABS_SAMPLE:
            raise AudioContractError("Impulse contains invalid or excessive coefficients")
        values = np.array(x, dtype=np.float64, order="C", copy=True)
        if self.normalization != "none":
            for receiver in values:
                if self.normalization == "receiver_peak":
                    # L1 gain bounds peak even when multiple sources add.
                    divisor = float(np.sum(np.abs(receiver)))
                else:
                    divisor = math.sqrt(float(np.sum(receiver * receiver)))
                if divisor:
                    receiver /= divisor
        immutable = np.frombuffer(values.tobytes(), dtype=np.float64)
        object.__setattr__(self, "coefficients", immutable.reshape(
            geometry.outputs, geometry.inputs, geometry.taps,
        ))

    @property
    def geometry(self) -> ConvolutionGeometry:
        outputs, inputs, taps = self.coefficients.shape
        return ConvolutionGeometry(inputs, outputs, taps, self.block_frames)

    @classmethod
    def diagonal(cls, impulse: np.ndarray, rate: int, channels: int,
                 **kwargs) -> "ImpulseMatrix":
        """Apply a shared mono response independently to each channel."""
        _integer(channels, "channels", 1, MAX_CHANNELS)
        if (not isinstance(impulse, np.ndarray) or impulse.ndim != 1
                or impulse.dtype not in (np.float32, np.float64)):
            raise AudioContractError("Expected one floating impulse")
        block = kwargs.get("block_frames", 1024)
        ConvolutionGeometry(channels, channels, len(impulse), block)
        matrix = np.zeros((channels, channels, len(impulse)), dtype=np.float64)
        for i in range(channels):
            matrix[i, i] = impulse
        return cls(matrix, rate, **kwargs)


class MatrixConvolver:
    """Fixed-block overlap-add streaming convolution.

    push emits complete blocks only. finish emits the remaining input clock
    and optionally the causal FIR tail. Concatenating all returned chunks
    yields the linear convolution. Output chunks are float64, including an
    empty array when no complete block is available. The stream is one-shot.
    """

    def __init__(self, impulse: ImpulseMatrix):
        if not isinstance(impulse, ImpulseMatrix):
            raise AudioContractError("Expected ImpulseMatrix")
        self._impulse = impulse
        self._geometry = impulse.geometry
        g = self._geometry
        self._filters = np.fft.rfft(impulse.coefficients, n=g.fft_size, axis=-1)
        self._pending = np.zeros((g.inputs, g.block), dtype=np.float64)
        self._overlap = np.zeros((g.outputs, g.taps - 1), dtype=np.float64)
        self._pending_count = 0
        self._input_frames = 0
        self._emitted_frames = 0
        self._finished = False

    @property
    def input_frames(self) -> int:
        return self._input_frames

    @property
    def emitted_frames(self) -> int:
        return self._emitted_frames

    @property
    def finished(self) -> bool:
        return self._finished

    def _check_open(self) -> None:
        if self._finished:
            raise AudioContractError("Convolution stream is already finished")

    def _process(self, block: np.ndarray) -> np.ndarray:
        g = self._geometry
        length = block.shape[1]
        source = np.fft.rfft(block, n=g.fft_size, axis=-1)
        spectrum = np.zeros((g.outputs, g.bins), dtype=np.complex128)
        # Explicit receiver/source reduction order across all blocks.
        for receiver in range(g.outputs):
            for channel in range(g.inputs):
                spectrum[receiver] += self._filters[receiver, channel] * source[channel]
        time = np.fft.irfft(spectrum, n=g.fft_size, axis=-1)
        linear = time[:, :length + g.taps - 1]
        if g.taps > 1:
            linear[:, :g.taps - 1] += self._overlap
            self._overlap = linear[:, length:].copy()
        return linear[:, :length].copy()

    def _validate_total(self, frames: int) -> None:
        g = self._geometry
        if frames > 120 * self._impulse.sample_rate:
            raise AudioLimitError("Convolution input exceeds duration limit")
        g.validate_work(frames)

    def push(self, samples: np.ndarray) -> np.ndarray:
        self._check_open()
        check_audio(samples)
        g = self._geometry
        x = samples.reshape(1, -1) if samples.ndim == 1 else samples
        if x.shape[0] != g.inputs:
            raise AudioContractError("Convolution source channel count mismatch")
        if np.max(np.abs(x)) > MAX_ABS_SAMPLE:
            raise AudioContractError("Convolution source exceeds amplitude bound")
        frames = x.shape[1]
        self._validate_total(self._input_frames + frames)
        complete = (self._pending_count + frames) // g.block * g.block
        if complete * g.outputs * 8 > MAX_BYTES:
            raise AudioLimitError("Convolution push output exceeds memory limit")
        out = np.empty((g.outputs, complete), dtype=np.float64)
        offset = emitted = 0
        while offset < frames:
            take = min(g.block - self._pending_count, frames - offset)
            self._pending[:, self._pending_count:self._pending_count + take] = (
                x[:, offset:offset + take]
            )
            self._pending_count += take
            offset += take
            if self._pending_count == g.block:
                out[:, emitted:emitted + g.block] = self._process(self._pending)
                emitted += g.block
                self._pending_count = 0
        self._input_frames += frames
        self._emitted_frames += emitted
        return out

    def finish(self, *, include_tail: bool = True) -> np.ndarray:
        self._check_open()
        if type(include_tail) is not bool:
            raise AudioContractError("include_tail must be boolean")
        if not self._input_frames:
            raise AudioContractError("Cannot finish an empty convolution stream")
        g = self._geometry
        tail = g.taps - 1 if include_tail else 0
        out = np.empty((g.outputs, self._pending_count + tail), dtype=np.float64)
        if self._pending_count:
            out[:, :self._pending_count] = self._process(
                self._pending[:, :self._pending_count]
            )
        if tail:
            out[:, self._pending_count:] = self._overlap
        self._pending_count = 0
        self._finished = True
        self._emitted_frames += out.shape[1]
        return out


def convolve_matrix(audio: AudioBuffer, impulse: ImpulseMatrix, *,
                    include_tail: bool = True,
                    output_layout: AudioLayout | None = None) -> AudioBuffer:
    """Render a complete causal response with an explicit receiver layout."""
    if not isinstance(audio, AudioBuffer) or not isinstance(impulse, ImpulseMatrix):
        raise AudioContractError("Expected AudioBuffer and ImpulseMatrix")
    if type(include_tail) is not bool:
        raise AudioContractError("include_tail must be boolean")
    g = impulse.geometry
    if audio.sample_rate != impulse.sample_rate or audio.channels != g.inputs:
        raise AudioContractError("Impulse and source clock or channels disagree")
    layout = output_layout if output_layout is not None else (
        audio.spec.layout if g.inputs == g.outputs else AudioLayout.for_channels(g.outputs)
    )
    if not isinstance(layout, AudioLayout) or layout.channels != g.outputs:
        raise AudioContractError("Convolution receiver layout mismatch")
    frames = audio.frames + (g.taps - 1 if include_tail else 0)
    AudioSpec(audio.sample_rate, layout, frames)
    g.validate_work(audio.frames)
    if frames * g.outputs * 8 > MAX_BYTES:
        raise AudioLimitError("Convolution working output exceeds memory limit")
    engine = MatrixConvolver(impulse)
    result = np.empty((g.outputs, frames), dtype=np.float32)
    offset = 0
    # Keep the transient streaming output small even for a large source.
    for start in range(0, audio.frames, g.block):
        chunk = engine.push(audio.samples[:, start:start + g.block])
        result[:, offset:offset + chunk.shape[1]] = chunk
        offset += chunk.shape[1]
    chunk = engine.finish(include_tail=include_tail)
    result[:, offset:] = chunk
    return make_buffer(result, audio.sample_rate, layout=layout,
                       source_note="causal matrix convolution")


def matrix_reverb(audio: AudioBuffer, impulse: ImpulseMatrix, *,
                  wet: float = .35, include_tail: bool = False) -> AudioBuffer:
    """Mix a same-channel-count response with the dry input at frame zero."""
    if not isinstance(audio, AudioBuffer) or not isinstance(impulse, ImpulseMatrix):
        raise AudioContractError("Expected AudioBuffer and ImpulseMatrix")
    amount = finite_float(wet, "wet", minimum=0, maximum=1)
    if impulse.geometry.outputs != audio.channels:
        raise AudioContractError("Wet/dry mixing requires matching receiver count")
    rendered = convolve_matrix(audio, impulse, include_tail=include_tail)
    result = rendered.mutable_copy(dtype="float64")
    result *= amount
    result[:, :audio.frames] += (1 - amount) * audio.samples
    return make_buffer(result, audio.sample_rate, layout=audio.spec.layout,
                       source_note="matrix convolution reverb")
