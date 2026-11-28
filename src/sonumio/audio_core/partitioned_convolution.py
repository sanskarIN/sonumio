"""Uniform partitioned multichannel FIR convolution.

The frequency-delay line uses a 2B-point FFT for every B-frame block,
independently of impulse length. The kernel is indexed receiver/source/tap.
The fixed block clock makes results independent of caller chunk boundaries.
No normalization, resampling, or real-time deadline is implied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, MAX_BYTES, MAX_CHANNELS,
    MAX_SECONDS, check_audio, make_buffer,
)
from .errors import AudioContractError, AudioLimitError
from .matrix_convolution import (
    ImpulseMatrix, MAX_ABS_SAMPLE, MAX_BLOCK, MAX_TAPS, MIN_BLOCK,
)

MAX_PARTITION_BYTES = 192 * 1024 * 1024
MAX_PARTITION_PRODUCTS = 200_000_000
MAX_FFT_POINTS = 100_000_000


def _integer(value: int, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise AudioContractError(f"{name} must be integer in [{low}, {high}]")
    return value


@dataclass(frozen=True, slots=True)
class PartitionGeometry:
    """Preflight allocation and cumulative work limits."""

    inputs: int
    outputs: int
    taps: int
    block: int

    def __post_init__(self) -> None:
        _integer(self.inputs, "inputs", 1, MAX_CHANNELS)
        _integer(self.outputs, "outputs", 1, MAX_CHANNELS)
        _integer(self.taps, "taps", 1, MAX_TAPS)
        _integer(self.block, "block", MIN_BLOCK, MAX_BLOCK)
        if self.block & (self.block - 1):
            raise AudioContractError("Partition block must be a power of two")
        if self.work_bytes > MAX_PARTITION_BYTES:
            raise AudioLimitError("Partitioned convolution exceeds memory limit")

    @property
    def partitions(self) -> int:
        return (self.taps + self.block - 1) // self.block

    @property
    def fft_size(self) -> int:
        return 2 * self.block

    @property
    def bins(self) -> int:
        return self.block + 1

    @property
    def work_bytes(self) -> int:
        # Include original immutable kernel, filter bank, input history,
        # FFT temporaries, reduction scratch, pending input and overlap.
        routes = self.inputs * self.outputs
        kernel = routes * self.taps * 8
        spectral = (
            (routes + self.inputs) * self.partitions
            + 3 * self.inputs + 4 * self.outputs + 4
        ) * self.bins * 16
        time = (3 * self.inputs + 4 * self.outputs + 4) * self.fft_size * 8
        return kernel + spectral + time

    def blocks_for(self, frames: int, include_tail: bool) -> int:
        if type(frames) is not int or frames <= 0:
            raise AudioContractError("Expected positive input frame count")
        if type(include_tail) is not bool:
            raise AudioContractError("include_tail must be boolean")
        length = frames + (self.taps - 1 if include_tail else 0)
        return (length + self.block - 1) // self.block

    def validate_work(self, frames: int, *, include_tail: bool) -> None:
        blocks = self.blocks_for(frames, include_tail)
        products = blocks * self.partitions * self.inputs * self.outputs * self.bins
        fft_points = (
            self.partitions * self.inputs * self.outputs
            + blocks * (self.inputs + self.outputs)
        ) * self.fft_size
        if products > MAX_PARTITION_PRODUCTS or fft_points > MAX_FFT_POINTS:
            raise AudioLimitError("Partitioned convolution exceeds cumulative work limit")


class PartitionedConvolver:
    """One-shot uniform partitioned overlap-add stream.

    Each input block contributes to the present and future output blocks.
    The ring retains only the input spectra needed by the finite FIR.
    push returns complete blocks; finish returns exactly the remaining
    input frames, followed by the optional causal tail.
    """

    def __init__(self, impulse: ImpulseMatrix):
        if not isinstance(impulse, ImpulseMatrix):
            raise AudioContractError("Expected ImpulseMatrix")
        outputs, inputs, taps = impulse.coefficients.shape
        self._g = g = PartitionGeometry(inputs, outputs, taps, impulse.block_frames)
        self._rate = impulse.sample_rate
        # Bound initialization itself, even if the stream is never fed.
        g.validate_work(1, include_tail=False)
        self._filters = np.empty(
            (g.partitions, g.outputs, g.inputs, g.bins), dtype=np.complex128,
        )
        for part in range(g.partitions):
            lo = part * g.block
            hi = min(lo + g.block, g.taps)
            self._filters[part] = np.fft.rfft(
                impulse.coefficients[:, :, lo:hi], n=g.fft_size, axis=-1,
            )
        self._history = np.zeros(
            (g.partitions, g.inputs, g.bins), dtype=np.complex128,
        )
        self._pending = np.zeros((g.inputs, g.block), dtype=np.float64)
        self._overlap = np.zeros((g.outputs, g.block), dtype=np.float64)
        self._next = 0
        self._processed = 0
        self._pending_count = 0
        self._input_frames = 0
        self._emitted_frames = 0
        self._finished = False

    @property
    def geometry(self) -> PartitionGeometry:
        return self._g

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
            raise AudioContractError("Partitioned convolution stream is finished")

    def _block(self, source: np.ndarray) -> np.ndarray:
        g = self._g
        slot = self._next
        self._history[slot] = np.fft.rfft(source, n=g.fft_size, axis=-1)
        spectrum = np.zeros((g.outputs, g.bins), dtype=np.complex128)
        # Ignore zero prehistory. Reduction order is fixed by partition
        # age then source index; caller push boundaries cannot affect it.
        for age in range(min(self._processed + 1, g.partitions)):
            history = self._history[(slot - age) % g.partitions]
            for receiver in range(g.outputs):
                for channel in range(g.inputs):
                    spectrum[receiver] += (
                        self._filters[age, receiver, channel] * history[channel]
                    )
        time = np.fft.irfft(spectrum, n=g.fft_size, axis=-1)
        time[:, :g.block] += self._overlap
        self._overlap[:] = time[:, g.block:]
        self._next = (slot + 1) % g.partitions
        self._processed += 1
        return time[:, :g.block]

    def push(self, samples: np.ndarray) -> np.ndarray:
        self._check_open()
        check_audio(samples)
        g = self._g
        x = samples.reshape(1, -1) if samples.ndim == 1 else samples
        if x.shape[0] != g.inputs:
            raise AudioContractError("Partitioned convolution source channel mismatch")
        if float(np.max(np.abs(x))) > MAX_ABS_SAMPLE:
            raise AudioContractError("Partitioned convolution source exceeds amplitude bound")
        frames = x.shape[1]
        total = self._input_frames + frames
        if total > int(MAX_SECONDS * self._rate):
            raise AudioLimitError("Partitioned convolution duration exceeds limit")
        g.validate_work(total, include_tail=False)
        complete = (self._pending_count + frames) // g.block * g.block
        if complete * g.outputs * 8 > MAX_BYTES:
            raise AudioLimitError("Partitioned convolution push exceeds output budget")
        result = np.empty((g.outputs, complete), dtype=np.float64)
        offset = emitted = 0
        while offset < frames:
            take = min(g.block - self._pending_count, frames - offset)
            end = self._pending_count + take
            self._pending[:, self._pending_count:end] = x[:, offset:offset + take]
            self._pending_count = end
            offset += take
            if end == g.block:
                result[:, emitted:emitted + g.block] = self._block(self._pending)
                emitted += g.block
                self._pending_count = 0
        self._input_frames = total
        self._emitted_frames += emitted
        return result

    def finish(self, *, include_tail: bool = True) -> np.ndarray:
        self._check_open()
        if type(include_tail) is not bool:
            raise AudioContractError("include_tail must be boolean")
        if not self._input_frames:
            raise AudioContractError("Cannot finish an empty stream")
        g = self._g
        g.validate_work(self._input_frames, include_tail=include_tail)
        remaining = self._pending_count + (g.taps - 1 if include_tail else 0)
        if remaining * g.outputs * 8 > MAX_BYTES:
            raise AudioLimitError("Partitioned convolution finish exceeds output budget")
        result = np.empty((g.outputs, remaining), dtype=np.float64)
        if remaining:
            self._pending[:, self._pending_count:] = 0
            offset = 0
            while offset < remaining:
                block = self._block(self._pending)
                take = min(g.block, remaining - offset)
                result[:, offset:offset + take] = block[:, :take]
                self._pending.fill(0)
                offset += take
        self._pending_count = 0
        self._emitted_frames += remaining
        self._finished = True
        return result


def convolve_partitioned(
    audio: AudioBuffer,
    impulse: ImpulseMatrix,
    *,
    include_tail: bool = True,
    output_layout: AudioLayout | None = None,
) -> AudioBuffer:
    """Offline bounded partitioned FIR with the existing audio contracts."""
    if not isinstance(audio, AudioBuffer) or not isinstance(impulse, ImpulseMatrix):
        raise AudioContractError("Expected AudioBuffer and ImpulseMatrix")
    if type(include_tail) is not bool:
        raise AudioContractError("include_tail must be boolean")
    outputs, inputs, taps = impulse.coefficients.shape
    if audio.channels != inputs or audio.sample_rate != impulse.sample_rate:
        raise AudioContractError("Source and impulse clocks or channels disagree")
    layout = output_layout if output_layout is not None else (
        audio.spec.layout if inputs == outputs else AudioLayout.for_channels(outputs)
    )
    if not isinstance(layout, AudioLayout) or layout.channels != outputs:
        raise AudioContractError("Partitioned convolution output layout mismatch")
    frames = audio.frames + (taps - 1 if include_tail else 0)
    AudioSpec(audio.sample_rate, layout, frames)
    g = PartitionGeometry(inputs, outputs, taps, impulse.block_frames)
    g.validate_work(audio.frames, include_tail=include_tail)
    engine = PartitionedConvolver(impulse)
    result = np.empty((outputs, frames), dtype=np.float32)
    offset = 0
    for start in range(0, audio.frames, g.block):
        chunk = engine.push(audio.samples[:, start:start + g.block])
        result[:, offset:offset + chunk.shape[1]] = chunk
        offset += chunk.shape[1]
    tail = engine.finish(include_tail=include_tail)
    result[:, offset:] = tail
    return make_buffer(
        result, audio.sample_rate, layout=layout,
        source_note="causal partitioned matrix convolution",
    )
