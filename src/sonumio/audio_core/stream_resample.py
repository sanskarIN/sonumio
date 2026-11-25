"""Packet-invariant offline sinc resampling with bounded retained PCM.

This processor requires a declared source length and contiguous immutable
packets. A centered filter needs future samples; push may return no output.
finish supplies the right-edge truncation only after the source is complete.
No implicit padding, realtime scheduling, model inference or device access.
"""
from __future__ import annotations

import math

import numpy as np

from .contracts import MAX_CHANNELS, check_rate
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .resample import (
    ResampleSettings, output_frames, _window, _finite_float32, _MAX_KERNEL_CELLS,
)
from .streaming import FramePacket
from .wave_stream import MAX_STREAM_SECONDS, MAX_STREAM_PACKET_FRAMES

# Bound both the newly accepted source data and materialized output per call.
# At the supported 24:1 maximum upsampling ratio, 8192 source frames plus
# filter lookahead produce at most about 200k output frames.
MAX_RESAMPLE_INPUT_FRAMES = 8192


class StreamingSincResampler:
    """Convert consecutive source packets to a declared destination clock.

    Output frame coordinates are absolute and use the same sample origin,
    rounding, window and edge normalization as the offline sinc resampler.
    Retained history is bounded by the kernel support; it is independent
    of the recording length. Returned packets are at most 65536 frames.
    Calls are synchronous and must be serialized by the caller.
    """

    def __init__(
        self, source_rate: int, channels: int, expected_frames: int,
        settings: ResampleSettings,
    ) -> None:
        check_rate(source_rate)
        if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
            raise AudioContractError("Invalid resampling channel count")
        if (type(expected_frames) is not int
                or not 1 <= expected_frames <= source_rate * MAX_STREAM_SECONDS):
            raise AudioLimitError("Invalid declared resampling source length")
        if not isinstance(settings, ResampleSettings) or settings.method != "sinc":
            raise AudioContractError("Streaming conversion requires sinc settings")
        self.source_rate = source_rate
        self.target_rate = settings.target_rate
        self.channels = channels
        self.expected_frames = expected_frames
        self.expected_output_frames = output_frames(
            expected_frames, source_rate, settings.target_rate,
        )
        self.settings = settings
        self._identity = source_rate == settings.target_rate
        self._cutoff = min(1.0, settings.target_rate / source_rate)
        self._radius = settings.lobes / self._cutoff
        self._width = math.ceil(self._radius)
        self._taps = np.arange(-self._width + 1, self._width + 1, dtype=np.int64)
        self._step = source_rate / settings.target_rate
        self._batch = min(settings.chunk_frames,
                          max(1, _MAX_KERNEL_CELLS // self._taps.size))
        self._history = np.empty((channels, 0), dtype=np.float32)
        self._history_start = 0
        self._received = 0
        self._emitted = 0
        self._finished = False

    @property
    def received_frames(self) -> int:
        return self._received

    @property
    def emitted_frames(self) -> int:
        return self._emitted

    @property
    def retained_frames(self) -> int:
        return self._history.shape[1]

    @property
    def finished(self) -> bool:
        return self._finished

    def _center(self, index: int) -> int:
        return math.floor(index * self._step)

    def _ready_stop(self, received: int) -> int:
        """Exclusive output index whose full nominal support has arrived.

        A monotone binary search avoids rounding-dependent off-by-one
        calculations at fractional clock positions.
        """
        low, high = self._emitted, self.expected_output_frames
        while low < high:
            mid = (low + high) // 2
            if self._center(mid) + self._width < received:
                low = mid + 1
            else:
                high = mid
        return low

    def _render(
        self, data: np.ndarray, origin: int, start: int, stop: int,
    ) -> tuple[FramePacket, ...]:
        packets = []
        for out_start in range(start, stop, MAX_STREAM_PACKET_FRAMES):
            out_stop = min(stop, out_start + MAX_STREAM_PACKET_FRAMES)
            output = np.empty((self.channels, out_stop - out_start), dtype=np.float32)
            for first in range(out_start, out_stop, self._batch):
                last = min(out_stop, first + self._batch)
                pos = np.arange(first, last, dtype=np.float64) * self._step
                centers = np.floor(pos).astype(np.int64)
                indices = centers[:, None] + self._taps[None, :]
                delta = pos[:, None] - indices
                kernel = self._cutoff * np.sinc(self._cutoff * delta) * _window(
                    delta, self._radius, self.settings.window, float(self.settings.beta),
                )
                valid = (indices >= 0) & (indices < self.expected_frames)
                local = indices - origin
                if np.any(valid & ((local < 0) | (local >= data.shape[1]))):
                    raise AudioIntegrityError("Required resampling context unavailable")
                kernel = np.where(valid, kernel, 0.0)
                weight = kernel.sum(axis=1)
                if np.any(np.abs(weight) <= 1e-12):
                    raise AudioContractError("Sinc interpolation kernel underflow")
                kernel /= weight[:, None]
                safe = np.clip(local, 0, data.shape[1] - 1)
                for ch in range(self.channels):
                    output[ch, first - out_start:last - out_start] = _finite_float32(
                        np.sum(data[ch, safe] * kernel, axis=1, dtype=np.float64),
                    )
            packets.append(FramePacket(self.target_rate, out_start, output))
        return tuple(packets)

    def push(self, packet: FramePacket) -> tuple[FramePacket, ...]:
        """Accept one packet; validation/render failure leaves state unchanged."""
        if self._finished:
            raise AudioContractError("Cannot resample after finalization")
        if not isinstance(packet, FramePacket) or (
            packet.sample_rate != self.source_rate
            or packet.channels != self.channels
            or packet.start_frame != self._received
        ):
            raise AudioContractError("Resampling requires consecutive clock-matched packets")
        if packet.frames > MAX_RESAMPLE_INPUT_FRAMES:
            raise AudioLimitError("Resampling packet exceeds 8192-frame input cap")
        if packet.stop_frame > self.expected_frames:
            raise AudioLimitError("Resampling input exceeds declared frame count")
        if self._identity:
            self._received = packet.stop_frame
            self._emitted = packet.stop_frame
            return (packet,)
        origin = self._history_start
        data = np.concatenate((self._history, packet.samples), axis=1)
        received = packet.stop_frame
        stop = self._ready_stop(received)
        output = self._render(data, origin, self._emitted, stop)
        # No future output can need an earlier nominal tap. Copy the
        # surviving tail so a small view never pins a large source packet.
        keep_from = (received if stop == self.expected_output_frames else
                     max(origin, min(received, self._center(stop) - self._width + 1)))
        history = data[:, keep_from - origin:].copy()
        self._history = history
        self._history_start = keep_from
        self._received = received
        self._emitted = stop
        return output

    def finish(self) -> tuple[FramePacket, ...]:
        """Emit the final right-edge samples, only for the declared length."""
        if self._finished:
            raise AudioContractError("Resampling already finalized")
        if self._received != self.expected_frames:
            raise AudioIntegrityError("Incomplete resampling source stream")
        output = self._render(
            self._history, self._history_start,
            self._emitted, self.expected_output_frames,
        )
        self._history = np.empty((self.channels, 0), dtype=np.float32)
        self._history_start = self._received
        self._emitted = self.expected_output_frames
        self._finished = True
        return output
