"""Frame-aligned PCM chunking for deterministic local DSP pipelines.

Unlike an arbitrary byte-stream API, all packets have an explicit sample
rate, channel layout, absolute source offset and finite float32 frames.
Overlap-frame extraction retains only bounded history; it never silently
resamples, opens files, downloads models or accesses devices. The processor
is synchronous, NOT a low-latency real-time scheduler or multithreaded queue.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, check_rate, as_channel_first,
    make_buffer, MAX_BYTES,
)
from .errors import AudioContractError, AudioLimitError


@dataclass(frozen=True, slots=True, eq=False)
class FramePacket:
    """One immutable consecutive block with absolute timeline position."""

    sample_rate: int
    start_frame: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if type(self.start_frame) is not int or self.start_frame < 0:
            raise AudioContractError("Frame packet offset cannot be negative")
        data = as_channel_first(self.samples, copy=True)
        if data.shape[1] > 1_048_576:
            raise AudioLimitError("Streaming frame packet exceeds 1M frame cap")
        data.setflags(write=False)
        object.__setattr__(self, "samples", data)

    @property
    def stop_frame(self) -> int:
        return self.start_frame + self.frames

    @property
    def frames(self) -> int:
        return self.samples.shape[1]

    @property
    def channels(self) -> int:
        return self.samples.shape[0]

    def as_buffer(self) -> AudioBuffer:
        return make_buffer(self.samples, self.sample_rate,
                           source_note="local streamed PCM packet")

    def sliced(self, start: int, stop: int) -> "FramePacket":
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start < stop <= self.frames):
            raise AudioContractError("Packet slice outside source block")
        return FramePacket(
            self.sample_rate, self.start_frame + start,
            self.samples[:, start:stop],
        )


def split_packets(audio: AudioBuffer, *, packet_frames: int = 4096
                  ) -> tuple[FramePacket, ...]:
    """Create exact non-overlapping packet partitions of immutable source."""
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected canonical AudioBuffer")
    if type(packet_frames) is not int or not 1 <= packet_frames <= 1_048_576:
        raise AudioContractError("Invalid bounded stream packet size")
    if (audio.frames + packet_frames - 1) // packet_frames > 50_000:
        raise AudioLimitError("Too many queued PCM packets")
    return tuple(
        FramePacket(audio.sample_rate, start,
                    audio.samples[:, start:min(audio.frames, start + packet_frames)])
        for start in range(0, audio.frames, packet_frames)
    )


class PacketCollector:
    """Strictly sequential packet assembly with a bounded total output."""

    def __init__(self, *, sample_rate: int, layout: AudioLayout,
                 maximum_frames: int):
        check_rate(sample_rate)
        if not isinstance(layout, AudioLayout):
            raise AudioContractError("PacketCollector needs AudioLayout")
        AudioSpec(sample_rate, layout, maximum_frames)
        self.sample_rate = sample_rate
        self.layout = layout
        self.maximum_frames = maximum_frames
        self._packets: list[FramePacket] = []
        self._position = 0
        self._closed = False

    @property
    def frames_seen(self) -> int:
        return self._position

    @property
    def finished(self) -> bool:
        return self._closed

    def push(self, packet: FramePacket) -> None:
        if self._closed:
            raise AudioContractError("Cannot append PCM after finalization")
        if not isinstance(packet, FramePacket):
            raise AudioContractError("Expected validated FramePacket")
        if packet.sample_rate != self.sample_rate or packet.channels != self.layout.channels:
            raise AudioContractError("Streaming source clock or channels changed")
        if packet.start_frame != self._position:
            raise AudioContractError("Streaming packet gap, overlap or reordering")
        if packet.stop_frame > self.maximum_frames:
            raise AudioLimitError("Stream exceeds declared bounded duration")
        self._packets.append(packet)
        self._position = packet.stop_frame

    def finish(self) -> AudioBuffer:
        if self._closed or self._position <= 0:
            raise AudioContractError("Stream cannot be finalized twice or empty")
        self._closed = True
        return make_buffer(
            np.concatenate([p.samples for p in self._packets], axis=1),
            self.sample_rate, layout=self.layout,
            source_note="reassembled fixed-clock PCM stream",
        )


@dataclass(frozen=True, slots=True)
class FramerConfig:
    window_frames: int = 1024
    hop_frames: int = 256
    pad_last: bool = False

    def __post_init__(self):
        if type(self.window_frames) is not int or not 16 <= self.window_frames <= 65536:
            raise AudioContractError("Streaming window length outside 16..65536")
        if (type(self.hop_frames) is not int
                or not 1 <= self.hop_frames <= self.window_frames):
            raise AudioContractError("Streaming hop must be inside analysis window")
        if type(self.pad_last) is not bool:
            raise AudioContractError("pad_last must be Boolean")


@dataclass(frozen=True, slots=True, eq=False)
class WindowPacket:
    start_frame: int
    original_frames: int
    samples: np.ndarray

    def __post_init__(self):
        if (type(self.start_frame) is not int or self.start_frame < 0
                or type(self.original_frames) is not int
                or self.original_frames <= 0
                or not isinstance(self.samples, np.ndarray)
                or self.samples.ndim != 2
                or not 1 <= self.original_frames <= self.samples.shape[1]):
            raise AudioContractError("Invalid audio analysis window")
        safe = as_channel_first(self.samples, copy=True)
        safe.setflags(write=False)
        object.__setattr__(self, "samples", safe)


class StreamingFramer:
    """Deterministic fixed-hop window extraction across arbitrary chunk sizes.

    On each push, append packet bytes to bounded history, emit all full
    window spans, then discard frames before the next window start. Final
    optional incomplete window is zero-padded to the analysis window size.
    """

    def __init__(self, *, sample_rate: int, channels: int,
                 config: FramerConfig = FramerConfig()):
        check_rate(sample_rate)
        if type(channels) is not int or not 1 <= channels <= 8:
            raise AudioContractError("Invalid framer channel count")
        if not isinstance(config, FramerConfig):
            raise AudioContractError("Expected FramerConfig")
        self.sample_rate = sample_rate
        self.channels = channels
        self.config = config
        self.received = 0
        self._start = 0
        self._buffer_start = 0
        self._buffer = np.empty((channels, 0), dtype=np.float32)
        self._closed = False

    def push(self, packet: FramePacket) -> tuple[WindowPacket, ...]:
        if self._closed or not isinstance(packet, FramePacket):
            raise AudioContractError("StreamingFramer is closed or packet invalid")
        if (packet.sample_rate != self.sample_rate or
                packet.channels != self.channels or
                packet.start_frame != self.received):
            raise AudioContractError("StreamingFramer needs contiguous clock-matched packets")
        if self._buffer.shape[1] + packet.frames > (
            1_048_576 + self.config.window_frames
        ):
            raise AudioLimitError("Streaming framer packet/history exceeded budget")
        self._buffer = np.concatenate((self._buffer, packet.samples), axis=1)
        self.received = packet.stop_frame
        results = []
        while self._start + self.config.window_frames <= self.received:
            local = self._start - self._buffer_start
            block = self._buffer[:, local:local + self.config.window_frames]
            results.append(WindowPacket(self._start, self.config.window_frames, block))
            self._start += self.config.hop_frames
        self._discard_history()
        return tuple(results)

    def _discard_history(self):
        """Keep only frames needed for the next analysis window."""
        removable = min(
            self._start - self._buffer_start, self._buffer.shape[1]
        )
        if removable > 0:
            self._buffer = self._buffer[:, removable:].copy()
            self._buffer_start += removable

    def finish(self) -> tuple[WindowPacket, ...]:
        if self._closed:
            raise AudioContractError("StreamingFramer already finalized")
        self._closed = True
        if not self.config.pad_last or self._start >= self.received:
            return ()
        remaining = self.received - self._start
        if remaining <= 0:
            return ()
        local = self._start - self._buffer_start
        source = self._buffer[:, local:local + remaining]
        if source.shape[1] != remaining:
            raise AudioContractError("Lost streaming PCM history before final window")
        window = np.pad(source, ((0, 0),
                                 (0, self.config.window_frames - remaining)))
        return (WindowPacket(self._start, remaining, window),)


def offline_window_reference(audio: AudioBuffer, config: FramerConfig
                             ) -> tuple[WindowPacket, ...]:
    """Simple independent oracle for verifying chunk-boundary invariance."""
    if not isinstance(audio, AudioBuffer) or not isinstance(config, FramerConfig):
        raise AudioContractError("Expected canonical buffer and FramerConfig")
    frames = []
    for start in range(0, audio.frames, config.hop_frames):
        stop = start + config.window_frames
        if stop > audio.frames:
            if not config.pad_last:
                break
            present = audio.frames - start
            data = np.pad(audio.samples[:, start:],
                          ((0, 0), (0, config.window_frames - present)))
            frames.append(WindowPacket(start, present, data))
            break
        frames.append(WindowPacket(start, config.window_frames,
                                   audio.samples[:, start:stop]))
    return tuple(frames)


def frame_stream_report(packets: tuple[FramePacket, ...],
                        windows: tuple[WindowPacket, ...]) -> dict:
    if (not isinstance(packets, tuple) or not isinstance(windows, tuple)
            or not packets or any(not isinstance(x, FramePacket) for x in packets)
            or any(not isinstance(x, WindowPacket) for x in windows)):
        raise AudioContractError("Expected verified stream/analysis packet tuples")
    return {
        "sample_rate": packets[0].sample_rate,
        "channels": packets[0].channels,
        "stream_packets": len(packets),
        "window_packets": len(windows),
        "source_frames": sum(x.frames for x in packets),
        "last_packet_stop": packets[-1].stop_frame,
        "model_inference": False,
        "network_streaming_implemented": False,
        "real_time_latency_guaranteed": False,
    }
