"""Fixed-clock streaming STFT and overlap-add reconstruction.

Both streaming directions preserve the exact Hann/FFT frame geometry of
the existing offline stft/istft implementation. The sample clock is
independent of caller push boundaries. This is a bounded CPU reference,
not an allocation-free real-time audio callback API.
"""
from __future__ import annotations

import math

import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, MAX_BYTES, MAX_CHANNELS,
    MAX_SECONDS, check_audio, check_rate, make_buffer,
)
from .errors import AudioContractError, AudioLimitError
from .stft import StftConfig, Spectrogram
from .windows import window_bank

MAX_STREAM_SPECTRAL_PRODUCTS = 240_000_000
MAX_STREAM_SAMPLES = 12_000_000


def _count_frames(raw_frames: int, cfg: StftConfig) -> int:
    """Match offline _padded_signal's FFT geometry exactly."""
    if type(raw_frames) is not int or raw_frames < 1:
        raise AudioContractError("STFT source frame count must be positive")
    extra = cfg.n_fft if cfg.center else 0
    total = raw_frames + extra
    if total < cfg.n_fft:
        if not cfg.pad_end:
            raise AudioContractError("Input shorter than FFT without padding")
        total = cfg.n_fft
    if cfg.pad_end:
        leftover = (total - cfg.n_fft) % cfg.hop_length
        if leftover:
            total += cfg.hop_length - leftover
    return 1 + (total - cfg.n_fft) // cfg.hop_length


def _geometry(channels: int, sample_rate: int, frames: int, cfg: StftConfig):
    if (type(channels) is not int or not 1 <= channels <= MAX_CHANNELS
            or not isinstance(cfg, StftConfig)):
        raise AudioContractError("Invalid streaming STFT topology")
    check_rate(sample_rate)
    AudioSpec(sample_rate, AudioLayout.for_channels(channels), frames)
    count = _count_frames(frames, cfg)
    cost = channels * count * cfg.n_fft
    if cost > MAX_STREAM_SPECTRAL_PRODUCTS:
        raise AudioLimitError("Streaming spectral transform exceeds work budget")
    if channels * count * (cfg.n_fft // 2 + 1) * 16 > MAX_BYTES:
        raise AudioLimitError("Streaming spectral output exceeds memory budget")
    return count


class StreamingStft:
    """One-shot incremental analysis producing complex channels/bins/frames."""

    def __init__(self, channels: int, sample_rate: int,
                 config: StftConfig = StftConfig()):
        if (not isinstance(config, StftConfig)
                or type(channels) is not int
                or not 1 <= channels <= MAX_CHANNELS):
            raise AudioContractError("Expected bounded STFT channels and configuration")
        check_rate(sample_rate)
        self._channels = channels
        self._rate = sample_rate
        self._cfg = config
        self._window = window_bank(config.window, config.n_fft, periodic=True)
        self._left = config.n_fft // 2 if config.center else 0
        self._buffer = np.zeros((channels, self._left), dtype=np.float64)
        self._samples = 0
        self._frames = 0
        self._finished = False

    @property
    def input_frames(self) -> int:
        return self._samples

    @property
    def output_frames(self) -> int:
        return self._frames

    @property
    def finished(self) -> bool:
        return self._finished

    def _open(self):
        if self._finished:
            raise AudioContractError("STFT analysis already finished")

    def _consume(self) -> np.ndarray:
        cfg = self._cfg
        blocks = []
        while self._buffer.shape[1] >= cfg.n_fft:
            samples = self._buffer[:, :cfg.n_fft] * self._window
            spectrum = np.fft.rfft(samples, n=cfg.n_fft, axis=1)
            blocks.append(spectrum)
            self._buffer = self._buffer[:, cfg.hop_length:]
        count = len(blocks)
        self._frames += count
        if not blocks:
            return np.empty((self._channels, cfg.n_fft // 2 + 1, 0),
                            dtype=np.complex128)
        return np.stack(blocks, axis=2)

    def push(self, samples: np.ndarray) -> np.ndarray:
        self._open()
        check_audio(samples)
        data = samples.reshape(1, -1) if samples.ndim == 1 else samples
        if data.shape[0] != self._channels:
            raise AudioContractError("Streaming STFT channel mismatch")
        total = self._samples + data.shape[1]
        if total > int(MAX_SECONDS * self._rate) or total > MAX_STREAM_SAMPLES:
            raise AudioLimitError("Streaming STFT exceeds bounded duration")
        # A valid partial input must fit even if users push the entire source
        # as one call; no state is modified before this preflight.
        _geometry(self._channels, self._rate, total, self._cfg)
        if data.nbytes + self._buffer.nbytes > MAX_BYTES:
            raise AudioLimitError("Streaming STFT input allocation exceeds budget")
        self._buffer = np.concatenate(
            (self._buffer, data.astype(np.float64)), axis=1,
        )
        self._samples = total
        return self._consume()

    def finish(self) -> np.ndarray:
        self._open()
        if not self._samples:
            raise AudioContractError("Cannot finalize an empty STFT stream")
        cfg = self._cfg
        expected = _geometry(self._channels, self._rate, self._samples, cfg)
        pad = cfg.n_fft // 2 if cfg.center else 0
        total = self._samples + 2 * pad
        if total < cfg.n_fft:
            if not cfg.pad_end:
                raise AudioContractError("Unpadded STFT source too short")
            total = cfg.n_fft
        if cfg.pad_end:
            residual = (total - cfg.n_fft) % cfg.hop_length
            if residual:
                total += cfg.hop_length - residual
        append = total - (self._samples + pad)
        if append < 0:
            raise AudioContractError("Invalid streaming STFT tail padding")
        if append:
            self._buffer = np.concatenate(
                (self._buffer,
                 np.zeros((self._channels, append), dtype=np.float64)),
                axis=1,
            )
        result = self._consume()
        if self._frames != expected:
            raise AudioContractError("Streaming STFT frame count mismatch")
        self._finished = True
        return result


class StreamingIstft:
    """Incremental inverse FFT overlap-add from a known original frame count.

    The original audio duration is required up front to trim exact center
    padding and an optional incomplete final FFT. push emits only PCM
    samples that can no longer be changed by later spectral frames.
    """

    def __init__(self, channels: int, sample_rate: int,
                 original_frames: int, config: StftConfig = StftConfig(),
                 epsilon: float = 1e-10):
        self._expected = _geometry(channels, sample_rate, original_frames, config)
        if not isinstance(config, StftConfig):
            raise AudioContractError("Expected StftConfig")
        if (type(epsilon) not in (float, int)
                or not math.isfinite(epsilon)
                or not 0 <= epsilon <= .01):
            raise AudioContractError("Invalid inverse FFT normalization epsilon")
        self._channels = channels
        self._rate = sample_rate
        self._source_frames = original_frames
        self._cfg = config
        self._eps = epsilon
        self._window = window_bank(config.window, config.n_fft, periodic=True)
        self._weights_window = self._window * self._window
        self._buffer = np.zeros((channels, config.n_fft), dtype=np.float64)
        self._weight = np.zeros(config.n_fft, dtype=np.float64)
        self._left = config.n_fft // 2 if config.center else 0
        self._global_offset = 0
        self._frames = 0
        self._emitted = 0
        self._finished = False

    @property
    def spectral_frames(self):
        return self._frames

    @property
    def output_frames(self):
        return self._emitted

    @property
    def finished(self):
        return self._finished

    def _completed(self, count: int) -> np.ndarray:
        if type(count) is not int or not 0 <= count <= self._cfg.n_fft:
            raise AudioContractError("Invalid inverse FFT completed interval")
        completed = self._buffer[:, :count].copy()
        weights = self._weight[:count]
        valid = weights > self._eps
        completed[:, valid] /= weights[valid]
        completed[:, ~valid] = 0
        beginning = self._global_offset
        self._global_offset += count
        low = max(self._left - beginning, 0)
        high = min(self._left + self._source_frames - beginning, count)
        if high <= low:
            return np.empty((self._channels, 0), dtype=np.float64)
        piece = completed[:, low:high]
        self._emitted += piece.shape[1]
        return piece

    def _advance(self):
        hop = self._cfg.hop_length
        if hop == self._cfg.n_fft:
            self._buffer.fill(0)
            self._weight.fill(0)
        else:
            self._buffer[:, :-hop] = self._buffer[:, hop:]
            self._buffer[:, -hop:] = 0
            self._weight[:-hop] = self._weight[hop:]
            self._weight[-hop:] = 0

    def push(self, spectral_frames: np.ndarray) -> np.ndarray:
        if self._finished:
            raise AudioContractError("Inverse streaming STFT already finished")
        data = spectral_frames
        if (not isinstance(data, np.ndarray) or data.ndim != 3
                or data.dtype not in (np.complex64, np.complex128)
                or data.shape[0] != self._channels
                or data.shape[1] != self._cfg.n_fft // 2 + 1
                or data.shape[2] < 1
                or not np.isfinite(data).all()):
            raise AudioContractError("Expected finite complex STFT frame batch")
        if self._frames + data.shape[2] > self._expected:
            raise AudioContractError("Inverse STFT exceeds source frame geometry")
        if data.shape[2] * self._channels * self._cfg.hop_length * 8 > MAX_BYTES:
            raise AudioLimitError("Inverse STFT push result exceeds memory budget")
        pieces = []
        for index in range(data.shape[2]):
            time = np.fft.irfft(data[:, :, index], n=self._cfg.n_fft, axis=1)
            self._buffer += time * self._window
            self._weight += self._weights_window
            pieces.append(self._completed(self._cfg.hop_length))
            self._advance()
            self._frames += 1
        return np.concatenate(pieces, axis=1)

    def finish(self) -> np.ndarray:
        if self._finished:
            raise AudioContractError("Inverse streaming STFT already finished")
        if self._frames != self._expected:
            raise AudioContractError("Inverse STFT missing required spectral frames")
        # Every frame has already emitted one hop. The final overlap tail
        # has length FFT-hop on the original padded clock.
        count = self._cfg.n_fft - self._cfg.hop_length
        remainder = self._completed(count)
        missing = self._source_frames - self._emitted
        if missing < 0:
            raise AudioContractError("Inverse STFT emitted too many samples")
        if missing:
            if not self._cfg.pad_end:
                raise AudioContractError("Unpadded STFT cannot recover missing samples")
            remainder = np.pad(remainder, ((0, 0), (0, missing)))
            self._emitted += missing
        self._finished = True
        return remainder


def streaming_stft(buffer: AudioBuffer, config: StftConfig = StftConfig(),
                   *, push_frames: int = 8192) -> Spectrogram:
    """Offline parity adapter that exercises fixed-clock incremental analysis."""
    if not isinstance(buffer, AudioBuffer) or not isinstance(config, StftConfig):
        raise AudioContractError("Expected AudioBuffer and StftConfig")
    if type(push_frames) is not int or not 1 <= push_frames <= 65536:
        raise AudioContractError("Invalid streaming push frame size")
    stream = StreamingStft(buffer.channels, buffer.sample_rate, config)
    blocks = [
        stream.push(buffer.samples[:, start:start + push_frames])
        for start in range(0, buffer.frames, push_frames)
    ]
    blocks.append(stream.finish())
    spectrum = np.concatenate(blocks, axis=2)
    return Spectrogram(spectrum, buffer.sample_rate, buffer.frames, config)


def streaming_istft(spectrum: Spectrogram, *, push_frames: int = 32) -> AudioBuffer:
    """Offline parity adapter using chunked streaming overlap-add reconstruction."""
    if not isinstance(spectrum, Spectrogram):
        raise AudioContractError("Expected Spectrogram")
    if type(push_frames) is not int or not 1 <= push_frames <= 4096:
        raise AudioContractError("Invalid spectral push frame size")
    engine = StreamingIstft(
        spectrum.channels, spectrum.sample_rate,
        spectrum.original_frames, spectrum.config,
    )
    outputs = [
        engine.push(spectrum.spectrum[:, :, i:i + push_frames])
        for i in range(0, spectrum.frames, push_frames)
    ]
    outputs.append(engine.finish())
    complete = np.concatenate(outputs, axis=1)
    if not np.isfinite(complete).all():
        raise AudioContractError("Streaming ISTFT produced nonfinite PCM")
    if float(np.max(np.abs(complete))) > float(np.finfo(np.float32).max):
        raise AudioLimitError("Streaming ISTFT exceeds finite float32 range")
    return make_buffer(
        complete.astype(np.float32), spectrum.sample_rate,
        source_note="streaming overlap-add inverse STFT",
    )
