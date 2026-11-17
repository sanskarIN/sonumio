"""Strict bounded RIFF/WAVE PCM transport, with no implicit file writes.

PCM WAV is an interchange container, not proof of recording rights. Only
little-endian RIFF/WAVE with uncompressed integer PCM is supported. Float
WAV, WAVE64, RF64, RIFX, extensible compressed formats and enormous chunks
are intentionally rejected. All sample arrays are channel-first float32.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, check_rate, make_buffer,
    MAX_BYTES, MAX_CHANNELS, MAX_SECONDS, validate_allocation,
)
from .dither import (
    QuantizationSettings, encode_pcm_bytes, unpack_pcm, dequantize_pcm,
)
from .errors import AudioContractError, AudioFormatError, AudioLimitError

MAX_WAV_BYTES = min(MAX_BYTES, 96 * 1024 * 1024)
MAX_CHUNKS = 4096


@dataclass(frozen=True, slots=True)
class WavInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    frames: int
    bytes_per_frame: int
    data_bytes: int
    ignored_chunks: tuple[str, ...] = ()

    def __post_init__(self):
        check_rate(self.sample_rate)
        if type(self.channels) is not int or not 1 <= self.channels <= MAX_CHANNELS:
            raise AudioFormatError("Invalid WAVE channel count")
        if self.bits_per_sample not in (8, 16, 24, 32):
            raise AudioFormatError("Unsupported integer WAVE bit depth")
        if type(self.frames) is not int or self.frames <= 0:
            raise AudioFormatError("Expected nonempty WAVE payload")
        if self.frames > self.sample_rate * MAX_SECONDS:
            raise AudioLimitError("WAVE duration exceeds local memory limit")
        if self.bytes_per_frame != self.channels * self.bits_per_sample // 8:
            raise AudioFormatError("Incorrect WAVE block alignment")
        if self.data_bytes != self.frames * self.bytes_per_frame:
            raise AudioFormatError("Incorrect WAVE data chunk length")


def _fourcc(value: bytes) -> str:
    if len(value) != 4 or any(v < 32 or v > 126 for v in value):
        raise AudioFormatError("Invalid nonprinting RIFF chunk name")
    return value.decode("ascii")


def _read_fmt(payload: bytes) -> tuple[int, int, int, int]:
    if len(payload) < 16 or len(payload) > 1024:
        raise AudioFormatError("Invalid WAVE fmt chunk length")
    fmt, channels, sample_rate, byte_rate, align, bits = struct.unpack(
        "<HHIIHH", payload[:16]
    )
    if fmt != 1:
        raise AudioFormatError("Only unsigned8/signed16/24/32 integer PCM supported")
    check_rate(sample_rate)
    if not 1 <= channels <= MAX_CHANNELS:
        raise AudioFormatError("Invalid WAVE channels")
    if bits not in (8, 16, 24, 32):
        raise AudioFormatError("Unsupported WAVE bit depth")
    if align != channels * bits // 8 or byte_rate != sample_rate * align:
        raise AudioFormatError("WAVE fmt rate or block alignment is inconsistent")
    if len(payload) >= 18:
        (extension_bytes,) = struct.unpack("<H", payload[16:18])
        if extension_bytes > len(payload) - 18:
            raise AudioFormatError("Truncated WAVE extension")
        if extension_bytes > 0:
            raise AudioFormatError("PCM WAVE extensions are not supported")
    elif len(payload) not in (16,):
        raise AudioFormatError("Incomplete WAVE extension length")
    return channels, sample_rate, bits, align


def inspect_wav_bytes(encoded: bytes) -> tuple[WavInfo, memoryview]:
    """Validate entire bounded RIFF framing before allocating PCM arrays.

    Unknown chunk payloads are ignored but their boundaries are verified.
    Duplicate fmt/data chunks, suspicious header lengths and unsupported
    formats are rejected rather than relying on a forgiving wave parser.
    """
    if not isinstance(encoded, bytes) or len(encoded) < 44:
        raise AudioFormatError("Expected a complete nonempty RIFF/WAVE byte buffer")
    if len(encoded) > MAX_WAV_BYTES:
        raise AudioLimitError("WAV exceeds maximum transport byte budget")
    if encoded[:4] != b"RIFF" or encoded[8:12] != b"WAVE":
        raise AudioFormatError("Expected little-endian RIFF/WAVE PCM")
    (declared,) = struct.unpack_from("<I", encoded, 4)
    if declared != len(encoded) - 8:
        raise AudioFormatError("RIFF header size does not match complete payload")
    offset, chunks = 12, 0
    fmt = None
    data_range = None
    ignored = []
    while offset < len(encoded):
        if offset + 8 > len(encoded):
            raise AudioFormatError("Truncated RIFF chunk header")
        chunks += 1
        if chunks > MAX_CHUNKS:
            raise AudioLimitError("Too many RIFF chunks")
        kind = encoded[offset:offset + 4]
        size = struct.unpack_from("<I", encoded, offset + 4)[0]
        begin = offset + 8
        end = begin + size
        if end > len(encoded):
            raise AudioFormatError("Truncated RIFF chunk payload")
        name = _fourcc(kind)
        if kind == b"fmt ":
            if fmt is not None:
                raise AudioFormatError("Duplicate WAVE fmt chunk")
            fmt = _read_fmt(encoded[begin:end])
        elif kind == b"data":
            if data_range is not None:
                raise AudioFormatError("Duplicate WAVE data chunk")
            data_range = (begin, end)
        else:
            if len(ignored) < 64:
                ignored.append(name)
        offset = end + (size & 1)
        if offset > len(encoded):
            raise AudioFormatError("Missing required RIFF chunk padding")
    if offset != len(encoded) or fmt is None or data_range is None:
        raise AudioFormatError("Missing fmt/data or invalid RIFF chunk boundaries")
    channels, rate, bits, align = fmt
    begin, end = data_range
    if (end - begin) % align or end == begin:
        raise AudioFormatError("WAVE data payload is not frame aligned")
    frames = (end - begin) // align
    validate_allocation(channels, frames)
    info = WavInfo(rate, channels, bits, frames, align, end - begin, tuple(ignored))
    return info, memoryview(encoded)[begin:end]


def decode_wav_bytes(encoded: bytes, *,
                     source_note: str = "decoded PCM WAV") -> AudioBuffer:
    info, data = inspect_wav_bytes(encoded)
    ints = unpack_pcm(data.tobytes(), bits=info.bits_per_sample,
                      channels=info.channels)
    signal = dequantize_pcm(ints, bits=info.bits_per_sample)
    layout = AudioLayout.for_channels(info.channels)
    return make_buffer(signal, info.sample_rate, layout=layout,
                       source_note=source_note)


def _riff_chunk(kind: bytes, content: bytes) -> bytes:
    if not isinstance(kind, bytes) or len(kind) != 4:
        raise AudioContractError("Invalid four-byte RIFF identifier")
    if len(content) >= (1 << 32):
        raise AudioLimitError("RIFF chunk size overflow")
    return kind + struct.pack("<I", len(content)) + content + (
        b"\x00" if len(content) & 1 else b""
    )


def encode_wav_bytes(buffer: AudioBuffer, *,
                     settings: QuantizationSettings | None = None) -> bytes:
    if not isinstance(buffer, AudioBuffer):
        raise AudioContractError("Expected canonical AudioBuffer")
    if settings is None:
        settings = QuantizationSettings()
    if not isinstance(settings, QuantizationSettings):
        raise AudioContractError("Invalid WAVE quantization settings")
    channels = buffer.channels
    rate = buffer.sample_rate
    bits = settings.bits
    align = channels * bits // 8
    payload = encode_pcm_bytes(buffer.samples, settings=settings)
    if len(payload) != buffer.frames * align:
        raise AudioFormatError("Encoder produced misaligned PCM payload")
    fmt = struct.pack("<HHIIHH", 1, channels, rate,
                      rate * align, align, bits)
    raw = b"WAVE" + _riff_chunk(b"fmt ", fmt) + _riff_chunk(b"data", payload)
    if len(raw) + 8 > MAX_WAV_BYTES:
        raise AudioLimitError("Rendered WAVE output exceeds bounded transport")
    return b"RIFF" + struct.pack("<I", len(raw)) + raw


def read_wav_file(path: str | Path) -> AudioBuffer:
    file = Path(path)
    if not file.is_file() or file.is_symlink():
        raise AudioFormatError("Expected ordinary local WAV file")
    if file.stat().st_size > MAX_WAV_BYTES:
        raise AudioLimitError("WAV file exceeds configured size cap")
    payload = file.read_bytes()
    if len(payload) > MAX_WAV_BYTES:
        raise AudioLimitError("WAV changed during bounded snapshot read")
    return decode_wav_bytes(payload, source_note=file.name)


def write_wav_file(buffer: AudioBuffer, path: str | Path, *,
                   settings: QuantizationSettings | None = None,
                   overwrite: bool = False,
                   create_parent: bool = False) -> Path:
    """Atomic local file output. Overwrites require an explicit opt-in.

    Security boundary: this is a developer LOCAL utility. Do not pass
    arbitrary untrusted API output paths; callers must choose safe roots.
    """
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("File-writing options must be booleans")
    dest = Path(path)
    if dest.suffix.lower() != ".wav" or dest.name in ("", ".", ".."):
        raise AudioFormatError("Local audio destination must end in .wav")
    parent = dest.parent
    if create_parent:
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or dest.is_symlink():
        raise AudioFormatError("Invalid or symlink WAV output destination")
    if dest.exists() and not overwrite:
        raise AudioFormatError("Existing WAV requires explicit overwrite")
    data = encode_wav_bytes(buffer, settings=settings)
    scratch = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent, prefix=".sonumio-", suffix=".partial",
            delete=False
        ) as fp:
            scratch = Path(fp.name)
            fp.write(data)
            fp.flush()
            os.fsync(fp.fileno())
        if dest.exists() and not overwrite:
            raise AudioFormatError("WAV destination became occupied")
        os.replace(scratch, dest)
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
    return dest


def wave_metadata(buffer: AudioBuffer, settings: QuantizationSettings
                  ) -> dict[str, int | float | str]:
    if not isinstance(buffer, AudioBuffer) or not isinstance(
        settings, QuantizationSettings
    ):
        raise AudioContractError("Expected AudioBuffer and PCM options")
    return {
        "format": "RIFF/WAVE integer PCM",
        "channels": buffer.channels,
        "frames": buffer.frames,
        "sample_rate": buffer.sample_rate,
        "duration_seconds": buffer.seconds,
        "bits_per_sample": settings.bits,
        "estimated_payload_bytes": buffer.frames * buffer.channels * settings.bits // 8,
        "original_neural_model": "none",
    }
