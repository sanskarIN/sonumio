"""Bounded, canonical RIFF/WAVE PCM reader that yields immutable FramePackets.

Only the exact 44-byte-header integer PCM shape emitted by WavPacketWriter
is accepted; metadata chunks, WAVE_FORMAT_EXTENSIBLE, float WAV, RF64 and
noncanonical RIFF representations are explicitly unsupported. No model
loading, network access or recording-rights inference is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import struct

from .contracts import check_rate, MAX_CHANNELS
from .dither import dequantize_pcm, unpack_pcm
from .errors import AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError
from .hashing import strict_sha256
from .streaming import FramePacket
from .wave_stream import (
    MAX_STREAM_SECONDS, MAX_STREAM_PACKET_FRAMES,
)
from .waveio import MAX_WAV_BYTES

_HEADER_BYTES = 44


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare platform-exposed identity and change metadata for one file.

    This is best-effort local path integrity, not a filesystem sandbox or
    protection against hostile privileged processes able to spoof metadata.
    """
    return (
        left.st_dev, left.st_ino, left.st_size,
        left.st_mtime_ns, left.st_ctime_ns,
    ) == (
        right.st_dev, right.st_ino, right.st_size,
        right.st_mtime_ns, right.st_ctime_ns,
    )


@dataclass(frozen=True, slots=True)
class WavPacketInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    frames: int
    data_bytes: int
    file_bytes: int


def _decode_canonical_header(raw: bytes, file_bytes: int) -> WavPacketInfo:
    if len(raw) != _HEADER_BYTES:
        raise AudioFormatError("Truncated canonical WAVE header")
    if (raw[:4] != b"RIFF" or raw[8:12] != b"WAVE"
            or raw[12:16] != b"fmt " or raw[36:40] != b"data"):
        raise AudioFormatError("Only canonical RIFF/WAVE integer PCM is supported")
    riff_size = struct.unpack_from("<I", raw, 4)[0]
    fmt_size = struct.unpack_from("<I", raw, 16)[0]
    audio_format, channels, rate, byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", raw, 20,
    )
    data_bytes = struct.unpack_from("<I", raw, 40)[0]
    if fmt_size != 16 or audio_format != 1 or bits not in (8, 16, 24, 32):
        raise AudioFormatError("Unsupported or noncanonical WAV sample encoding")
    check_rate(rate)
    if type(channels) is not int or not 1 <= channels <= MAX_CHANNELS:
        raise AudioFormatError("Invalid streaming WAVE channel count")
    align = channels * bits // 8
    if (block_align != align or byte_rate != rate * align or
            data_bytes <= 0 or data_bytes % align):
        raise AudioFormatError("Invalid streaming WAVE PCM block geometry")
    padding = data_bytes & 1
    if (file_bytes != _HEADER_BYTES + data_bytes + padding or
            riff_size != file_bytes - 8):
        raise AudioFormatError("RIFF chunk sizes do not match the exact file")
    frames = data_bytes // align
    if frames > rate * MAX_STREAM_SECONDS:
        raise AudioLimitError("WAV stream exceeds 3600-second duration cap")
    return WavPacketInfo(rate, channels, bits, frames, data_bytes, file_bytes)


class WavPacketReader:
    """Read a bounded local WAV as sequential PCM packets.

    The source hash is verified only *after* all packets are consumed.
    Do not publish derived outputs until verified_sha256 becomes available.
    POSIX O_NOFOLLOW/O_NONBLOCK protect against common symlink/FIFO swaps;
    no hostile-path sandbox or proof of recording rights is implied.
    """

    def __init__(
        self, path: str | Path, *, packet_frames: int = 8192,
        expected_sha256: str | None = None,
    ) -> None:
        if (type(packet_frames) is not int or
                not 1 <= packet_frames <= MAX_STREAM_PACKET_FRAMES):
            raise AudioContractError("Invalid WAVE reader packet frame size")
        if expected_sha256 is not None:
            strict_sha256(expected_sha256)
        source = Path(path)
        # Snapshot the *leaf* before opening it, without following links.
        # O_NOFOLLOW is not universally available (notably on Windows), so
        # compare the opened descriptor with this exact preflight identity.
        try:
            preflight = source.lstat()
        except (OSError, ValueError) as exc:
            raise AudioFormatError("Expected ordinary local WAVE file") from exc
        if not stat.S_ISREG(preflight.st_mode):
            raise AudioFormatError("Expected ordinary local WAVE file")
        flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0) |
                 getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        try:
            fd = os.open(source, flags)
        except OSError as exc:
            raise AudioFormatError("Cannot open ordinary local WAVE file") from exc
        try:
            # Never prefetch into Python buffered IO: the buffered reader
            # can cache an entire small WAV before subsequent source edits.
            fp = os.fdopen(fd, "rb", buffering=0)
        except BaseException:
            os.close(fd)
            raise
        try:
            before = os.fstat(fp.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AudioFormatError("Expected ordinary local WAVE descriptor")
            if not _same_file_snapshot(preflight, before):
                raise AudioIntegrityError(
                    "WAVE source changed between path validation and descriptor open"
                )
            if before.st_size > MAX_WAV_BYTES:
                raise AudioLimitError("WAV stream file exceeds transport byte cap")
            raw_header = fp.read(_HEADER_BYTES)
            info = _decode_canonical_header(raw_header, before.st_size)
        except BaseException:
            fp.close()
            raise
        self.path = source
        self.info = info
        self.packet_frames = packet_frames
        self._expected_sha256 = expected_sha256
        self._fp = fp
        self._stat = before
        self._frames_read = 0
        self._state = "open"
        self._hash = hashlib.sha256()
        self._hash.update(raw_header)
        self._verified_sha256 = None

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def verified_sha256(self) -> str | None:
        return self._verified_sha256

    @property
    def state(self) -> str:
        return self._state

    def __enter__(self) -> "WavPacketReader":
        if self._state != "open":
            raise AudioContractError("Cannot re-enter a closed WAVE reader")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def __iter__(self) -> "WavPacketReader":
        return self

    def __next__(self) -> FramePacket:
        if self._state == "verified":
            raise StopIteration
        if self._state != "open":
            raise AudioContractError("Cannot read from closed WAVE reader")
        left = self.info.frames - self._frames_read
        if left <= 0:
            raise AudioIntegrityError("WAVE reader reached inconsistent frame state")
        count = min(left, self.packet_frames)
        size = count * self.info.channels * (self.info.bits_per_sample // 8)
        try:
            raw = self._fp.read(size)
        except BaseException:
            # An I/O error can happen before any bytes are returned. Do not
            # leave an unverified descriptor open outside a context manager.
            self.close()
            raise
        if len(raw) != size:
            self.close()
            raise AudioFormatError("Truncated WAVE payload during packet read")
        self._hash.update(raw)
        try:
            ints = unpack_pcm(
                raw, bits=self.info.bits_per_sample, channels=self.info.channels,
            )
            floats = dequantize_pcm(ints, bits=self.info.bits_per_sample)
            packet = FramePacket(self.info.sample_rate, self._frames_read, floats)
        except BaseException:
            self.close()
            raise
        self._frames_read += count
        if self._frames_read == self.info.frames:
            self._finish_verification()
        return packet

    def _finish_verification(self) -> None:
        try:
            if self.info.data_bytes & 1:
                pad = self._fp.read(1)
                if pad != b"\0":
                    raise AudioFormatError("Invalid canonical RIFF padding")
                self._hash.update(pad)
            if self._fp.read(1):
                raise AudioIntegrityError("WAVE file grew after header validation")
            after = os.fstat(self._fp.fileno())
            before = self._stat
            if (after.st_size != before.st_size or
                    after.st_mtime_ns != before.st_mtime_ns or
                    after.st_ctime_ns != before.st_ctime_ns):
                raise AudioIntegrityError("WAVE source changed during packet reading")
            # A verified descriptor checksum is not enough when the path
            # itself was renamed/replaced: the caller identifies this file
            # by path, so it must still name the same regular file at EOF.
            try:
                at_path = self.path.lstat()
            except (OSError, ValueError) as exc:
                raise AudioIntegrityError(
                    "WAVE source path disappeared during packet reading"
                ) from exc
            if (not stat.S_ISREG(at_path.st_mode)
                    or not _same_file_snapshot(before, at_path)):
                raise AudioIntegrityError(
                    "WAVE source path changed during packet reading"
                )
            actual = self._hash.hexdigest()
            if (self._expected_sha256 is not None and
                    actual != self._expected_sha256):
                raise AudioIntegrityError("WAVE source SHA256 mismatch")
            self._verified_sha256 = actual
            self._state = "verified"
        except BaseException:
            self.close()
            raise
        finally:
            if self._fp is not None and self._state == "verified":
                self._fp.close()
                self._fp = None

    def close(self) -> None:
        if self._state == "open":
            self._state = "closed"
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None
