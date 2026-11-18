"""Explicit, bounded, incremental *local* PCM WAVE export.

This is a transport utility for sequences of verified FramePacket objects.
No neural inference, model weights, audio capture, network activity, or
automatic publishing. A temporary WAV is visible only in the developer-owned
destination directory; destination creation requires an explicit finish().
The standard AudioBuffer WAV reader still enforces its 120-second limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import struct
import tempfile

from .contracts import check_rate, MAX_CHANNELS
from .dither import QuantizationSettings, encode_pcm_bytes
from .hashing import strict_sha256
from .stream_dither import StreamDitherQuantizer
from .errors import AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError
from .streaming import FramePacket
from .waveio import MAX_WAV_BYTES

MAX_STREAM_SECONDS = 3_600
MAX_STREAM_PACKET_FRAMES = 65_536
_HEADER_BYTES = 44
_HASH_BLOCK_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class WavPacketSpec:
    """Fixed PCM clock and a predeclared maximum destination length.

    Undithered PCM matches the established one-shot encoder. Seeded TPDF
    uses the explicitly versioned, stateful StreamDitherQuantizer protocol:
    RNG draws stay continuous over validated packet boundaries.
    """

    sample_rate: int
    channels: int
    maximum_frames: int
    settings: QuantizationSettings = field(default_factory=QuantizationSettings)

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if type(self.channels) is not int or not 1 <= self.channels <= MAX_CHANNELS:
            raise AudioContractError("WAV stream requires 1..8 channels")
        if not isinstance(self.settings, QuantizationSettings):
            raise AudioContractError("Invalid stream PCM settings")
        if type(self.maximum_frames) is not int or self.maximum_frames <= 0:
            raise AudioContractError("WAV stream maximum_frames must be positive")
        if self.maximum_frames > self.sample_rate * MAX_STREAM_SECONDS:
            raise AudioLimitError("WAV stream exceeds 3600-second ceiling")
        if (self.maximum_data_bytes + (self.maximum_data_bytes & 1)
                + _HEADER_BYTES > MAX_WAV_BYTES):
            raise AudioLimitError("WAV stream exceeds RIFF transport byte cap")

    @property
    def bytes_per_frame(self) -> int:
        return self.channels * (self.settings.bits // 8)

    @property
    def maximum_data_bytes(self) -> int:
        return self.maximum_frames * self.bytes_per_frame


@dataclass(frozen=True, slots=True)
class WavPacketResult:
    path: Path
    sample_rate: int
    channels: int
    frames: int
    bits_per_sample: int
    bytes_written: int
    wav_sha256: str


def _header(spec: WavPacketSpec, data_bytes: int) -> bytes:
    """Minimal RIFF/WAVE integer PCM header, including odd-data RIFF padding."""
    padding = data_bytes & 1
    riff_size = 36 + data_bytes + padding
    if riff_size + 8 > MAX_WAV_BYTES or riff_size >= (1 << 32):
        raise AudioLimitError("RIFF/WAVE stream length overflow")
    align = spec.bytes_per_frame
    header = (b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
              + b"fmt " + struct.pack(
                  "<IHHIIHH", 16, 1, spec.channels, spec.sample_rate,
                  spec.sample_rate * align, align, spec.settings.bits,
              ) + b"data" + struct.pack("<I", data_bytes))
    if len(header) != _HEADER_BYTES:
        raise AudioFormatError("Invalid streaming WAV header")
    return header


class WavPacketWriter:
    """Write exact sequential FramePackets, with explicit atomic publication.

    Always use as a context manager. Exiting without finish(), even from an
    exception, deletes the temporary output and NEVER creates the final WAV.
    Paths must be chosen by a trusted local developer, not untrusted API users.
    """

    def __init__(
        self, destination: str | Path, spec: WavPacketSpec, *,
        overwrite: bool = False, create_parent: bool = False,
    ) -> None:
        if not isinstance(spec, WavPacketSpec):
            raise AudioContractError("Expected WavPacketSpec")
        if type(overwrite) is not bool or type(create_parent) is not bool:
            raise AudioContractError("WAV file-writing options must be boolean")
        dest = Path(destination)
        if dest.suffix.lower() != ".wav" or dest.name in ("", ".", ".."):
            raise AudioFormatError("WAV stream destination must end in .wav")
        if create_parent:
            dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.parent.is_dir() or dest.is_symlink():
            raise AudioFormatError("WAV stream destination directory invalid")
        self.destination = dest
        self.spec = spec
        self.overwrite = overwrite
        self._state = "open"
        self._frames = 0
        self._dither = (StreamDitherQuantizer(spec.settings)
                        if spec.settings.dither else None)
        self._fp = None
        self._scratch = None
        try:
            fp = tempfile.NamedTemporaryFile(
                mode="w+b", dir=dest.parent, prefix=".sonumio-stream-",
                suffix=".partial", delete=False,
            )
            self._fp = fp
            self._scratch = Path(fp.name)
            if fp.write(b"\0" * _HEADER_BYTES) != _HEADER_BYTES:
                raise OSError("Cannot initialize temporary WAVE header")
        except BaseException:
            self.abort()
            raise

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def state(self) -> str:
        return self._state

    def __enter__(self) -> "WavPacketWriter":
        if self._state != "open":
            raise AudioContractError("Cannot re-enter finished WAV writer")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.abort()

    def push(self, packet: FramePacket) -> None:
        """Append a bounded packet without buffering the entire output."""
        if self._state != "open":
            raise AudioContractError("WAV packet writer is not open")
        if not isinstance(packet, FramePacket):
            raise AudioContractError("WAV packet writer needs FramePacket")
        if packet.sample_rate != self.spec.sample_rate:
            raise AudioContractError("WAV stream sample clock changed")
        if packet.channels != self.spec.channels:
            raise AudioContractError("WAV stream channel count changed")
        if packet.start_frame != self._frames:
            raise AudioContractError("WAV stream packet gap, overlap or reordering")
        if packet.frames > MAX_STREAM_PACKET_FRAMES:
            raise AudioLimitError("WAV stream packet exceeds 65536-frame cap")
        if packet.stop_frame > self.spec.maximum_frames:
            raise AudioLimitError("WAV stream exceeded declared maximum_frames")
        # Packet validation and PCM encoding occur before changing file state.
        encoded = (self._dither.encode(packet.samples)
                   if self._dither is not None
                   else encode_pcm_bytes(packet.samples, settings=self.spec.settings))
        if len(encoded) != packet.frames * self.spec.bytes_per_frame:
            raise AudioFormatError("WAV stream PCM encoder produced wrong byte count")
        try:
            if self._fp.write(encoded) != len(encoded):
                raise OSError("Short PCM write to scratch WAVE")
        except BaseException:
            self.abort()
            raise
        self._frames = packet.stop_frame

    def finish(self, *, expected_sha256: str | None = None) -> WavPacketResult:
        """Finalize, fsync, verify optional digest, then atomically publish WAV.

        overwrite=False uses atomic hard-link create-if-absent; unsupported
        filesystems fail closed. overwrite=True explicitly allows replacement.
        A supplied lowercase SHA256 is checked against the full scratch
        RIFF/WAVE bytes *before* a destination is created or overwritten.
        Scratch is deleted whether publication succeeds or fails.
        """
        if self._state != "open":
            raise AudioContractError("WAV packet writer already finalized")
        if expected_sha256 is not None:
            strict_sha256(expected_sha256)
        if self._frames == 0:
            raise AudioContractError("Cannot finalize an empty WAV stream")
        data_bytes = self._frames * self.spec.bytes_per_frame
        padding = data_bytes & 1
        size = _HEADER_BYTES + data_bytes + padding
        try:
            if padding and self._fp.write(b"\0") != 1:
                raise OSError("Cannot write WAVE data padding")
            self._fp.seek(0)
            if self._fp.write(_header(self.spec, data_bytes)) != _HEADER_BYTES:
                raise OSError("Cannot finalize WAVE header")
            self._fp.flush()
            os.fsync(self._fp.fileno())
            self._fp.seek(0)
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = self._fp.read(min(remaining, _HASH_BLOCK_BYTES))
                if not chunk:
                    raise AudioFormatError("Temporary WAV changed during checksum")
                digest.update(chunk)
                remaining -= len(chunk)
            if self._fp.read(1):
                raise AudioFormatError("Temporary WAV has unexpected trailing bytes")
            actual_sha256 = digest.hexdigest()
            if expected_sha256 is not None and actual_sha256 != expected_sha256:
                raise AudioIntegrityError(
                    "WAV output SHA256 mismatch before publication"
                )
            self._fp.close()
            self._fp = None
            if self.overwrite:
                if self.destination.is_symlink():
                    raise AudioFormatError("WAV output became a symlink")
                os.replace(self._scratch, self.destination)
            else:
                try:
                    os.link(self._scratch, self.destination)
                except FileExistsError as exc:
                    raise AudioFormatError(
                        "Existing WAV requires explicit overwrite"
                    ) from exc
                except OSError as exc:
                    raise AudioFormatError(
                        "Atomic no-clobber WAV output unsupported on filesystem"
                    ) from exc
            self._state = "finished"
            return WavPacketResult(
                self.destination, self.spec.sample_rate, self.spec.channels,
                self._frames, self.spec.settings.bits, size, actual_sha256,
            )
        except BaseException:
            self._state = "aborted"
            raise
        finally:
            if self._fp is not None:
                self._fp.close()
                self._fp = None
            if self._scratch is not None:
                self._scratch.unlink(missing_ok=True)
                self._scratch = None

    def abort(self) -> None:
        """Dispose of the unpublished partial file; no final WAV is removed."""
        if self._state == "finished":
            return
        self._state = "aborted"
        if self._fp is not None:
            try:
                self._fp.close()
            finally:
                self._fp = None
        if self._scratch is not None:
            self._scratch.unlink(missing_ok=True)
            self._scratch = None
