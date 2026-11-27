"""Reusable verified, bounded local PCM snapshot.

Keep the canonical packet reader's descriptor and full-file hash checks,
but assemble into a single preallocated offline buffer. The returned
digest identifies precisely the source bytes that were decoded, not a
separate pathname read.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .contracts import AudioBuffer, AudioSpec, AudioLayout, make_buffer
from .errors import AudioContractError, AudioIntegrityError, AudioLimitError
from .wave_reader import WavPacketReader, WavPacketInfo

MAX_SNAPSHOT_BYTES = 24 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class VerifiedPcmSnapshot:
    audio: AudioBuffer
    info: WavPacketInfo
    sha256: str

    def as_dict(self):
        return {
            "source_sha256": self.sha256,
            "source_frames": self.info.frames,
            "sample_rate": self.info.sample_rate,
            "channels": self.info.channels,
            "source_bits": self.info.bits_per_sample,
            "source_bytes": self.info.file_bytes,
            "source_verified": True,
        }


def read_verified_pcm(path, *, expected_sha256=None,
                      packet_frames=8192, maximum_pcm_bytes=MAX_SNAPSHOT_BYTES):
    if (type(maximum_pcm_bytes) is not int
            or not 4 <= maximum_pcm_bytes <= MAX_SNAPSHOT_BYTES):
        raise AudioContractError("Invalid verified snapshot byte budget")
    with WavPacketReader(Path(path), expected_sha256=expected_sha256,
                         packet_frames=packet_frames) as reader:
        info = reader.info
        layout = AudioLayout.for_channels(info.channels)
        spec = AudioSpec(info.sample_rate, layout, info.frames)
        if spec.byte_count > maximum_pcm_bytes:
            raise AudioLimitError("Verified PCM snapshot exceeds offline memory budget")
        samples = np.empty((info.channels, info.frames), dtype=np.float32)
        position = 0
        for packet in reader:
            if (packet.start_frame != position or packet.stop_frame > info.frames
                    or packet.channels != info.channels
                    or packet.sample_rate != info.sample_rate):
                raise AudioIntegrityError("Inconsistent canonical snapshot packet")
            samples[:, position:packet.stop_frame] = packet.samples
            position = packet.stop_frame
        digest = reader.verified_sha256
        if position != info.frames or digest is None:
            raise AudioIntegrityError("PCM snapshot is not fully verified")
    return VerifiedPcmSnapshot(
        make_buffer(samples, info.sample_rate, layout=layout,
                    source_note="verified local PCM snapshot"),
        info, digest,
    )
