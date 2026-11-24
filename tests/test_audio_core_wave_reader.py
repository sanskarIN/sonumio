"""Canonical long-form WAV packet import, integrity and format refusal."""
from __future__ import annotations

import hashlib
import os
import struct
import wave

import numpy as np
import pytest

import sonumio.audio_core.wave_reader as wave_reader
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_reader import WavPacketReader
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter


def _write(tmp_path, *, frames=311, channels=2, bits=16):
    path = tmp_path / "packet-source.wav"
    spec = WavPacketSpec(24000, channels, frames, QuantizationSettings(bits=bits))
    data = np.sin(np.arange(frames, dtype=np.float32) / 11).astype(np.float32)
    with WavPacketWriter(path, spec) as out:
        for start in range(0, frames, 91):
            fragment = np.stack([
                data[start:start+91] * (0.8 - i * 0.1)
                for i in range(channels)
            ])
            out.push(FramePacket(24000, start, fragment))
        result = out.finish()
    return path, result


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("packet_frames", [1, 17, 128])
def test_packet_reader_roundtrip_and_whole_file_sha(
    tmp_path, bits, channels, packet_frames
):
    path, result = _write(tmp_path, channels=channels, bits=bits)
    with WavPacketReader(path, packet_frames=packet_frames,
                         expected_sha256=result.wav_sha256) as reader:
        assert reader.verified_sha256 is None
        assert reader.info.channels == channels
        assert reader.info.bits_per_sample == bits
        packets = list(reader)
        assert reader.state == "verified"
        assert reader.frames_read == 311
        assert reader.verified_sha256 == result.wav_sha256
        assert sum(p.frames for p in packets) == 311
        assert [p.start_frame for p in packets] == list(
            range(0, 311, packet_frames)
        )
        assert all(p.samples.flags.writeable is False for p in packets)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result.wav_sha256


def test_reader_supports_121_second_file_without_full_memory_load(tmp_path):
    rate = 8000
    total = 121 * rate
    path = tmp_path / "long.wav"
    with WavPacketWriter(path, WavPacketSpec(rate, 1, total)) as writer:
        for offset in range(0, total, 8000):
            writer.push(FramePacket(rate, offset,
                                    np.full(8000, 0.125, dtype=np.float32)))
        result = writer.finish()
    with WavPacketReader(path, packet_frames=16000,
                         expected_sha256=result.wav_sha256) as reader:
        count = 0
        for packet in reader:
            assert packet.frames <= 16000
            assert np.isfinite(packet.samples).all()
            count += packet.frames
        assert count == total
        assert reader.verified_sha256 is not None
    with wave.open(str(path), "rb") as stream:
        assert stream.getnframes() == total


def test_reader_close_before_full_consumption_does_not_claim_verification(tmp_path):
    path, result = _write(tmp_path)
    with WavPacketReader(path, packet_frames=30,
                         expected_sha256=result.wav_sha256) as reader:
        first = next(reader)
        assert first.frames == 30
        assert reader.verified_sha256 is None
    assert reader.state == "closed"
    with pytest.raises(AudioContractError, match="closed"):
        next(reader)


def test_mismatched_expected_sha_fails_before_final_packet_delivery(tmp_path):
    path, _ = _write(tmp_path)
    with WavPacketReader(path, packet_frames=100,
                         expected_sha256="0"*64) as reader:
        with pytest.raises(AudioIntegrityError, match="SHA256"):
            list(reader)
        assert reader.verified_sha256 is None
        assert reader.state == "closed"


@pytest.mark.parametrize("byte_index, replacement", [
    (0, b"RIFX"),
    (12, b"JUNK"),
    (20, b"\x03\x00"),
    (34, b"\x18\x00"),
    (36, b"JUNK"),
])
def test_reader_rejects_unsupported_or_corrupted_pcm_header(
    tmp_path, byte_index, replacement
):
    path, _ = _write(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[byte_index:byte_index+len(replacement)] = replacement
    path.write_bytes(raw)
    with pytest.raises(AudioFormatError):
        WavPacketReader(path)


def test_bad_riff_size_and_truncated_data_rejected(tmp_path):
    path, _ = _write(tmp_path)
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(AudioFormatError, match="RIFF"):
        WavPacketReader(path)
    path.write_bytes(original)
    bad = bytearray(original)
    struct.pack_into("<I", bad, 4, 1)
    path.write_bytes(bad)
    with pytest.raises(AudioFormatError, match="RIFF"):
        WavPacketReader(path)


def test_corrupt_padding_is_rejected_at_last_frame(tmp_path):
    path, _ = _write(tmp_path, frames=5, channels=1, bits=24)
    raw = bytearray(path.read_bytes())
    assert len(raw) == 60
    raw[-1] = 42
    path.write_bytes(raw)
    with WavPacketReader(path) as reader:
        with pytest.raises(AudioFormatError, match="padding"):
            list(reader)


def test_midread_source_mutation_is_detected(tmp_path):
    path, exported = _write(tmp_path, frames=500)
    with WavPacketReader(path, packet_frames=100,
                         expected_sha256=exported.wav_sha256) as reader:
        first = next(reader)
        assert first.frames == 100
        with path.open("r+b") as writable:
            writable.seek(44 + 200 * 2 * 2)
            writable.write(b"\xff\x7f")
            writable.flush()
            os.fsync(writable.fileno())
        with pytest.raises(AudioIntegrityError, match="changed|SHA256"):
            list(reader)
        assert reader.verified_sha256 is None


def test_reader_requires_regular_file_and_bounded_packet_size(tmp_path):
    path, _ = _write(tmp_path)
    for limit in (0, -1, True, 65_537):
        with pytest.raises(AudioContractError, match="packet frame"):
            WavPacketReader(path, packet_frames=limit)
    with pytest.raises(AudioContractError, match="SHA256"):
        WavPacketReader(path, expected_sha256="bad")
    with pytest.raises(AudioFormatError, match="ordinary"):
        WavPacketReader(tmp_path)
    with pytest.raises(AudioFormatError, match="ordinary"):
        WavPacketReader(tmp_path / "missing.wav")


def test_reader_rejects_symlink_where_supported(tmp_path):
    path, _ = _write(tmp_path)
    link = tmp_path / "link.wav"
    try:
        link.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unavailable on this host")
    with pytest.raises(AudioFormatError, match="ordinary"):
        WavPacketReader(link)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX named pipes only")
def test_reader_rejects_fifo_without_blocking(tmp_path):
    path = tmp_path / "pipe.wav"
    os.mkfifo(path)
    with pytest.raises(AudioFormatError, match="ordinary"):
        WavPacketReader(path)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX O_NOFOLLOW")
def test_reader_closes_symlink_swap_after_preflight(tmp_path, monkeypatch):
    path, _ = _write(tmp_path)
    other = tmp_path / "other.wav"
    other.write_bytes(path.read_bytes())
    real_open = os.open

    def swap(path_arg, flags, *args, **kwargs):
        if str(path_arg) == str(path):
            path.unlink()
            path.symlink_to(other)
        return real_open(path_arg, flags, *args, **kwargs)

    monkeypatch.setattr(wave_reader.os, "open", swap)
    with pytest.raises(AudioFormatError, match="ordinary"):
        WavPacketReader(path)



def test_reader_rejects_same_bytes_new_inode_between_preflight_and_open(
    tmp_path, monkeypatch
):
    """SHA equality alone cannot validate that an opened path stayed put."""
    path, _ = _write(tmp_path)
    contents = path.read_bytes()
    archived = tmp_path / "renamed-original.wav"
    original_open = os.open
    swaps = []

    def swap_before_descriptor(file, flags, *args, **kwargs):
        if os.fspath(file) == os.fspath(path) and not swaps:
            path.replace(archived)
            path.write_bytes(contents)  # identical bytes, different identity
            swaps.append(True)
        return original_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(wave_reader.os, "open", swap_before_descriptor)
    with pytest.raises(AudioIntegrityError, match="between path validation"):
        WavPacketReader(path)
    assert swaps and archived.read_bytes() == contents


@pytest.mark.skipif(os.name == "nt", reason="Windows may prevent renaming open WAVs")
def test_reader_rejects_same_byte_replacement_during_stream(tmp_path):
    path, expected = _write(tmp_path, frames=500)
    contents = path.read_bytes()
    old = tmp_path / "old-descriptor.wav"
    with WavPacketReader(path, packet_frames=100,
                         expected_sha256=expected.wav_sha256) as reader:
        assert next(reader).frames == 100
        path.replace(old)
        path.write_bytes(contents)
        with pytest.raises(AudioIntegrityError, match="path changed"):
            list(reader)
        assert reader.verified_sha256 is None
        assert reader.state == "closed"


@pytest.mark.skipif(os.name == "nt", reason="Windows may prevent unlinking open WAVs")
def test_reader_rejects_unlinked_source_even_if_descriptor_bytes_match(tmp_path):
    path, expected = _write(tmp_path, frames=500)
    with WavPacketReader(path, packet_frames=100,
                         expected_sha256=expected.wav_sha256) as reader:
        assert next(reader).frames == 100
        path.unlink()
        with pytest.raises(AudioIntegrityError, match="path disappeared"):
            list(reader)
        assert reader.verified_sha256 is None
        assert reader.state == "closed"


def test_reader_does_not_follow_replacement_symlink_without_nofollow(
    tmp_path, monkeypatch
):
    """A no-follow-less platform still must reject a leaf link after preflight."""
    path, _ = _write(tmp_path)
    alternate = tmp_path / "alternate.wav"
    alternate.write_bytes(path.read_bytes())
    original_open = os.open
    swapped = []

    def swap(file, flags, *args, **kwargs):
        if os.fspath(file) == os.fspath(path) and not swapped:
            path.unlink()
            try:
                path.symlink_to(alternate)
            except (OSError, NotImplementedError):
                pytest.skip("Symlinks unavailable on this host")
            swapped.append(True)
            # Simulate a platform where O_NOFOLLOW is not available.
            flags &= ~getattr(os, "O_NOFOLLOW", 0)
        return original_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(wave_reader.os, "open", swap)
    with pytest.raises(AudioIntegrityError, match="between path validation"):
        WavPacketReader(path)
    assert swapped



def test_reader_closes_unverified_descriptor_on_packet_read_os_error(tmp_path):
    path, _ = _write(tmp_path, frames=500)
    reader = WavPacketReader(path, packet_frames=100)

    class FaultyStream:
        def __init__(self):
            self.closed = False

        def read(self, size):
            raise OSError("simulated device read failure")

        def close(self):
            self.closed = True

    faulty = FaultyStream()
    reader._fp.close()
    reader._fp = faulty
    with pytest.raises(OSError, match="simulated device"):
        next(reader)
    assert reader.state == "closed"
    assert reader.verified_sha256 is None
    assert faulty.closed is True


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor metadata mutation semantics")
def test_reader_inplace_metadata_change_keeps_precise_error_class(tmp_path):
    """A stable pathname with altered mtime is a changed source, not a rename."""
    path, exported = _write(tmp_path, frames=500)
    with WavPacketReader(
        path, packet_frames=100, expected_sha256=exported.wav_sha256,
    ) as reader:
        assert next(reader).frames == 100
        snapshot = path.stat()
        os.utime(path, ns=(
            snapshot.st_atime_ns, snapshot.st_mtime_ns + 2_000_000_000,
        ))
        with pytest.raises(AudioIntegrityError, match="source changed during packet reading"):
            list(reader)
        assert reader.state == "closed"
        assert reader.verified_sha256 is None
