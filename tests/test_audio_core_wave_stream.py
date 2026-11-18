"""Long-form bounded WAV stream exporter verification, with no model loading."""
from __future__ import annotations

import hashlib
import io
import os
import struct
import wave

import numpy as np
import pytest

import sonumio.audio_core.wave_stream as wav_stream
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import AudioContractError, AudioFormatError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import (
    WavPacketResult, WavPacketSpec, WavPacketWriter,
)
from sonumio.audio_core.waveio import (
    decode_wav_bytes, encode_wav_bytes, inspect_wav_bytes, read_wav_file,
)


def _audio(channels, frames):
    t = np.arange(frames, dtype=np.float32)
    base = (0.6 * np.sin(2 * np.pi * t / 127)).astype(np.float32)
    data = np.stack([base * (1 - i * 0.1) for i in range(channels)])
    return make_buffer(data, 24000)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("packet_frames", [1, 7, 64, 333])
def test_chunk_boundaries_produce_same_wav_as_one_shot(
    tmp_path, bits, channels, packet_frames
):
    audio = _audio(channels, 501)
    settings = QuantizationSettings(bits=bits, headroom_db=-0.5)
    destination = tmp_path / "partitioned.wav"
    spec = WavPacketSpec(24000, channels, 501, settings=settings)
    with WavPacketWriter(destination, spec) as writer:
        for start in range(0, audio.frames, packet_frames):
            writer.push(FramePacket(
                24000, start,
                audio.samples[:, start:start + packet_frames],
            ))
        assert writer.frames_written == 501
        assert not destination.exists()
        result = writer.finish()
    assert writer.state == "finished"
    assert isinstance(result, WavPacketResult)
    assert result.bytes_written == destination.stat().st_size
    data = destination.read_bytes()
    assert data == encode_wav_bytes(audio, settings=settings)
    assert result.wav_sha256 == hashlib.sha256(data).hexdigest()
    info, view = inspect_wav_bytes(data)
    assert info.frames == audio.frames
    assert len(view) == audio.frames * channels * bits // 8
    np.testing.assert_array_equal(
        decode_wav_bytes(data).samples, decode_wav_bytes(
            encode_wav_bytes(audio, settings=settings)
        ).samples,
    )
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_odd_size_data_has_exact_riff_padding(tmp_path):
    audio = _audio(1, 5)
    path = tmp_path / "odd.wav"
    with WavPacketWriter(path, WavPacketSpec(24000, 1, 5,
                                            QuantizationSettings(bits=24))) as writer:
        writer.push(FramePacket(24000, 0, audio.samples))
        result = writer.finish()
    raw = path.read_bytes()
    assert len(raw) == 60
    assert struct.unpack_from("<I", raw, 4)[0] == 52
    assert struct.unpack_from("<I", raw, 40)[0] == 15
    assert raw[-1:] == b"\0"
    assert result.bytes_written == len(raw)


def test_stream_aborts_by_default_on_normal_scope_exit(tmp_path):
    path = tmp_path / "not-published.wav"
    with WavPacketWriter(path, WavPacketSpec(24000, 1, 10)) as writer:
        writer.push(FramePacket(24000, 0, _audio(1, 10).samples))
        assert writer.frames_written == 10
    assert writer.state == "aborted"
    assert not path.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_stream_aborts_on_exception_without_publishing(tmp_path):
    path = tmp_path / "unpublished.wav"
    with pytest.raises(RuntimeError, match="cancel"):
        with WavPacketWriter(path, WavPacketSpec(24000, 1, 100)) as writer:
            writer.push(FramePacket(24000, 0, _audio(1, 30).samples))
            raise RuntimeError("cancel")
    assert not path.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_empty_finalization_rejected_but_session_can_continue(tmp_path):
    path = tmp_path / "empty.wav"
    with WavPacketWriter(path, WavPacketSpec(24000, 1, 20)) as writer:
        with pytest.raises(AudioContractError, match="empty"):
            writer.finish()
        assert writer.state == "open"
        writer.push(FramePacket(24000, 0, _audio(1, 8).samples))
        writer.finish()
        with pytest.raises(AudioContractError, match="finalized"):
            writer.finish()
        with pytest.raises(AudioContractError, match="not open"):
            writer.push(FramePacket(24000, 8, _audio(1, 2).samples))
    assert read_wav_file(path).frames == 8


def test_wrong_clock_channels_and_order_rejected_without_advancing(tmp_path):
    path = tmp_path / "order.wav"
    with WavPacketWriter(path, WavPacketSpec(24000, 2, 50)) as writer:
        a = _audio(2, 10).samples
        writer.push(FramePacket(24000, 0, a))
        bad = (
            FramePacket(24000, 9, a),
            FramePacket(24000, 11, a),
            FramePacket(44100, 10, a),
            FramePacket(24000, 10, a[:1]),
        )
        for packet in bad:
            with pytest.raises(AudioContractError):
                writer.push(packet)
            assert writer.frames_written == 10
        with pytest.raises(AudioContractError):
            writer.push("not-pcm")
        with pytest.raises(AudioLimitError, match="maximum_frames"):
            writer.push(FramePacket(24000, 10, _audio(2, 41).samples))
        assert writer.frames_written == 10
        writer.push(FramePacket(24000, 10, _audio(2, 10).samples))
        writer.finish()
    assert read_wav_file(path).frames == 20


def test_packet_size_cap_rejects_large_chunk_without_side_effects(tmp_path):
    frames = wav_stream.MAX_STREAM_PACKET_FRAMES + 1
    path = tmp_path / "large-chunk.wav"
    with WavPacketWriter(path, WavPacketSpec(24000, 1, frames)) as writer:
        packet = FramePacket(24000, 0, np.zeros(frames, dtype=np.float32))
        with pytest.raises(AudioLimitError, match="packet"):
            writer.push(packet)
        assert writer.frames_written == 0
        assert not path.exists()


def test_longer_than_in_memory_duration_is_valid_standard_wav(tmp_path):
    rate = 8000
    total_frames = rate * 121
    path = tmp_path / "long-form.wav"
    spec = WavPacketSpec(rate, 1, total_frames)
    with WavPacketWriter(path, spec) as writer:
        for start in range(0, total_frames, rate):
            block = np.full(rate, start / total_frames * 0.05, dtype=np.float32)
            writer.push(FramePacket(rate, start, block))
        result = writer.finish()
    assert result.frames == total_frames
    assert result.bytes_written == 44 + total_frames * 2
    with wave.open(str(path), "rb") as fp:
        assert fp.getnchannels() == 1
        assert fp.getframerate() == rate
        assert fp.getnframes() == total_frames
        assert fp.getsampwidth() == 2
        assert len(fp.readframes(8)) == 16
    # Legacy one-shot AudioBuffer decoder is intentionally still limited to
    # <=120 seconds. The stream writer does not weaken that import contract.
    with pytest.raises(AudioLimitError):
        read_wav_file(path)


def test_stream_settings_and_declared_geometry_fail_closed(tmp_path):
    for kwargs in (
        {"sample_rate": 7999, "channels": 1, "maximum_frames": 10},
        {"sample_rate": 24000, "channels": 0, "maximum_frames": 10},
        {"sample_rate": 24000, "channels": True, "maximum_frames": 10},
        {"sample_rate": 24000, "channels": 1, "maximum_frames": 0},
        {"sample_rate": 24000, "channels": 1, "maximum_frames": True},
    ):
        with pytest.raises(AudioContractError):
            WavPacketSpec(**kwargs)
    with pytest.raises(AudioContractError, match="dither"):
        WavPacketSpec(24000, 1, 10,
                      settings=QuantizationSettings(dither=True, seed=42))
    with pytest.raises(AudioLimitError, match="3600-second"):
        WavPacketSpec(8000, 1, 8000 * 3601)
    with pytest.raises(AudioLimitError, match="transport"):
        WavPacketSpec(192000, 8, 192000 * 120,
                      settings=QuantizationSettings(bits=32))
    with pytest.raises(AudioFormatError, match="\.wav"):
        WavPacketWriter(tmp_path / "not-a-wav.bin", WavPacketSpec(24000, 1, 10))


def test_no_clobber_atomic_create_and_explicit_replace(tmp_path):
    path = tmp_path / "occupied.wav"
    original = b"other user's file"
    path.write_bytes(original)
    spec = WavPacketSpec(24000, 1, 10)
    with WavPacketWriter(path, spec) as writer:
        writer.push(FramePacket(24000, 0, _audio(1, 10).samples))
        with pytest.raises(AudioFormatError, match="overwrite"):
            writer.finish()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))
    with WavPacketWriter(path, spec, overwrite=True) as writer:
        writer.push(FramePacket(24000, 0, _audio(1, 10).samples))
        writer.finish()
    assert read_wav_file(path).frames == 10


def test_no_clobber_if_racer_creates_destination_during_link(tmp_path, monkeypatch):
    path = tmp_path / "competing.wav"
    spec = WavPacketSpec(24000, 1, 10)
    real_link = os.link

    def competitor(src, dest, *args, **kwargs):
        path.write_bytes(b"competitor")
        return real_link(src, dest, *args, **kwargs)

    monkeypatch.setattr(wav_stream.os, "link", competitor)
    with WavPacketWriter(path, spec) as writer:
        writer.push(FramePacket(24000, 0, _audio(1, 10).samples))
        with pytest.raises(AudioFormatError, match="overwrite"):
            writer.finish()
    assert path.read_bytes() == b"competitor"
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_symlink_output_rejected(tmp_path):
    target = tmp_path / "real.wav"
    target.write_bytes(b"regular")
    link = tmp_path / "symlink.wav"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Host does not allow creating symlinks")
    with pytest.raises(AudioFormatError, match="directory"):
        WavPacketWriter(link, WavPacketSpec(24000, 1, 10))


def test_cannot_use_writer_after_explicit_abort(tmp_path):
    path = tmp_path / "aborted.wav"
    writer = WavPacketWriter(path, WavPacketSpec(24000, 1, 10))
    writer.abort()
    writer.abort()
    assert writer.state == "aborted"
    with pytest.raises(AudioContractError, match="not open"):
        writer.push(FramePacket(24000, 0, _audio(1, 10).samples))
    assert not path.exists()


def test_constructor_does_not_create_parent_without_optin(tmp_path):
    spec = WavPacketSpec(24000, 1, 10)
    child = tmp_path / "missing" / "audio.wav"
    with pytest.raises(AudioFormatError, match="directory"):
        WavPacketWriter(child, spec)
    with WavPacketWriter(child, spec, create_parent=True) as writer:
        assert child.parent.is_dir()
        writer.push(FramePacket(24000, 0, _audio(1, 10).samples))
        writer.finish()
    assert child.exists()
