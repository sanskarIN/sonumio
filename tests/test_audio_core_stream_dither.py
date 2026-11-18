"""Seeded TPDF PCM stream invariance across frames, channels and bit depths."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sonumio.audio_core.dither import (
    QuantizationSettings, encode_pcm_bytes,
)
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.stream_dither import (
    MAX_DITHER_SAMPLES_PER_CALL, STREAM_TPDF_VERSION, StreamDitherQuantizer,
)
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.audio_core.waveio import inspect_wav_bytes


def _samples(channels=2, frames=257):
    frames_axis = np.arange(frames, dtype=np.float32)
    return np.stack([
        .3 * np.sin(frames_axis / 9 + channel / 11)
        for channel in range(channels)
    ]).astype(np.float32)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("packet_frames", [1, 3, 17, 64, 257])
def test_seeded_tpdf_wav_bytes_are_identical_across_packet_boundaries(
    tmp_path, bits, channels, packet_frames
):
    samples = _samples(channels, 257)
    settings = QuantizationSettings(
        bits=bits, dither=True, seed=917, headroom_db=-1.25,
    )
    expected_pcm = StreamDitherQuantizer(settings).encode(samples)
    target = tmp_path / "stream.wav"
    with WavPacketWriter(
        target, WavPacketSpec(24000, channels, 257, settings)
    ) as writer:
        for offset in range(0, 257, packet_frames):
            writer.push(FramePacket(
                24000, offset, samples[:, offset:offset + packet_frames]
            ))
        result = writer.finish()
    raw = target.read_bytes()
    assert raw[44:44 + len(expected_pcm)] == expected_pcm
    assert raw[44 + len(expected_pcm):] == (b"\0" if len(expected_pcm) & 1 else b"")
    assert result.wav_sha256 == hashlib.sha256(raw).hexdigest()
    assert result.bytes_written == len(raw)
    info, data = inspect_wav_bytes(raw)
    assert info.channels == channels and info.bits_per_sample == bits
    assert data.tobytes() == expected_pcm


def test_dither_stream_never_replays_random_draws_when_packet_size_changes():
    data = _samples(2, 400)
    settings = QuantizationSettings(bits=16, dither=True, seed=12345)
    one = StreamDitherQuantizer(settings)
    entire = one.encode(data)
    chunk = StreamDitherQuantizer(settings)
    groups = [data[:, :1], data[:, 1:128], data[:, 128:399], data[:, 399:]]
    partitioned = b"".join(chunk.encode(part) for part in groups)
    assert entire == partitioned
    assert one.samples_processed == chunk.samples_processed == data.size
    assert STREAM_TPDF_VERSION == "pcg64-frame-major-interleaved-pairs-v1"


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_different_seed_changes_dithered_output_but_identical_seed_repeats(bits):
    data = np.zeros((2, 500), dtype=np.float32)
    a = QuantizationSettings(bits=bits, dither=True, seed=1)
    b = QuantizationSettings(bits=bits, dither=True, seed=2)
    first = StreamDitherQuantizer(a).encode(data)
    assert first == StreamDitherQuantizer(a).encode(data)
    assert first != StreamDitherQuantizer(b).encode(data)


def test_dithered_stereo_is_deterministic_in_interleaved_frame_order():
    data = np.zeros((2, 100), dtype=np.float32)
    setting = QuantizationSettings(bits=16, dither=True, seed=23)
    quantizer = StreamDitherQuantizer(setting)
    encoded = quantizer.quantize(data)
    expected_rng = np.random.Generator(np.random.PCG64(23))
    pairs = expected_rng.random((200, 2))
    expected = np.rint(pairs[:, 0] - pairs[:, 1])
    np.testing.assert_array_equal(encoded, expected.reshape(100, 2).T)
    assert quantizer.samples_processed == 200


def test_legacy_nonstream_dither_remains_unmodified():
    samples = _samples(2, 97)
    settings = QuantizationSettings(bits=16, dither=True, seed=5)
    previous = encode_pcm_bytes(samples, settings=settings)
    after = encode_pcm_bytes(samples, settings=settings)
    assert after == previous


def test_requires_explicit_seed_and_dither_true():
    with pytest.raises(AudioContractError, match="seed"):
        QuantizationSettings(bits=16, dither=True)
    with pytest.raises(AudioContractError, match="enabled"):
        StreamDitherQuantizer(QuantizationSettings(bits=16))
    with pytest.raises(AudioContractError, match="enabled"):
        StreamDitherQuantizer("invalid")


def test_batch_validation_does_not_advance_random_state():
    settings = QuantizationSettings(bits=16, dither=True, seed=112)
    q = StreamDitherQuantizer(settings)
    with pytest.raises(AudioLimitError, match="sample cap"):
        q.quantize(np.zeros((8, MAX_DITHER_SAMPLES_PER_CALL // 8 + 1),
                            dtype=np.float32))
    assert q.samples_processed == 0
    with pytest.raises(AudioContractError):
        q.quantize(np.array([np.nan], dtype=np.float32))
    assert q.samples_processed == 0
    samples = _samples(2, 30)
    assert q.encode(samples) == StreamDitherQuantizer(settings).encode(samples)


def test_invalid_wav_packets_do_not_advance_stream_seed(tmp_path):
    samples = _samples(2, 50)
    settings = QuantizationSettings(bits=16, dither=True, seed=451)
    a_path, b_path = tmp_path / "a.wav", tmp_path / "b.wav"
    with WavPacketWriter(a_path, WavPacketSpec(24000, 2, 50, settings)) as a:
        a.push(FramePacket(24000, 0, samples[:, :20]))
        with pytest.raises(AudioContractError, match="gap|overlap"):
            a.push(FramePacket(24000, 21, samples[:, 20:30]))
        assert a.frames_written == 20
        a.push(FramePacket(24000, 20, samples[:, 20:]))
        a.finish()
    with WavPacketWriter(b_path, WavPacketSpec(24000, 2, 50, settings)) as b:
        b.push(FramePacket(24000, 0, samples))
        b.finish()
    assert a_path.read_bytes() == b_path.read_bytes()


def test_dithered_wav_context_abort_leaves_no_destination(tmp_path):
    samples = _samples(1, 32)
    target = tmp_path / "not-published.wav"
    settings = QuantizationSettings(bits=24, dither=True, seed=777)
    with WavPacketWriter(
        target, WavPacketSpec(24000, 1, 32, settings)
    ) as writer:
        writer.push(FramePacket(24000, 0, samples))
    assert not target.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_pcg64_tpdf_v1_has_portable_golden_pcm_bytes():
    """Any unexpected NumPy/OS RNG or quantization drift breaks this SHA."""
    source = np.array([
        [-.75, -.125, 0.0, .125, .75],
        [.8, .2, -.2, -.8, .00001],
    ], dtype=np.float32)
    cfg = QuantizationSettings(bits=16, dither=True, seed=1483)
    bytes_pcm = StreamDitherQuantizer(cfg).encode(source)
    assert bytes_pcm.hex() == (
        "00a0676600f09919000066e60010999901600000"
    )
    assert hashlib.sha256(bytes_pcm).hexdigest() == (
        "a7dff6f4de002c9958126242563f100c"
        "2322a4d6a7352e6b60e3328423c82743"
    )
