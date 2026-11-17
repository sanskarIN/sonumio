"""Strict WAVE byte transport, PCM sample integrity and resampling tests."""
from __future__ import annotations

from dataclasses import replace
import io
import math
import os
from pathlib import Path
import struct
import wave

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.waveio import (
    WavInfo, decode_wav_bytes, encode_wav_bytes, inspect_wav_bytes,
    read_wav_file, write_wav_file, wave_metadata,
)
from sonumio.audio_core.resample import (
    ResampleSettings, linear_resample, output_frames, resample_audio,
    resample_ratio, resample_report, sinc_resample,
)
from sonumio.audio_core.errors import (
    AudioContractError, AudioFormatError, AudioLimitError,
)


@pytest.fixture
def sample():
    t = np.arange(4410, dtype=np.float32) / 44100
    return make_buffer(np.stack((
        .5 * np.sin(2 * np.pi * 440 * t),
        .25 * np.sin(2 * np.pi * 660 * t)
    )).astype(np.float32), 44100)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_pcm_wav_full_multichannel_roundtrip(sample, bits):
    encoded = encode_wav_bytes(sample, settings=QuantizationSettings(bits=bits))
    info, view = inspect_wav_bytes(encoded)
    assert info.sample_rate == 44100
    assert info.channels == 2
    assert info.frames == sample.frames
    assert info.bits_per_sample == bits
    assert info.data_bytes == sample.frames * 2 * bits // 8
    assert bytes(view) == encoded[-info.data_bytes:]
    decoded = decode_wav_bytes(encoded)
    assert decoded.samples.shape == sample.samples.shape
    assert decoded.sample_rate == sample.sample_rate
    np.testing.assert_allclose(decoded.samples, sample.samples,
                               atol=2 / (1 << (bits - 1)))


def test_wav_is_python_wave_compatible_and_byte_count_is_correct(sample):
    result = encode_wav_bytes(sample)
    with wave.open(io.BytesIO(result), "rb") as fp:
        assert fp.getnchannels() == 2
        assert fp.getframerate() == 44100
        assert fp.getsampwidth() == 2
        assert fp.getnframes() == sample.frames
    declared = struct.unpack_from("<I", result, 4)[0]
    assert declared + 8 == len(result)
    metadata = wave_metadata(sample, QuantizationSettings())
    assert metadata["estimated_payload_bytes"] == sample.frames * 4
    assert metadata["original_neural_model"] == "none"


def test_extra_chunk_is_skipped_when_still_frame_aligned(sample):
    data = encode_wav_bytes(sample)
    chunk = b"JUNK" + struct.pack("<I", 3) + b"123" + b"\x00"
    modified = data[:36] + chunk + data[36:]
    modified = modified[:4] + struct.pack("<I", len(modified) - 8) + modified[8:]
    info, _ = inspect_wav_bytes(modified)
    assert "JUNK" in info.ignored_chunks


def test_invalid_wav_headers_duplicate_fmt_and_data_chunks_rejected(sample):
    data = encode_wav_bytes(sample)
    broken = [
        data[:4] + b"\x00\x00\x00\x00" + data[8:],
        b"RIFX" + data[4:],
        data[:4] + struct.pack("<I", 2) + data[8:],
        data[:-1],
        data[:12] + data[12:36] + data[12:],
    ]
    for corrupted in broken:
        with pytest.raises((AudioFormatError, AudioLimitError)):
            inspect_wav_bytes(corrupted)


def test_unsupported_compressed_audio_is_not_guessed_as_pcm(sample):
    data = bytearray(encode_wav_bytes(sample))
    data[20:22] = struct.pack("<H", 3)
    with pytest.raises(AudioFormatError, match="Only"):
        inspect_wav_bytes(bytes(data))
    data = bytearray(encode_wav_bytes(sample))
    data[34:36] = struct.pack("<H", 20)
    with pytest.raises(AudioFormatError):
        inspect_wav_bytes(bytes(data))


def test_atomic_wav_output_overwrite_opt_in_and_no_symlinks(sample, tmp_path):
    path = tmp_path / "audio.wav"
    written = write_wav_file(sample, path)
    assert written == path
    decoded = read_wav_file(path)
    assert decoded.sample_rate == sample.sample_rate
    with pytest.raises(AudioFormatError, match="overwrite"):
        write_wav_file(sample, path)
    path.write_bytes(b"intermediate")
    write_wav_file(sample, path, overwrite=True)
    assert read_wav_file(path).frames == sample.frames
    link = tmp_path / "alias.wav"
    link.symlink_to(path)
    with pytest.raises(AudioFormatError):
        write_wav_file(sample, link)
    with pytest.raises(AudioFormatError):
        read_wav_file(link)


def test_wav_without_parent_creation_fails_closed(sample, tmp_path):
    path = tmp_path / "sub" / "audio.wav"
    with pytest.raises(AudioFormatError):
        write_wav_file(sample, path)
    write_wav_file(sample, path, create_parent=True)
    assert path.is_file()
    with pytest.raises(AudioFormatError):
        write_wav_file(sample, tmp_path / "raw.bin")


@pytest.mark.parametrize("source,target,frames,expected", [
    (24000, 8000, 120, 40),
    (24000, 48000, 100, 200),
    (44100, 48000, 441, 480),
    (48000, 48000, 128, 128),
])
def test_integer_ratio_frame_count(source, target, frames, expected):
    assert output_frames(frames, source, target) == expected


def test_sinc_resampling_preserves_constant_dc_level_and_channels():
    ones = np.full((2, 2400), .25, dtype=np.float32)
    converted = sinc_resample(ones, 24000, 8000,
                              settings=ResampleSettings(8000, lobes=24))
    assert converted.shape == (2, 800)
    np.testing.assert_allclose(converted, .25, atol=1e-5)
    assert np.isfinite(converted).all()


def test_bounded_sinc_downsampling_reduces_above_nyquist_tone():
    sample_rate = 24000
    source = np.sin(2 * np.pi * 9000 * np.arange(2400) / sample_rate)
    filtered = sinc_resample(source.astype(np.float32), sample_rate, 8000,
                             settings=ResampleSettings(8000, lobes=48))
    raw = linear_resample(source.astype(np.float32), sample_rate, 8000,
                          allow_aliasing=True)
    assert np.sqrt(np.mean(filtered[:, 100:-100] ** 2)) < .15
    assert np.sqrt(np.mean(raw[:, 100:-100] ** 2)) > .5


def test_linear_downsampling_needs_explicit_aliasing_permission():
    mono = np.linspace(-1, 1, 1024, dtype=np.float32)
    with pytest.raises(AudioContractError, match="aliases"):
        linear_resample(mono, 44100, 16000)
    upsampled = linear_resample(mono, 16000, 32000)
    assert upsampled.shape == (1, 2048)
    np.testing.assert_allclose(upsampled[0, ::2], mono, atol=1e-7)


def test_full_resample_buffer_and_report_preserve_source(sample):
    options = ResampleSettings(24000, lobes=24)
    result = resample_audio(sample, options)
    assert result.frames == 2400
    assert result.channels == 2
    assert result.sample_rate == 24000
    assert sample.frames == 4410
    report = resample_report(sample, result, options)
    assert report["anti_alias_filter_applied"] is True
    assert report["neural_training_data_modified"] is False
    with pytest.raises(AudioContractError):
        resample_audio(sample, ResampleSettings(24000, method="linear"))


def test_resample_ratio_rejects_fractional_output_sample_clocks():
    buffer = make_buffer(np.arange(128, dtype=np.float32), 44100)
    with pytest.raises(AudioContractError, match="noninteger"):
        resample_ratio(buffer, 1, 8)
    other = resample_ratio(buffer, 1, 2)
    assert other.sample_rate == 22050
    assert other.frames == 64


def test_resample_controls_and_invalid_ratios_rejected(sample):
    with pytest.raises(AudioContractError):
        ResampleSettings(24000, method="nearest")
    with pytest.raises(AudioContractError):
        ResampleSettings(24000, lobes=0)
    with pytest.raises(AudioContractError):
        ResampleSettings(24000, beta=float("nan"))
    with pytest.raises(AudioContractError):
        sinc_resample(sample.samples, 44100, 24000,
                      settings=ResampleSettings(16000))
    for values in ((0, 1), (1, 0), (100, 1), (1, 100)):
        with pytest.raises(AudioContractError):
            resample_ratio(sample, *values)
