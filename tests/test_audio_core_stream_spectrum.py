"""Independent DFT, Parseval and verified-file tests for the spectral engine."""
from __future__ import annotations

import hashlib
import json
import wave

import numpy as np
import pytest

from sonumio.audio_core.errors import (
    AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.stream_spectrum import (
    SpectrumConfig, StreamingSpectrumAccumulator, inspect_wav_spectrum,
)


def accumulate(data, sizes, n=64, rate=8000):
    meter = StreamingSpectrumAccumulator(
        rate, data.shape[0], config=SpectrumConfig(n),
        expected_frames=data.shape[1],
    )
    start = 0
    for size in sizes:
        stop = min(start + size, data.shape[1])
        if start == stop:
            break
        meter.update(FramePacket(rate, start, data[:, start:stop]))
        start = stop
    assert start == data.shape[1]
    return meter.finalize()


def direct_dft_reference(data, n):
    """Independent complex exponential matrix, no FFT routine."""
    original = data.shape[1]
    padded = np.pad(data.astype(np.float64), ((0, 0), (0, (-original) % n)))
    kernel = np.exp(-2j * np.pi * np.outer(np.arange(n // 2 + 1), np.arange(n)) / n)
    power = np.zeros((data.shape[0], n // 2 + 1))
    for start in range(0, padded.shape[1], n):
        power += np.abs(padded[:, start:start+n] @ kernel.T) ** 2
    power[:, 1:-1] *= 2
    return power / (n * original)


@pytest.mark.parametrize("channels", [1, 2, 8])
@pytest.mark.parametrize("frames", [1, 63, 64, 65, 129, 331])
def test_direct_dft_parseval_and_partition_invariance(channels, frames):
    rng = np.random.default_rng(42 + frames + channels)
    data = rng.normal(0, .25, (channels, frames)).astype(np.float32)
    reference = direct_dft_reference(data, 64)
    time_energy = np.mean(data.astype(np.float64) ** 2, axis=1)
    for sizes in ([1] * frames, [7] * frames, [63] * frames, [frames],
                  rng.integers(1, 89, size=frames).tolist()):
        report = accumulate(data, sizes)
        assert report.frames == frames
        assert report.complete_blocks == frames // 64
        assert report.padded_tail_frames == frames % 64
        assert report.blocks == (frames + 63) // 64
        for i, channel in enumerate(report.channel_spectra):
            np.testing.assert_allclose(channel.bin_mean_square, reference[i],
                                       atol=1e-14, rtol=1e-11)
            assert channel.mean_square == pytest.approx(time_energy[i], rel=1e-12)
            # Equal-energy bins may differ at roundoff scale between FFT and DFT.
            assert reference[i, channel.dominant_bin] == pytest.approx(
                float(np.max(reference[i])), rel=1e-11, abs=1e-14
            )
            assert channel.dominant_frequency_hz == channel.dominant_bin * 125
            assert channel.centroid_hz == pytest.approx(
                np.dot(reference[i], np.arange(33) * 125) / time_energy[i]
            )
        json.dumps(report.as_dict(include_bins=True), allow_nan=False)


@pytest.mark.parametrize("n", [64, 256, 1024, 8192])
def test_exact_bin_tone_dc_nyquist_and_silence(n):
    t = np.arange(n)
    tone = .5 * np.sin(2 * np.pi * 5 * t / n)
    dc = np.full(n, .25)
    nyquist = .5 * (-1.) ** t
    data = np.stack([tone, dc, nyquist, np.zeros(n)]).astype(np.float32)
    report = accumulate(data, [31] * n, n=n, rate=48000)
    a, b, c, d = report.channel_spectra
    assert a.dominant_bin == 5
    assert a.dominant_frequency_hz == 5 * 48000 / n
    assert a.centroid_hz == pytest.approx(a.dominant_frequency_hz)
    assert a.mean_square == pytest.approx(.125)
    assert b.dominant_bin == 0
    assert b.dc_bin_fraction == 1
    assert b.centroid_hz == 0
    assert c.dominant_bin == n // 2
    assert c.dominant_frequency_hz == 24000
    assert c.nyquist_bin_fraction == 1
    assert d.mean_square == 0
    assert d.dominant_bin is None
    assert d.dominant_frequency_hz is None
    assert d.centroid_hz is None
    assert d.dc_bin_fraction is None
    assert d.nyquist_bin_fraction is None
    assert all(v == 0 for v in d.bin_mean_square)


def test_one_sample_tail_preserves_original_frame_normalization():
    data = np.array([[.5]], dtype=np.float32)
    r = accumulate(data, [1], n=64)
    spectrum = r.channel_spectra[0]
    assert r.complete_blocks == 0
    assert r.padded_tail_frames == 1
    assert spectrum.mean_square == pytest.approx(.25)
    assert spectrum.bin_mean_square[0] == pytest.approx(.25 / 64)
    assert spectrum.bin_mean_square[1] == pytest.approx(.5 / 64)
    assert spectrum.bin_mean_square[-1] == pytest.approx(.25 / 64)


def test_batch_cap_and_large_packet_agree_with_smaller_packets():
    rng = np.random.default_rng(39)
    data = rng.normal(0, .1, (2, 140001)).astype(np.float32)
    large = accumulate(data, [140001], n=64)
    small = accumulate(data, [257] * 600, n=64)
    for a, b in zip(large.channel_spectra, small.channel_spectra):
        np.testing.assert_allclose(a.bin_mean_square, b.bin_mean_square, rtol=1e-12)
    assert large.complete_blocks == small.complete_blocks
    assert large.padded_tail_frames == small.padded_tail_frames


@pytest.mark.parametrize("n", [True, 0, 32, 63, 65, 8193, 16384, 64., "64"])
def test_invalid_fft_geometry(n):
    with pytest.raises(AudioContractError):
        SpectrumConfig(n)


@pytest.mark.parametrize("rate,channels", [(7999, 1), (192001, 1), (True, 1),
                                         (8000, 0), (8000, 9), (8000, True)])
def test_invalid_clock_layout(rate, channels):
    with pytest.raises(AudioContractError):
        StreamingSpectrumAccumulator(rate, channels)


@pytest.mark.parametrize("count", [True, 0, -1, 8000 * 3600 + 1])
def test_invalid_declared_duration(count):
    with pytest.raises(AudioLimitError):
        StreamingSpectrumAccumulator(8000, 1, expected_frames=count)


def test_incomplete_invalid_and_duplicate_lifecycle():
    meter = StreamingSpectrumAccumulator(
        8000, 1, config=SpectrumConfig(64), expected_frames=65,
    )
    with pytest.raises(AudioIntegrityError):
        meter.finalize()
    meter.update(FramePacket(8000, 0, np.zeros(64, np.float32)))
    with pytest.raises(AudioIntegrityError):
        meter.finalize()
    for packet in (
        FramePacket(8000, 0, np.zeros(1, np.float32)),
        FramePacket(8000, 65, np.zeros(1, np.float32)),
        FramePacket(16000, 64, np.zeros(1, np.float32)),
        FramePacket(8000, 64, np.zeros((2, 1), np.float32)),
    ):
        with pytest.raises(AudioContractError):
            meter.update(packet)
    with pytest.raises(AudioLimitError):
        meter.update(FramePacket(8000, 64, np.zeros(2, np.float32)))
    assert meter.frames == 64
    meter.update(FramePacket(8000, 64, np.ones(1, np.float32)))
    assert meter.finalize().padded_tail_frames == 1
    with pytest.raises(AudioContractError):
        meter.finalize()
    with pytest.raises(AudioContractError):
        meter.update(FramePacket(8000, 65, np.zeros(1, np.float32)))


def make_wave(tmp_path):
    path = tmp_path / "spectrum.wav"
    samples = np.array([0, 4096, 0, -4096] * 32 + [1024], dtype="<i2")
    pcm = np.stack([samples, -samples], axis=1).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(pcm.tobytes())
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), pcm.T / 32768.


@pytest.mark.parametrize("chunk", [1, 7, 64, 8192])
def test_verified_wav_reports_exact_geometry_and_energy(tmp_path, chunk):
    path, digest, pcm = make_wave(tmp_path)
    result = inspect_wav_spectrum(
        path, expected_sha256=digest, packet_frames=chunk,
        config=SpectrumConfig(64),
    )
    assert result.info.frames == 129
    assert result.info.channels == 2
    assert result.source_sha256 == digest
    plain = result.as_dict()
    full = result.as_dict(include_bins=True)
    assert plain["source_verified"]
    assert plain["expected_sha256_required"]
    assert plain["spectrum"]["window"] == "rectangular"
    assert "bin_mean_square" not in plain["spectrum"]["channel_spectra"][0]
    assert len(full["spectrum"]["channel_spectra"][0]["bin_mean_square"]) == 33
    ref = direct_dft_reference(pcm, 64)
    for i, spectrum in enumerate(result.spectrum.channel_spectra):
        np.testing.assert_allclose(spectrum.bin_mean_square, ref[i], atol=1e-14)
        assert spectrum.dominant_frequency_hz == 2000
    with pytest.raises(AudioContractError):
        result.as_dict(include_bins=1)


def test_wrong_digest_and_truncated_source_rejected(tmp_path):
    path, _, _ = make_wave(tmp_path)
    with pytest.raises(AudioIntegrityError, match="SHA256"):
        inspect_wav_spectrum(path, expected_sha256="0" * 64)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(AudioFormatError):
        inspect_wav_spectrum(path)
