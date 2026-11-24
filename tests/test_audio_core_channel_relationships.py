"""Independent offline reference and verified integration for channel metrics."""
from __future__ import annotations

import hashlib
import json
import math
import wave

import numpy as np
import pytest

from sonumio.audio_core import (
    FramePacket, StreamingChannelRelationshipAccumulator, inspect_wav_quality,
)
from sonumio.audio_core.errors import (
    AudioContractError, AudioIntegrityError, AudioLimitError,
)
from sonumio.cli import main


def accumulate(samples, sizes, rate=8000):
    meter = StreamingChannelRelationshipAccumulator(
        rate, samples.shape[0], expected_frames=samples.shape[1],
    )
    start = 0
    for size in sizes:
        stop = min(start + size, samples.shape[1])
        if start == stop:
            break
        meter.update(FramePacket(rate, start, samples[:, start:stop]))
        start = stop
    assert start == samples.shape[1]
    return meter.finalize()


def verify_against_offline(report, samples):
    data = samples.astype(np.float64)
    channels, frames = data.shape
    assert report.channels == channels
    assert report.frames == frames
    assert report.channel_rms == pytest.approx(np.sqrt(np.mean(data ** 2, axis=1)))
    assert report.channel_dc == pytest.approx(np.mean(data, axis=1), abs=1e-12)
    expected_pairs = [(i, j) for i in range(channels) for j in range(i + 1, channels)]
    assert [(p.first_channel, p.second_channel) for p in report.pairs] == expected_pairs
    for p in report.pairs:
        a, b = data[p.first_channel], data[p.second_channel]
        expected_reference = float(np.sqrt(np.mean((a ** 2 + b ** 2) / 2)))
        expected_mix = float(np.sqrt(np.mean(((a + b) / 2) ** 2)))
        assert p.reference_rms == pytest.approx(expected_reference, abs=1e-12)
        assert p.equal_mix_rms == pytest.approx(expected_mix, abs=1e-12)
        if expected_reference:
            assert p.equal_mix_retention == pytest.approx(expected_mix / expected_reference, abs=1e-12)
        else:
            assert p.equal_mix_retention is None
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            assert p.centered_correlation is None
        else:
            expected = float(np.corrcoef(a, b)[0, 1])
            assert p.centered_correlation == pytest.approx(expected, abs=1e-9)
    exported = report.as_dict()
    assert not exported["physical_channel_layout_inferred"]
    assert not exported["perceptual_audio_quality_measured"]
    assert not exported["phase_inversion_proven"]
    # Reject accidental NaN/Infinity JSON serialization.
    json.dumps(exported, allow_nan=False)


@pytest.mark.parametrize("channels", [1, 2, 3, 8])
@pytest.mark.parametrize("frames", [1, 2, 17, 193])
def test_random_partitions_and_offline_oracle(channels, frames):
    rng = np.random.default_rng(channels * 100 + frames)
    data = rng.normal(0, .1, (channels, frames)).astype(np.float32)
    for sizes in ([1] * frames, [7] * frames, [frames],
                  rng.integers(1, 20, size=frames).tolist()):
        verify_against_offline(accumulate(data, sizes), data)


@pytest.mark.parametrize("chunk", [1, 3, 7, 127])
def test_identical_opposite_orthogonal_and_silent_channels(chunk):
    a = np.tile(np.array([.5, -.5, .5, -.5], np.float32), 31)
    b = np.tile(np.array([.5, .5, -.5, -.5], np.float32), 31)
    data = np.stack([a, a, -a, b, np.zeros_like(a)])
    report = accumulate(data, [chunk] * len(a))
    verify_against_offline(report, data)
    pairs = {(p.first_channel, p.second_channel): p for p in report.pairs}
    assert pairs[0, 1].centered_correlation == pytest.approx(1)
    assert pairs[0, 1].equal_mix_retention == pytest.approx(1)
    assert pairs[0, 2].centered_correlation == pytest.approx(-1)
    assert pairs[0, 2].equal_mix_rms == 0
    assert pairs[0, 2].equal_mix_retention == 0
    assert pairs[0, 3].centered_correlation == pytest.approx(0, abs=1e-12)
    assert pairs[0, 4].centered_correlation is None
    assert pairs[0, 4].equal_mix_retention == pytest.approx(1 / math.sqrt(2))


@pytest.mark.parametrize("chunk", [1, 11, 128])
def test_large_dc_small_variations_and_near_cancellation(chunk):
    x = np.tile(np.array([-3, -1, 1, 3], np.float32), 32) * 2 ** -19
    a = .75 + x
    b = -.75 - x + 2 ** -22
    c = .5 - x
    data = np.stack([a, b, c]).astype(np.float32)
    report = accumulate(data, [chunk] * 128)
    verify_against_offline(report, data)
    pair = report.pairs[0]
    assert pair.equal_mix_rms > 0
    assert pair.equal_mix_retention < 1e-5
    assert pair.centered_correlation == pytest.approx(-1)


@pytest.mark.parametrize("value", [0., .125, -.9])
def test_constant_and_zero_pairs_have_undefined_correlation(value):
    data = np.full((2, 19), value, dtype=np.float32)
    for chunk in (1, 3, 19):
        report = accumulate(data, [chunk] * 19)
        verify_against_offline(report, data)
        assert report.pairs[0].centered_correlation is None
        if value == 0:
            assert report.pairs[0].equal_mix_retention is None


@pytest.mark.parametrize("rate,channels", [(7999, 2), (True, 2), (8000, 0),
                                             (8000, 9), (8000, True)])
def test_invalid_geometry(rate, channels):
    with pytest.raises(AudioContractError):
        StreamingChannelRelationshipAccumulator(rate, channels)


@pytest.mark.parametrize("frames", [0, -1, True, 8000 * 3600 + 1])
def test_invalid_expected_frames(frames):
    with pytest.raises(AudioLimitError):
        StreamingChannelRelationshipAccumulator(8000, 2, expected_frames=frames)


def test_rejects_incomplete_and_nonconsecutive_stream():
    meter = StreamingChannelRelationshipAccumulator(8000, 2, expected_frames=4)
    data = np.zeros((2, 2), np.float32)
    with pytest.raises(AudioIntegrityError):
        meter.finalize()
    meter.update(FramePacket(8000, 0, data))
    for packet in (
        FramePacket(8000, 0, data),
        FramePacket(8000, 3, data),
        FramePacket(16000, 2, data),
        FramePacket(8000, 2, data[:1]),
    ):
        with pytest.raises(AudioContractError):
            meter.update(packet)
    with pytest.raises(AudioLimitError):
        meter.update(FramePacket(8000, 2, np.zeros((2, 3), np.float32)))
    assert meter.frames == 2
    with pytest.raises(AudioIntegrityError):
        meter.finalize()
    meter.update(FramePacket(8000, 2, data))
    assert meter.finalize().frames == 4
    with pytest.raises(AudioContractError):
        meter.finalize()
    with pytest.raises(AudioContractError):
        meter.update(FramePacket(8000, 4, data))


def make_wave(tmp_path):
    path = tmp_path / "opposite.wav"
    a = np.array([0, 4096, -4096, 2048, -2048, 0], dtype=np.int16)
    pcm = np.stack([a, -a], axis=1).astype("<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(pcm.tobytes())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("chunk", [1, 2, 3, 8192])
def test_verified_wav_optional_combined_report(tmp_path, chunk):
    path, digest = make_wave(tmp_path)
    baseline = inspect_wav_quality(path)
    assert baseline.channel_relationships is None
    assert not baseline.as_dict()["inspection_controls"]["channel_relationships_enabled"]
    report = inspect_wav_quality(
        path, packet_frames=chunk, expected_sha256=digest,
        include_channel_relationships=True, include_timeline=True,
        include_continuity=True, include_quiet_intervals=True,
    )
    assert report.inspection_policy_passed == baseline.inspection_policy_passed
    exported = report.as_dict()
    assert exported["source_sha256"] == digest
    assert exported["source_verified"]
    assert exported["inspection_controls"]["channel_relationships_enabled"]
    p = exported["channel_relationships"]["pairs"][0]
    assert p["centered_correlation"] == pytest.approx(-1)
    assert p["equal_mix_retention"] == 0
    assert exported["quiet_intervals"]["quiet_frames"] == 2


def test_cli_observation_and_bad_sha_never_publishes_json(tmp_path, capsys):
    path, digest = make_wave(tmp_path)
    args = ["wav-inspect", str(path), "--channel-relationships"]
    assert main(args + ["--expected-sha256", digest, "--fail-on-warnings"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["channel_relationships"]["pairs"][0]["equal_mix_retention"] == 0
    assert main(args + ["--expected-sha256", "0" * 64]) == 2
    result = capsys.readouterr()
    assert not result.out
    assert "SHA256" in result.err
    with pytest.raises(AudioContractError, match="include_channel_relationships"):
        inspect_wav_quality(path, include_channel_relationships=1)
