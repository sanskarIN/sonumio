"""Trim context and raised-cosine fade invariants for hash-verified local PCM WAV."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_pipeline import WavProcessingConfig, process_wav_file
from sonumio.audio_core.wave_reader import WavPacketReader
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.cli import main


def _source(tmp_path, samples):
    array = np.asarray(samples, dtype=np.float32)
    path = tmp_path / "source.wav"
    with WavPacketWriter(
        path, WavPacketSpec(
            24000, array.shape[0], array.shape[1],
            QuantizationSettings(bits=16),
        ),
    ) as writer:
        writer.push(FramePacket(24000, 0, array))
        result = writer.finish()
    return path, result.wav_sha256


def _decode(path, packet_frames=3):
    with WavPacketReader(path, packet_frames=packet_frames) as reader:
        samples = np.concatenate([p.samples for p in reader], axis=1)
        assert reader.verified_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    return samples


@pytest.mark.parametrize("packet_frames", [1, 2, 7, 45])
def test_trim_handles_preserve_original_multichannel_alignment(tmp_path, packet_frames):
    signal = np.zeros((2, 45), dtype=np.float32)
    signal[0, 10:14] = 0.4
    signal[1, 12:17] = -0.25
    source, source_sha = _source(tmp_path, signal)
    output = tmp_path / f"trim-{packet_frames}.wav"
    report = process_wav_file(
        source, output, expected_source_sha256=source_sha,
        config=WavProcessingConfig(
            trim_silence_dbfs=-45, trim_pre_roll_frames=4,
            trim_post_roll_frames=6, packet_frames=packet_frames,
        ),
    )
    assert (report.retained_start_frame, report.retained_end_frame) == (6, 23)
    assert report.frames == 17 and report.source_frames == 45
    assert (report.trim_pre_roll_frames, report.trim_post_roll_frames) == (4, 6)
    assert report.output_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    np.testing.assert_allclose(_decode(output), signal[:, 6:23], atol=1 / 32768, rtol=0)
    if packet_frames > 1:
        assert output.read_bytes() == (tmp_path / "trim-1.wav").read_bytes()


def test_trim_handles_clip_at_source_boundaries_without_padding(tmp_path):
    signal = np.zeros((1, 31), dtype=np.float32)
    signal[0, 2] = 0.5
    signal[0, 28] = -0.5
    source, original_sha = _source(tmp_path, signal)
    report = process_wav_file(
        source, tmp_path / "all.wav", config=WavProcessingConfig(
            trim_silence_dbfs=-30, trim_pre_roll_frames=9,
            trim_post_roll_frames=8,
        ),
    )
    assert (report.retained_start_frame, report.retained_end_frame) == (0, 31)
    assert report.output_sha256 == original_sha
    assert report.frames == 31


@pytest.mark.parametrize("packet_frames", [1, 3, 8, 15])
def test_cosine_fade_uses_absolute_frames_and_stable_bytes(tmp_path, packet_frames):
    source, sha = _source(tmp_path, np.full((1, 15), 0.5, dtype=np.float32))
    output = tmp_path / f"cos-{packet_frames}.wav"
    report = process_wav_file(
        source, output, expected_source_sha256=sha,
        config=WavProcessingConfig(
            fade_in_frames=5, fade_out_frames=7,
            fade_curve="cosine", packet_frames=packet_frames,
        ),
    )
    frame = np.arange(15, dtype=np.float64)
    progress_in = np.minimum(frame / 4, 1)
    progress_out = np.minimum((14 - frame) / 6, 1)
    expected = (0.5 * (0.5 - 0.5 * np.cos(np.pi * progress_in)) *
                (0.5 - 0.5 * np.cos(np.pi * progress_out)))
    np.testing.assert_allclose(_decode(output)[0], expected, atol=1 / 32768, rtol=0)
    assert report.fade_curve == "cosine"
    assert report.output_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    if packet_frames > 1:
        assert output.read_bytes() == (tmp_path / "cos-1.wav").read_bytes()


def test_overlapping_cosine_fades_multiply_instead_of_adding(tmp_path):
    source, _ = _source(tmp_path, np.full((1, 7), 0.5, dtype=np.float32))
    report = process_wav_file(
        source, tmp_path / "overlap.wav", config=WavProcessingConfig(
            fade_in_frames=6, fade_out_frames=6, fade_curve="cosine",
        ),
    )
    frame = np.arange(7, dtype=np.float64)
    left = 0.5 - 0.5 * np.cos(np.pi * np.minimum(frame / 5, 1))
    right = 0.5 - 0.5 * np.cos(np.pi * np.minimum((6 - frame) / 5, 1))
    np.testing.assert_allclose(_decode(report.destination)[0],
                               0.5 * left * right, atol=1 / 32768, rtol=0)


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "12", None, 1920001])
def test_invalid_context_lengths_fail_before_processing(invalid):
    with pytest.raises(AudioContractError, match="trim_pre_roll_frames"):
        WavProcessingConfig(trim_silence_dbfs=-30, trim_pre_roll_frames=invalid)
    with pytest.raises(AudioContractError, match="trim_post_roll_frames"):
        WavProcessingConfig(trim_silence_dbfs=-30, trim_post_roll_frames=invalid)


@pytest.mark.parametrize("invalid", [None, True, "raised", "cosine^2", 1])
def test_invalid_fade_curve_fails_before_processing(invalid):
    with pytest.raises(AudioContractError, match="fade_curve"):
        WavProcessingConfig(fade_curve=invalid)


def test_context_requires_threshold_and_preserves_positional_constructor():
    with pytest.raises(AudioContractError, match="require trim_silence_dbfs"):
        WavProcessingConfig(trim_pre_roll_frames=1)
    with pytest.raises(AudioContractError, match="require trim_silence_dbfs"):
        WavProcessingConfig(trim_post_roll_frames=1)
    legacy = WavProcessingConfig(
        0.0, None, False, False, None, "hard", 7, 24, None, 3, 4,
    )
    assert legacy.fade_in_frames == 3 and legacy.fade_out_frames == 4
    assert legacy.fade_curve == "linear"
    assert legacy.trim_pre_roll_frames == legacy.trim_post_roll_frames == 0


def test_silent_trim_with_context_fails_before_destination_created(tmp_path):
    source, _ = _source(tmp_path, np.zeros((1, 23), dtype=np.float32))
    dest = tmp_path / "new-folder" / "no.wav"
    with pytest.raises(AudioLimitError, match="No WAV frames"):
        process_wav_file(
            source, dest, create_parent=True,
            config=WavProcessingConfig(
                trim_silence_dbfs=-40, trim_pre_roll_frames=7,
                trim_post_roll_frames=7,
            ),
        )
    assert not dest.parent.exists()
    assert not list(tmp_path.rglob(".sonumio-stream-*.partial"))


def test_cli_reports_context_and_cosine_curve_with_verified_hash(tmp_path, capsys):
    signal = np.zeros((1, 27), dtype=np.float32)
    signal[0, 8:19] = 0.5
    source, sha = _source(tmp_path, signal)
    dest = tmp_path / "cli.wav"
    assert main([
        "wav-process", str(source), "--output", str(dest),
        "--expected-sha256", sha,
        "--trim-silence-dbfs", "-40", "--trim-pre-roll-frames", "3",
        "--trim-post-roll-frames", "4",
        "--fade-in-frames", "5", "--fade-out-frames", "7",
        "--fade-curve", "cosine", "--packet-frames", "2",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["retained_start_frame"] == 5
    assert report["retained_end_frame"] == 23
    assert report["frames"] == 18
    assert report["trim_pre_roll_frames"] == 3
    assert report["trim_post_roll_frames"] == 4
    assert report["fade_curve"] == "cosine"
    assert report["source_sha256"] == sha
    assert report["output_sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
