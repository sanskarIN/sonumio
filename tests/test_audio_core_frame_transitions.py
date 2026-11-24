"""Exact adjacent-frame PCM continuity across arbitrary packet partitions."""
from __future__ import annotations

import hashlib
import json
import wave

import numpy as np
import pytest

from sonumio.audio_core.dither import pack_pcm
from sonumio.audio_core.errors import (
    AudioContractError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.stream_quality import inspect_wav_quality
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.frame_transitions import StreamingFrameTransitionAccumulator
from sonumio.cli import main


@pytest.mark.parametrize("packet_frames", [1, 2, 3, 7, 41])
@pytest.mark.parametrize("channels", [1, 2, 4])
def test_adjacent_steps_are_packet_partition_invariant(packet_frames, channels):
    source = np.stack([
        np.array([0., .125, .25, -.75, -.875, .5, .5, 0., -.5, .75, .1],
                 dtype=np.float32) * (1. - j * .125)
        for j in range(channels)
    ])
    ref = np.abs(np.diff(source.astype(np.float64), axis=1))
    acc = StreamingFrameTransitionAccumulator(
        8000, channels, expected_frames=source.shape[1],
        max_allowed_step=.8,
    )
    for start in range(0, source.shape[1], packet_frames):
        acc.update(FramePacket(
            8000, start, source[:, start:start + packet_frames],
        ))
    r = acc.finalize()
    assert r.frames == source.shape[1]
    assert r.transitions_per_channel == source.shape[1] - 1
    assert r.max_adjacent_step == pytest.approx(float(np.max(ref)))
    assert r.channel_max_steps == pytest.approx(tuple(np.max(ref, axis=1)))
    assert r.over_limit_samples == int(np.count_nonzero(ref > .8))
    assert r.as_dict()["over_limit_fraction"] == pytest.approx(
        np.count_nonzero(ref > .8) / ref.size
    )
    assert r.passed is (r.over_limit_samples == 0)
    assert r.as_dict()["audible_click_proven"] is False


def test_single_frame_and_duplicate_finalization_are_well_defined():
    acc = StreamingFrameTransitionAccumulator(
        8000, 2, expected_frames=1, max_allowed_step=0,
    )
    acc.update(FramePacket(8000, 0, np.array([[.8], [-.8]], dtype=np.float32)))
    result = acc.finalize()
    assert result.transitions_per_channel == 0
    assert result.max_adjacent_step == 0
    assert result.channel_max_steps == (0.0, 0.0)
    assert result.over_limit_samples == 0
    assert result.as_dict()["over_limit_fraction"] == 0.0
    assert result.passed
    with pytest.raises(AudioContractError, match="already finalized"):
        acc.finalize()
    with pytest.raises(AudioContractError, match="finalized"):
        acc.update(FramePacket(8000, 1, np.zeros((2, 1), dtype=np.float32)))


def test_rejects_gaps_duplicates_and_missing_frames_without_state_mutation():
    acc = StreamingFrameTransitionAccumulator(8000, 1, expected_frames=4)
    acc.update(FramePacket(8000, 0, np.zeros(2, dtype=np.float32)))
    for invalid in (
        FramePacket(8000, 0, np.zeros(2, dtype=np.float32)),
        FramePacket(8000, 3, np.zeros(1, dtype=np.float32)),
        FramePacket(16000, 2, np.zeros(1, dtype=np.float32)),
        FramePacket(8000, 2, np.zeros((2, 1), dtype=np.float32)),
    ):
        with pytest.raises(AudioContractError, match="consecutive"):
            acc.update(invalid)
        assert acc.frames == 2
    with pytest.raises(AudioIntegrityError, match="Incomplete"):
        acc.finalize()
    with pytest.raises(AudioLimitError, match="duration"):
        acc.update(FramePacket(8000, 2, np.zeros(3, dtype=np.float32)))
    assert acc.frames == 2
    acc.update(FramePacket(8000, 2, np.zeros(2, dtype=np.float32)))
    assert acc.finalize().frames == 4


@pytest.mark.parametrize("invalid", [True, -1, 0, 8000*3600+1])
def test_rejects_invalid_expected_frames(invalid):
    with pytest.raises(AudioLimitError):
        StreamingFrameTransitionAccumulator(
            8000, 1, expected_frames=invalid,
        )


@pytest.mark.parametrize("invalid", [True, -1.0, 2.01, float("nan"), float("inf"), "0.2"])
def test_rejects_invalid_step_policy(invalid):
    with pytest.raises(AudioContractError):
        StreamingFrameTransitionAccumulator(
            8000, 1, max_allowed_step=invalid,
        )


def _make_rail_wave(tmp_path, bits):
    full = 1 << (bits - 1)
    pcm = np.array([[-full, -full, full - 1, full - 1, 0, 0]], dtype=np.int64)
    path = tmp_path / f"steps-{bits}.wav"
    with wave.open(str(path), "wb") as writer:
        writer.setframerate(8000)
        writer.setnchannels(1)
        writer.setsampwidth(bits // 8)
        writer.writeframes(pack_pcm(pcm, bits=bits))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
@pytest.mark.parametrize("chunk", [1, 2, 6])
def test_verified_wav_continuity_measures_exact_cross_packet_step(
    tmp_path, bits, chunk
):
    path, digest = _make_rail_wave(tmp_path, bits)
    result = inspect_wav_quality(
        path, expected_sha256=digest, packet_frames=chunk,
        include_continuity=True, max_adjacent_step=1.1,
    )
    continuity = result.frame_continuity
    assert continuity is not None
    assert continuity.transitions_per_channel == 5
    assert continuity.max_adjacent_step > 1.9
    assert continuity.over_limit_samples == 1
    assert continuity.passed is False
    assert result.inspection_policy_passed is False
    report = result.as_dict()
    assert report["source_sha256"] == digest
    assert report["source_verified"] is True
    assert report["inspection_controls"]["continuity_enabled"] is True
    assert report["inspection_controls"]["max_adjacent_step"] == .9
    assert report["frame_continuity"]["over_limit_samples"] == 1
    assert report["frame_continuity"]["audible_click_proven"] is False


def test_continuity_is_opt_in_and_bad_sha_never_emits_json(tmp_path, capsys):
    path, digest = _make_rail_wave(tmp_path, 16)
    quiet = inspect_wav_quality(path).as_dict()
    assert quiet["inspection_controls"]["continuity_enabled"] is False
    assert "frame_continuity" not in quiet
    with_threshold = inspect_wav_quality(
        path, max_adjacent_step=1.1,
    ).as_dict()
    assert with_threshold["inspection_controls"]["continuity_enabled"] is True
    assert with_threshold["frame_continuity"]["passed"] is False

    assert main([
        "wav-inspect", str(path), "--continuity",
        "--expected-sha256", "0" * 64,
    ]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "SHA256" in out.err
    assert digest != "0" * 64


def test_cli_step_policy_is_explicit_and_preserves_observation_mode(tmp_path, capsys):
    path, _ = _make_rail_wave(tmp_path, 16)
    args = [
        "wav-inspect", str(path),
        "--max-peak", "1", "--max-clipping-fraction", "1",
        "--max-dc-absolute", "1", "--max-boundary-jump", "10",
        "--max-channel-imbalance-db", "120", "--min-active-fraction", "0",
    ]
    assert main(args + ["--continuity"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert "frame_continuity" in info
    assert info["frame_continuity"]["max_allowed_step"] is None

    assert main(args + ["--max-adjacent-step", "1.1"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["inspection_policy_passed"] is False

    assert main(args + [
        "--max-adjacent-step", "1.1", "--fail-on-warnings",
    ]) == 1
    info = json.loads(capsys.readouterr().out)
    assert info["frame_continuity"]["passed"] is False
    assert info["frame_continuity"]["over_limit_samples"] == 1

    assert main(args + [
        "--max-adjacent-step", "2.0", "--fail-on-warnings",
    ]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["frame_continuity"]["passed"] is True

    assert main(args + ["--max-adjacent-step", "nan"]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "max_adjacent_step" in out.err
