"""Independent quiet-run oracle and verified-file/CLI regression tests."""
from __future__ import annotations

import hashlib
import itertools
import json
import wave

import numpy as np
import pytest

from sonumio.audio_core import (
    FramePacket, StreamingQuietIntervalAccumulator, inspect_wav_quality,
)
from sonumio.audio_core.errors import (
    AudioContractError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.quality import QualityLimits
from sonumio.cli import main


def oracle(data, floor_dbfs):
    quiet = [all(abs(float(sample)) <= 10 ** (floor_dbfs / 20)
                 for sample in frame) for frame in data.T]
    runs = []
    start = None
    for index, value in enumerate(quiet + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    internal = [(a, b) for a, b in runs if a > 0 and b < len(quiet)]
    longest = max(internal, key=lambda ab: ab[1] - ab[0], default=None)
    return (
        sum(quiet), len(runs),
        runs[0][1] if runs and runs[0][0] == 0 else 0,
        len(quiet) - runs[-1][0] if runs and runs[-1][1] == len(quiet) else 0,
        len(internal),
        longest[1] - longest[0] if longest else 0,
        longest[0] if longest else None,
    )


def accumulate(data, chunks, floor=-70, budget=None):
    meter = StreamingQuietIntervalAccumulator(
        8000, data.shape[0], expected_frames=data.shape[1],
        activity_floor_dbfs=floor, max_internal_quiet_frames=budget,
    )
    start = 0
    for size in chunks:
        stop = min(start + size, data.shape[1])
        if start == stop:
            break
        meter.update(FramePacket(8000, start, data[:, start:stop]))
        start = stop
    assert start == data.shape[1]
    return meter.finalize()


def assert_matches(report, expected):
    assert (
        report.quiet_frames, report.quiet_runs, report.leading_quiet_frames,
        report.trailing_quiet_frames, report.internal_quiet_runs,
        report.longest_internal_quiet_frames, report.longest_internal_start,
    ) == expected
    exported = report.as_dict()
    assert exported["quiet_fraction"] == report.quiet_frames / report.frames
    assert exported["longest_internal_stop"] == (
        None if expected[-1] is None else expected[-1] + expected[-2]
    )
    assert exported["audible_dropout_proven"] is False


def test_exhaustive_short_masks_and_every_fixed_packet_size():
    for length in range(1, 8):
        for values in itertools.product((0., .5), repeat=length):
            data = np.array([values], dtype=np.float32)
            expected = oracle(data, -70)
            for chunk in range(1, length + 1):
                report = accumulate(data, [chunk] * length, budget=2)
                assert_matches(report, expected)
                assert report.passed is (expected[-2] <= 2)


@pytest.mark.parametrize("channels", [1, 2, 8])
def test_random_multichannel_variable_partitions(channels):
    rng = np.random.default_rng(414)
    for length in (1, 2, 19, 113, 513):
        data = rng.choice(
            np.array([0., .01, -.01, .2, -.2], dtype=np.float32),
            size=(channels, length),
        )
        data[:, length // 4:length // 2] = 0
        expected = oracle(data, -30)
        for _ in range(5):
            chunks = rng.integers(1, 24, size=length).tolist()
            report = accumulate(data, chunks, floor=-30)
            assert_matches(report, expected)


def test_equal_length_runs_choose_earliest_and_exclude_edges():
    data = np.array([[0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]], np.float32)
    for chunk in (1, 2, 3, 12):
        result = accumulate(data, [chunk] * 12, budget=2)
        assert result.leading_quiet_frames == 2
        assert result.trailing_quiet_frames == 3
        assert result.internal_quiet_runs == 2
        assert result.longest_internal_start == 3
        assert result.passed
        assert result.as_dict()["longest_internal_seconds"] == 2 / 8000
    assert not accumulate(data, [12], budget=1).passed


def test_all_quiet_counts_one_run_and_no_internal_gap():
    result = accumulate(np.zeros((2, 32), np.float32), [3] * 32, budget=0)
    assert result.quiet_runs == 1
    assert result.leading_quiet_frames == result.trailing_quiet_frames == 32
    assert result.quiet_frames == 32
    assert result.internal_quiet_runs == 0
    assert result.longest_internal_start is None
    assert result.passed


@pytest.mark.parametrize("budget", [True, -1, 1.0, "1", float("nan")])
def test_invalid_budget(budget):
    with pytest.raises(AudioContractError, match="max_internal_quiet_frames"):
        StreamingQuietIntervalAccumulator(8000, 1, max_internal_quiet_frames=budget)


def test_rejects_missing_and_reordered_packets_without_mutation():
    acc = StreamingQuietIntervalAccumulator(8000, 1, expected_frames=4)
    with pytest.raises(AudioIntegrityError):
        acc.finalize()
    acc.update(FramePacket(8000, 0, np.zeros(2, np.float32)))
    with pytest.raises(AudioIntegrityError):
        acc.finalize()
    for packet in (
        FramePacket(8000, 0, np.zeros(2, np.float32)),
        FramePacket(8000, 3, np.zeros(1, np.float32)),
        FramePacket(16000, 2, np.zeros(1, np.float32)),
        FramePacket(8000, 2, np.zeros((2, 1), np.float32)),
    ):
        with pytest.raises(AudioContractError):
            acc.update(packet)
    with pytest.raises(AudioLimitError):
        acc.update(FramePacket(8000, 2, np.zeros(3, np.float32)))
    assert acc.frames == 2
    acc.update(FramePacket(8000, 2, np.ones(2, np.float32)))
    assert acc.finalize().leading_quiet_frames == 2
    with pytest.raises(AudioContractError):
        acc.finalize()
    with pytest.raises(AudioContractError):
        acc.update(FramePacket(8000, 4, np.ones(1, np.float32)))


def make_wave(tmp_path):
    path = tmp_path / "quiet.wav"
    data = np.array([0, 1000, 0, 0, -1000, 0], dtype="<i2")
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(data.tobytes())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("chunk", [1, 2, 3, 8192])
def test_verified_quality_and_policy(tmp_path, chunk):
    path, digest = make_wave(tmp_path)
    plain = inspect_wav_quality(path).as_dict()
    assert "quiet_intervals" not in plain
    assert not plain["inspection_controls"]["quiet_intervals_enabled"]
    result = inspect_wav_quality(
        path, packet_frames=chunk, expected_sha256=digest,
        max_internal_quiet_frames=1, include_continuity=True,
        include_timeline=True,
    )
    assert result.signal_health.passed
    assert not result.inspection_policy_passed
    report = result.as_dict()
    assert report["source_sha256"] == digest
    assert report["source_verified"]
    assert report["inspection_controls"]["max_internal_quiet_frames"] == 1
    assert report["quiet_intervals"]["longest_internal_start"] == 2
    assert report["quiet_intervals"]["longest_internal_stop"] == 4
    assert report["quiet_intervals"]["quiet_frames"] == 4
    assert report["quiet_intervals"]["quiet_fraction"] == pytest.approx(
        1 - result.signal_health.active_fraction
    )
    assert inspect_wav_quality(path, max_internal_quiet_frames=2).inspection_policy_passed
    # A higher floor makes every frame quiet; observation has no extra policy.
    all_quiet = inspect_wav_quality(
        path, limits=QualityLimits(activity_floor_dbfs=-20),
        include_quiet_intervals=True,
    )
    assert all_quiet.quiet_intervals.quiet_frames == 6
    assert all_quiet.quiet_intervals.passed


def test_cli_verified_output_exit_codes_and_opt_in(tmp_path, capsys):
    path, digest = make_wave(tmp_path)
    args = ["wav-inspect", str(path), "--expected-sha256", digest]
    assert main(args + ["--quiet-intervals"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["quiet_intervals"]["passed"]
    assert main(args + ["--max-internal-quiet-frames", "1"]) == 0
    assert not json.loads(capsys.readouterr().out)["inspection_policy_passed"]
    assert main(args + ["--max-internal-quiet-frames", "1", "--fail-on-warnings"]) == 1
    assert not json.loads(capsys.readouterr().out)["quiet_intervals"]["passed"]
    assert main(args + ["--max-internal-quiet-frames", "2", "--fail-on-warnings"]) == 0
    assert json.loads(capsys.readouterr().out)["quiet_intervals"]["passed"]
    assert main(args + ["--max-internal-quiet-frames", "-1"]) == 2
    out = capsys.readouterr()
    assert not out.out
    assert "max_internal_quiet_frames" in out.err
    assert main(["wav-inspect", str(path), "--quiet-intervals",
                 "--expected-sha256", "0" * 64]) == 2
    out = capsys.readouterr()
    assert not out.out
    assert "SHA256" in out.err
