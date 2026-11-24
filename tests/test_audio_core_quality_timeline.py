"""Packet-invariant fixed-window signal metrics with explicit bounds."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import (
    AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.quality import measure_quality
from sonumio.audio_core.quality_timeline import (
    MAX_TIMELINE_WINDOWS, StreamingQualityTimeline,
)
from sonumio.audio_core.stream_quality import inspect_wav_quality
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.cli import main


def _packets(source: np.ndarray, *, packet_frames: int, sample_rate: int = 8000):
    return [
        FramePacket(sample_rate, i, source[:, i:i + packet_frames])
        for i in range(0, source.shape[1], packet_frames)
    ]


@pytest.mark.parametrize("packet_frames", [1, 2, 3, 7, 15, 64])
@pytest.mark.parametrize("window_frames", [1, 5, 9, 31, 64])
def test_windows_are_frame_exact_across_packet_partitions(
    packet_frames, window_frames
):
    n = 31
    x = np.arange(n, dtype=np.float32)
    raw = np.stack([
        (np.sin(x * .29) * .75).astype(np.float32),
        (np.cos(x * .37) * .25 + .017).astype(np.float32),
    ])
    raw[:, 3] = 1.0
    raw[:, 17] = 0.0
    accumulator = StreamingQualityTimeline(
        8000, 2, window_frames=window_frames,
        expected_frames=n, activity_floor_dbfs=-70,
    )
    for packet in _packets(raw, packet_frames=packet_frames):
        accumulator.update(packet)
    report = accumulator.finalize()
    assert report.frames == n
    assert report.sample_rate == 8000
    assert len(report.windows) == (n + window_frames - 1) // window_frames
    assert report.as_dict()["text_prompt_adherence_measured"] is False
    assert report.windows[0].start_frame == 0
    assert report.windows[-1].stop_frame == n
    for i, window in enumerate(report.windows):
        start, stop = i * window_frames, min(n, (i + 1) * window_frames)
        baseline = measure_quality(make_buffer(raw[:, start:stop], 8000))
        assert window.start_frame == start
        assert window.stop_frame == stop
        assert window.frames == stop - start
        assert window.peak_amplitude == pytest.approx(
            baseline.peak_amplitude, abs=1e-12
        )
        assert window.rms_amplitude == pytest.approx(
            baseline.rms_amplitude, abs=1e-12
        )
        assert window.dc_offset_absolute == pytest.approx(
            baseline.dc_offset_absolute, abs=1e-12
        )
        assert window.clipping_fraction == baseline.clipping_fraction
        assert window.active_fraction == baseline.active_fraction
        entry = window.as_dict()
        assert entry["start_seconds"] == start / 8000
        assert entry["stop_seconds"] == stop / 8000


def test_window_clipping_and_activity_in_multichannel_audio():
    data = np.zeros((2, 20), dtype=np.float32)
    data[0, 0] = 1.0
    data[1, 2] = -1.3
    data[0, 5:10] = .5
    data[1, 18] = -0.1
    acc = StreamingQualityTimeline(
        8000, 2, window_frames=10, expected_frames=20,
    )
    acc.update(FramePacket(8000, 0, data[:, :11]))
    acc.update(FramePacket(8000, 11, data[:, 11:]))
    report = acc.finalize()
    assert len(report.windows) == 2
    first, second = report.windows
    assert first.clipping_fraction == pytest.approx(2 / 20)
    assert first.active_fraction == pytest.approx(7 / 10)
    assert second.clipping_fraction == 0.0
    assert second.active_fraction == pytest.approx(1 / 10)
    assert report.as_dict()["window_count"] == 2


def test_timeline_rejects_incomplete_gapped_overlength_and_duplicate_inputs():
    acc = StreamingQualityTimeline(8000, 1, window_frames=4, expected_frames=6)
    acc.update(FramePacket(8000, 0, np.zeros(2, dtype=np.float32)))
    for wrong in (
        FramePacket(8000, 0, np.zeros(2, dtype=np.float32)),
        FramePacket(8000, 3, np.zeros(2, dtype=np.float32)),
        FramePacket(16000, 2, np.zeros(2, dtype=np.float32)),
        FramePacket(8000, 2, np.zeros((2, 2), dtype=np.float32)),
    ):
        with pytest.raises(AudioContractError, match="clock-aligned"):
            acc.update(wrong)
        assert acc.frames == 2
    with pytest.raises(AudioIntegrityError, match="Incomplete"):
        acc.finalize()
    with pytest.raises(AudioLimitError, match="length"):
        acc.update(FramePacket(8000, 2, np.zeros(5, dtype=np.float32)))
    assert acc.frames == 2
    acc.update(FramePacket(8000, 2, np.zeros(4, dtype=np.float32)))
    report = acc.finalize()
    assert [(w.start_frame, w.stop_frame) for w in report.windows] == [(0, 4), (4, 6)]
    with pytest.raises(AudioContractError, match="finalized"):
        acc.finalize()
    with pytest.raises(AudioContractError, match="finalized"):
        acc.update(FramePacket(8000, 6, np.zeros(1, dtype=np.float32)))


@pytest.mark.parametrize("channels", [0, True, 9])
def test_invalid_timeline_channels_are_rejected(channels):
    with pytest.raises(AudioContractError):
        StreamingQualityTimeline(8000, channels)


@pytest.mark.parametrize("window", [0, -1, True, 8000 * 60 + 1])
def test_invalid_window_geometry_is_rejected(window):
    with pytest.raises(AudioContractError):
        StreamingQualityTimeline(8000, 1, window_frames=window)


def test_window_allocation_and_full_duration_are_bounded():
    with pytest.raises(AudioLimitError, match="capacity"):
        StreamingQualityTimeline(
            8000, 1, window_frames=1, expected_frames=MAX_TIMELINE_WINDOWS + 1
        )
    for invalid in (0, -1, True, 8000 * 3600 + 1):
        with pytest.raises(AudioLimitError):
            StreamingQualityTimeline(8000, 1, expected_frames=invalid)
    with pytest.raises(AudioContractError):
        StreamingQualityTimeline(8000, 1, activity_floor_dbfs=float("nan"))
    with pytest.raises(AudioContractError):
        StreamingQualityTimeline(8000, 1).update("not a packet")
    with pytest.raises(AudioIntegrityError):
        StreamingQualityTimeline(8000, 1).finalize()


def test_window_limit_without_expected_frames_rejects_packet_without_mutation():
    accumulator = StreamingQualityTimeline(8000, 1, window_frames=1)
    for offset in range(0, MAX_TIMELINE_WINDOWS, 64):
        packet = FramePacket(8000, offset, np.zeros(64, dtype=np.float32))
        accumulator.update(packet)
    assert accumulator.frames == MAX_TIMELINE_WINDOWS
    with pytest.raises(AudioLimitError, match="capacity"):
        accumulator.update(FramePacket(
            8000, MAX_TIMELINE_WINDOWS, np.zeros(1, dtype=np.float32)
        ))
    assert accumulator.frames == MAX_TIMELINE_WINDOWS
    assert len(accumulator.finalize().windows) == MAX_TIMELINE_WINDOWS


def _make_canonical(tmp_path, *, frames=17_003, rate=8000, bits=16):
    path = tmp_path / "timeline.wav"
    x = np.arange(frames, dtype=np.float64) / rate
    y = (.4 * np.sin(2 * np.pi * 330 * x)).astype(np.float32)
    with WavPacketWriter(
        path, WavPacketSpec(rate, 1, frames, QuantizationSettings(bits=bits))
    ) as writer:
        for offset in range(0, frames, 1253):
            writer.push(FramePacket(rate, offset, y[offset:offset + 1253]))
        result = writer.finish()
    return path, result


@pytest.mark.parametrize("packet_frames", [7, 503, 8192])
def test_verified_wav_timeline_covers_full_source(tmp_path, packet_frames):
    path, published = _make_canonical(tmp_path)
    result = inspect_wav_quality(
        path, packet_frames=packet_frames,
        expected_sha256=published.wav_sha256, include_timeline=True,
    )
    assert result.timeline is not None
    assert result.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result.timeline.frames == 17003
    assert [(w.start_frame, w.stop_frame) for w in result.timeline.windows] == [
        (0, 8000), (8000, 16000), (16000, 17003)
    ]
    payload = result.as_dict()
    assert payload["source_verified"] is True
    assert payload["timeline"]["window_count"] == 3
    assert all(w["frames"] > 0 for w in payload["timeline"]["windows"])
    assert payload["timeline"]["perceptual_audio_quality_measured"] is False
    assert payload["signal_health"]["perceptual_audio_quality_measured"] is False
    without = inspect_wav_quality(path)
    assert without.timeline is None
    assert "timeline" not in without.as_dict()


def test_timeline_rejects_bad_source_before_exposing_report(tmp_path, capsys):
    path, _ = _make_canonical(tmp_path, frames=5, bits=24)
    raw = bytearray(path.read_bytes())
    assert len(raw) == 60
    raw[-1] = 99
    path.write_bytes(raw)
    with pytest.raises(AudioFormatError, match="padding"):
        inspect_wav_quality(path, packet_frames=2, include_timeline=True)
    assert main(["wav-inspect", str(path), "--timeline"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "padding" in captured.err


def test_timeline_cli_opt_in_and_default_shape(tmp_path, capsys):
    path, expected = _make_canonical(tmp_path, frames=17003)
    assert main(["wav-inspect", str(path), "--timeline",
                 "--expected-sha256", expected.wav_sha256]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["source_verified"]
    assert info["timeline"]["window_count"] == 3
    assert info["timeline"]["window_frames"] == 8000
    assert info["timeline"]["windows"][-1]["stop_frame"] == 17003

    assert main(["wav-inspect", str(path)]) == 0
    no_timeline = json.loads(capsys.readouterr().out)
    assert no_timeline["source_verified"]
    assert "timeline" not in no_timeline


def test_public_timeline_api_validates_bool_option(tmp_path):
    path, _ = _make_canonical(tmp_path, frames=5)
    for bad in (1, None, "yes"):
        with pytest.raises(AudioContractError, match="boolean"):
            inspect_wav_quality(path, include_timeline=bad)
