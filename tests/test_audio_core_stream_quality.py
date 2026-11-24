"""Verified, bounded-memory WAV signal diagnostics and CLI regressions."""
from __future__ import annotations

import hashlib
import json
import wave

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import (
    AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.quality import QualityLimits, measure_quality
from sonumio.audio_core.stream_quality import (
    StreamingQualityAccumulator, inspect_wav_quality,
)
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_reader import WavPacketReader
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.cli import main


def _source(tmp_path, *, bits=16, channels=2, frames=257, rate=24000):
    path = tmp_path / "candidate.wav"
    positions = np.arange(frames, dtype=np.float64)
    signal = (0.54 * np.sin(2 * np.pi * 311 * positions / rate) *
              np.sin(np.linspace(0, np.pi, frames)) ** 2)
    samples = np.stack([
        (signal * (1.0 - idx * .15) + idx * .006).astype(np.float32)
        for idx in range(channels)
    ])
    with WavPacketWriter(
        path, WavPacketSpec(rate, channels, frames, QuantizationSettings(bits=bits))
    ) as writer:
        for offset in range(0, frames, 37):
            writer.push(FramePacket(rate, offset, samples[:, offset:offset + 37]))
        output = writer.finish()
    return path, output


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("packet_frames", [1, 7, 64, 257])
def test_stream_quality_matches_existing_in_memory_report(
    tmp_path, bits, channels, packet_frames
):
    path, exported = _source(tmp_path, bits=bits, channels=channels)
    checked = inspect_wav_quality(
        path, packet_frames=packet_frames,
        expected_sha256=exported.wav_sha256,
    )
    with WavPacketReader(path, packet_frames=73) as reader:
        reconstructed = np.concatenate([p.samples for p in reader], axis=1)
        assert reader.verified_sha256 == exported.wav_sha256
    baseline = measure_quality(make_buffer(reconstructed, 24000))
    streamed = checked.signal_health
    assert checked.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert checked.info.bits_per_sample == bits
    assert checked.info.channels == channels
    assert checked.info.frames == 257
    assert streamed.sample_rate == baseline.sample_rate
    assert streamed.frames == baseline.frames
    assert streamed.channels == baseline.channels
    for key in (
        "peak_amplitude", "rms_amplitude", "dc_offset_absolute",
        "clipping_fraction", "active_fraction", "boundary_jump",
        "channel_imbalance_db",
    ):
        assert getattr(streamed, key) == pytest.approx(
            getattr(baseline, key), abs=1e-10, rel=1e-8
        ), key
    assert [f.code for f in streamed.findings] == [f.code for f in baseline.findings]
    report = checked.as_dict()
    assert report["source_verified"] is True
    assert report["model_quality_measured"] is False
    assert report["recording_rights_verified"] is False
    assert report["signal_health"]["text_prompt_adherence_measured"] is False


def test_stream_quality_detects_clipping_dc_activity_and_boundary_jumps():
    data = np.zeros((2, 11), dtype=np.float32)
    data[0, 0] = 1.25
    data[1, -1] = -0.7
    data[0, 5] = 0.4
    accumulator = StreamingQualityAccumulator(
        8000, 2, expected_frames=11, limits=QualityLimits()
    )
    for start in (0, 3, 8):
        end = 3 if start == 0 else (8 if start == 3 else 11)
        accumulator.update(FramePacket(8000, start, data[:, start:end]))
    result = accumulator.finalize()
    baseline = measure_quality(make_buffer(data, 8000))
    assert set(f.code for f in result.findings) == set(f.code for f in baseline.findings)
    assert result.peak_amplitude == 1.25
    assert result.clipping_fraction == pytest.approx(1 / 22)
    assert result.boundary_jump == 1.25
    assert result.active_fraction == pytest.approx(3 / 11)


def test_stream_quality_rejects_gaps_duplicates_clock_channel_mismatch():
    accumulator = StreamingQualityAccumulator(24000, 2, expected_frames=9)
    good = FramePacket(24000, 0, np.zeros((2, 3), dtype=np.float32))
    accumulator.update(good)
    for bad in (
        good,
        FramePacket(24000, 4, np.zeros((2, 1), dtype=np.float32)),
        FramePacket(48000, 3, np.zeros((2, 1), dtype=np.float32)),
        FramePacket(24000, 3, np.zeros((1, 1), dtype=np.float32)),
    ):
        with pytest.raises(AudioContractError):
            accumulator.update(bad)
        assert accumulator.frames == 3
    with pytest.raises(AudioIntegrityError, match="Incomplete"):
        accumulator.finalize()
    accumulator.update(FramePacket(24000, 3, np.zeros((2, 6), dtype=np.float32)))
    report = accumulator.finalize()
    assert report.frames == 9
    with pytest.raises(AudioContractError, match="finalized"):
        accumulator.finalize()
    with pytest.raises(AudioContractError, match="finalized"):
        accumulator.update(FramePacket(24000, 9, np.zeros((2, 1), dtype=np.float32)))


def test_stream_quality_prevents_excess_frames_and_invalid_inputs(tmp_path):
    for channels in (0, 9, True):
        with pytest.raises(AudioContractError):
            StreamingQualityAccumulator(24000, channels)
    for frames in (0, -1, True, 24000 * 3600 + 1):
        with pytest.raises(AudioLimitError):
            StreamingQualityAccumulator(24000, 1, expected_frames=frames)
    with pytest.raises(AudioContractError):
        StreamingQualityAccumulator(24000, 2, limits=object())
    with pytest.raises(AudioContractError):
        StreamingQualityAccumulator(24000, 1).update(object())
    with pytest.raises(AudioIntegrityError):
        StreamingQualityAccumulator(24000, 1).finalize()
    accumulator = StreamingQualityAccumulator(24000, 1, expected_frames=2)
    with pytest.raises(AudioLimitError):
        accumulator.update(FramePacket(24000, 0, np.zeros(3, dtype=np.float32)))
    assert accumulator.frames == 0
    path, _ = _source(tmp_path)
    for chunk in (0, True, 65_537):
        with pytest.raises(AudioContractError):
            inspect_wav_quality(path, packet_frames=chunk)


def test_file_sha_mismatch_and_corrupted_padding_fail_closed(tmp_path):
    path, exported = _source(tmp_path, bits=24, channels=1, frames=5)
    with pytest.raises(AudioIntegrityError, match="SHA256"):
        inspect_wav_quality(path, packet_frames=2, expected_sha256="0" * 64)
    with pytest.raises(AudioContractError, match="SHA256"):
        inspect_wav_quality(path, expected_sha256="not-a-hash")
    assert inspect_wav_quality(
        path, expected_sha256=exported.wav_sha256,
    ).source_sha256 == exported.wav_sha256
    raw = bytearray(path.read_bytes())
    assert len(raw) == 60
    raw[-1] = 7
    path.write_bytes(raw)
    with pytest.raises(AudioFormatError, match="padding"):
        inspect_wav_quality(path, packet_frames=1)


def test_cli_wav_inspect_json_and_failure_exit_codes(tmp_path, capsys):
    path, exported = _source(tmp_path)
    assert main([
        "wav-inspect", str(path), "--packet-frames", "7",
        "--expected-sha256", exported.wav_sha256,
    ]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    data = json.loads(captured.out)
    assert data["source_sha256"] == exported.wav_sha256
    assert data["frames"] == 257
    assert data["signal_health"]["protocol"] == "sonumio-basic-signal-health-v1"

    assert main([
        "wav-inspect", str(path), "--expected-sha256", "0" * 64,
    ]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SHA256" in captured.err

    assert main([
        "wav-inspect", str(path), "--max-peak", "0.01",
        "--fail-on-warnings",
    ]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["signal_health"]["passed"] is False
    assert "peak" in {x["code"] for x in report["signal_health"]["findings"]}


def test_cli_inspect_rejects_unsupported_wav_without_partial_json(tmp_path, capsys):
    path = tmp_path / "noncanonical.wav"
    path.write_bytes(b"not wave")
    assert main(["wav-inspect", str(path)]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err
