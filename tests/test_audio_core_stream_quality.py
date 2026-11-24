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


def test_single_frame_boundary_definition_matches_existing_quality():
    one = FramePacket(8000, 0, np.array([0.8], dtype=np.float32))
    accumulator = StreamingQualityAccumulator(8000, 1, expected_frames=1)
    accumulator.update(one)
    report = accumulator.finalize()
    baseline = measure_quality(make_buffer(one.samples, 8000))
    assert report.boundary_jump == baseline.boundary_jump == 0.0


def test_inspection_handles_longer_than_in_memory_audio_limit(tmp_path):
    rate = 8000
    count = 121 * rate
    path = tmp_path / "long-quality.wav"
    with WavPacketWriter(path, WavPacketSpec(rate, 1, count)) as writer:
        for start in range(0, count, 8000):
            writer.push(FramePacket(rate, start, np.full(8000, .125, dtype=np.float32)))
        written = writer.finish()
    report = inspect_wav_quality(
        path, packet_frames=17_777, expected_sha256=written.wav_sha256
    )
    assert report.info.frames == count
    assert report.signal_health.rms_amplitude == pytest.approx(.125, abs=4e-5)
    assert report.signal_health.active_fraction == 1.0
    assert report.signal_health.clipping_fraction == 0.0
    assert report.source_sha256 == written.wav_sha256



@pytest.mark.parametrize("bits", [8, 16, 24, 32])
@pytest.mark.parametrize("packet_frames", [1, 3, 8])
def test_exact_integer_pcm_rails_survive_float_rounding(
    tmp_path, bits, packet_frames
):
    """32-bit float32 may coalesce near-rail values; integers do not."""
    from sonumio.audio_core.dither import pack_pcm
    from sonumio.audio_core.stream_quality import PcmRailReport

    full = 1 << (bits - 1)
    raw = np.array([[
        -full, full - 1, -full + 1, full - 2,
        0, 1, -1, 0,
    ]], dtype=np.int64)
    path = tmp_path / f"exact-{bits}.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(bits // 8)
        out.setframerate(24000)
        out.writeframes(pack_pcm(raw, bits=bits))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    with WavPacketReader(path, packet_frames=packet_frames,
                         expected_sha256=expected) as reader:
        assert reader.verified_pcm_rail_counts is None
        assert len(list(reader)) == (8 + packet_frames - 1) // packet_frames
        assert reader.verified_pcm_rail_counts == (1, 1)
    report = inspect_wav_quality(
        path, packet_frames=packet_frames, expected_sha256=expected
    )
    rails = report.pcm_rails
    assert isinstance(rails, PcmRailReport)
    assert rails.positive_samples == 1
    assert rails.negative_samples == 1
    assert rails.total_samples == 8
    assert rails.rail_fraction == pytest.approx(.25)
    document = report.as_dict()["pcm_rails"]
    assert document["possible_full_scale_saturation"]
    assert document["audible_clipping_proven"] is False
    if bits <= 24:
        # Positive integer maximum decodes *below* +1.0.
        assert report.signal_health.clipping_fraction == pytest.approx(1 / 8)


def test_pcm_rail_counts_are_never_verified_on_failure(tmp_path):
    path, _ = _source(tmp_path, bits=16, channels=1)
    with WavPacketReader(path, expected_sha256="0" * 64) as reader:
        assert reader.verified_pcm_rail_counts is None
        with pytest.raises(AudioIntegrityError, match="SHA256"):
            list(reader)
        assert reader.verified_pcm_rail_counts is None
        assert reader.state == "closed"


@pytest.mark.parametrize("counts", [
    (-1, 0, 8), (0, -1, 8), (0, 0, -1), (2, 7, 8),
    (0, 0, 0), (True, 0, 8),
])
def test_pcm_rail_metadata_rejects_impossible_counts(counts):
    from sonumio.audio_core.stream_quality import PcmRailReport
    with pytest.raises(AudioContractError, match="PCM rail"):
        PcmRailReport(*counts)


def test_cli_includes_verified_pcm_rail_metadata(tmp_path, capsys):
    path, result = _source(tmp_path, bits=8, channels=1)
    assert main([
        "wav-inspect", str(path), "--expected-sha256", result.wav_sha256,
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["source_verified"] is True
    assert "pcm_rails" in output
    assert output["pcm_rails"]["total_samples"] == output["frames"] * output["channels"]
    assert 0 <= output["pcm_rails"]["rail_fraction"] <= 1
