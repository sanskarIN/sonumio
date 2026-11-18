"""Two-pass local per-channel DC removal with SHA, packet and peak safety tests."""
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


def _source(tmp_path, values):
    path = tmp_path / "input.wav"
    data = np.asarray(values, dtype=np.float32)
    assert data.ndim == 2
    with WavPacketWriter(
        path, WavPacketSpec(
            24000, data.shape[0], data.shape[1], QuantizationSettings(bits=16),
        ),
    ) as writer:
        for start in range(0, data.shape[1], 37):
            writer.push(FramePacket(24000, start, data[:, start:start + 37]))
        result = writer.finish()
    return path, result.wav_sha256


def _read(path):
    with WavPacketReader(path, packet_frames=11) as reader:
        chunks = [packet.samples for packet in reader]
        assert reader.verified_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    return np.concatenate(chunks, axis=1)


@pytest.mark.parametrize("packet_frames", [1, 7, 53, 257])
def test_constant_stereo_dc_removed_exactly(tmp_path, packet_frames):
    values = np.stack([
        np.full(257, 0.5, dtype=np.float32),
        np.full(257, -0.25, dtype=np.float32),
    ])
    source, sha = _source(tmp_path, values)
    result = process_wav_file(
        source, tmp_path / f"center-{packet_frames}.wav",
        expected_source_sha256=sha,
        config=WavProcessingConfig(remove_dc=True, packet_frames=packet_frames),
    )
    assert result.dc_removed
    assert result.removed_dc_offsets == pytest.approx((0.5, -0.25))
    assert result.source_peak == pytest.approx(0.5)
    assert result.source_rms > 0
    assert result.output_peak_before_quantization == 0
    assert result.output_rms_before_quantization == 0
    assert not np.any(_read(result.destination))
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_default_identity_still_bit_exact(tmp_path):
    t = np.arange(281, dtype=np.float32)
    source, sha = _source(tmp_path, np.stack([
        0.23 + 0.17 * np.sin(t / 6),
        -0.16 + 0.22 * np.cos(t / 11),
    ]))
    result = process_wav_file(source, tmp_path / "identity.wav")
    assert result.output_sha256 == sha
    assert not result.dc_removed
    assert result.removed_dc_offsets == ()


def test_dc_with_tpdf_is_packet_partition_invariant(tmp_path):
    t = np.arange(293, dtype=np.float32)
    source, sha = _source(tmp_path, np.stack([
        0.27 + 0.21 * np.sin(t / 9),
        -0.18 + 0.23 * np.cos(t / 13),
    ]))
    payloads = []
    for size in (1, 5, 29, 293):
        path = tmp_path / f"partition-{size}.wav"
        result = process_wav_file(
            source, path, expected_source_sha256=sha,
            config=WavProcessingConfig(
                remove_dc=True, gain_db=-2.0,
                dither_seed=2026, output_bits=24, packet_frames=size,
            ),
        )
        assert result.dc_removed and result.dither_applied
        assert len(result.removed_dc_offsets) == 2
        assert result.output_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        payloads.append(path.read_bytes())
    assert all(payload == payloads[0] for payload in payloads[1:])


def test_centering_can_raise_peak_above_one_and_abort(tmp_path):
    values = np.full((1, 101), 0.75, dtype=np.float32)
    values[0, 0] = -1
    source, sha = _source(tmp_path, values)
    target = tmp_path / "unsafe.wav"
    with pytest.raises(AudioLimitError, match="would clip"):
        process_wav_file(
            source, target, expected_source_sha256=sha,
            config=WavProcessingConfig(remove_dc=True, packet_frames=3),
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_ceiling_reduces_gain_based_on_centered_peak(tmp_path):
    values = np.full((1, 101), 0.75, dtype=np.float32)
    values[0, 0] = -1
    source, sha = _source(tmp_path, values)
    result = process_wav_file(
        source, tmp_path / "ceiling.wav", expected_source_sha256=sha,
        config=WavProcessingConfig(
            remove_dc=True, peak_ceiling_dbfs=-3, packet_frames=7,
        ),
    )
    assert result.clipped_samples == 0
    assert result.source_peak == pytest.approx(1)
    assert result.effective_gain_db < 0
    assert result.output_peak_before_quantization <= 10 ** (-3 / 20) + 1e-6
    assert np.max(np.abs(_read(result.destination))) < 1


def test_centered_extrema_control_explicit_boost(tmp_path):
    values = np.stack([
        np.tile([0.7, 0.9], 100),
        np.tile([-0.7, -0.9], 100),
    ]).astype(np.float32)
    source, sha = _source(tmp_path, values)
    result = process_wav_file(
        source, tmp_path / "boost.wav", expected_source_sha256=sha,
        config=WavProcessingConfig(
            remove_dc=True, peak_ceiling_dbfs=-6, allow_peak_boost=True,
        ),
    )
    assert result.effective_gain_db == pytest.approx(12, abs=1e-9)
    assert result.clipped_samples == 0
    assert result.output_peak_before_quantization < 0.5


@pytest.mark.parametrize("invalid", [1, 0, "true", 3.0, None])
def test_remove_dc_must_be_real_boolean(invalid):
    with pytest.raises(AudioContractError, match="remove_dc"):
        WavProcessingConfig(remove_dc=invalid)


def test_cli_dc_opt_in_reports_per_channel_offsets(tmp_path, capsys):
    source, sha = _source(tmp_path, np.stack([
        np.full(53, 0.25, dtype=np.float32),
        np.full(53, -0.5, dtype=np.float32),
    ]))
    path = tmp_path / "cli.wav"
    assert main([
        "wav-process", str(source), "--output", str(path),
        "--expected-sha256", sha, "--remove-dc", "--packet-frames", "9",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dc_removed"] is True
    assert report["removed_dc_offsets"] == pytest.approx([0.25, -0.5])
    assert report["output_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not np.any(_read(path))
