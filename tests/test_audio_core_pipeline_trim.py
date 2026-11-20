"""Frame-accurate verified streaming WAV quiet-edge trimming and provenance."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import sonumio.audio_core.wave_pipeline as wave_pipeline
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import (
    AudioContractError, AudioIntegrityError, AudioLimitError,
)
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_pipeline import WavProcessingConfig, process_wav_file
from sonumio.audio_core.wave_reader import WavPacketReader
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.cli import main


def _source(tmp_path, samples, bits=16):
    data = np.asarray(samples, dtype=np.float32)
    assert data.ndim == 2 and data.shape[1] > 0
    path = tmp_path / "source.wav"
    with WavPacketWriter(path, WavPacketSpec(
        24000, data.shape[0], data.shape[1],
        QuantizationSettings(bits=bits),
    )) as writer:
        for start in range(0, data.shape[1], 37):
            writer.push(FramePacket(24000, start, data[:, start:start + 37]))
        record = writer.finish()
    return path, record.wav_sha256


def _read(path, packet_frames=19):
    with WavPacketReader(path, packet_frames=packet_frames) as reader:
        parts = [packet.samples for packet in reader]
        assert reader.verified_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    return np.concatenate(parts, axis=1)


@pytest.mark.parametrize("packet_frames", [1, 3, 17, 257])
@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_trim_shared_stereo_frame_clock_is_exact(tmp_path, packet_frames, bits):
    data = np.zeros((2, 67), dtype=np.float32)
    data[:, :9] = .004  # below a -40 dBFS threshold
    data[:, 59:] = -.004
    data[0, 9] = .5
    data[1, 58] = -.25
    data[0, 21] = -.125  # retain the internal quiet gap
    source, source_sha = _source(tmp_path, data, bits=bits)
    original = _read(source)
    target = tmp_path / f"trim-{bits}-{packet_frames}.wav"
    result = process_wav_file(
        source, target, expected_source_sha256=source_sha,
        config=WavProcessingConfig(
            trim_silence_dbfs=-40, packet_frames=packet_frames,
        ),
    )
    assert result.source_sha256 == source_sha
    assert result.source_frames == 67
    assert result.retained_start_frame == 9
    assert result.retained_end_frame == 59
    assert result.frames == 50
    assert result.output_bytes == len(target.read_bytes())
    assert result.trim_silence_dbfs == -40
    np.testing.assert_array_equal(_read(target), original[:, 9:59])
    assert result.output_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_default_off_preserves_byte_identical_output_and_full_coordinates(tmp_path):
    data = np.zeros((1, 40), dtype=np.float32)
    data[0, 12:24] = np.linspace(-.5, .5, 12, dtype=np.float32)
    source, sha = _source(tmp_path, data)
    result = process_wav_file(source, tmp_path / "copy.wav")
    assert result.source_frames == result.frames == 40
    assert result.retained_start_frame == 0
    assert result.retained_end_frame == 40
    assert result.trim_silence_dbfs is None
    assert result.output_sha256 == sha


@pytest.mark.parametrize("packet_frames", [1, 11, 67, 128])
def test_trim_dc_dither_and_output_digest_are_partition_invariant(tmp_path, packet_frames):
    t = np.arange(128, dtype=np.float32)
    data = np.zeros((2, 128), dtype=np.float32)
    data[0, 7:121] = .15 + .23 * np.sin(t[7:121] / 7)
    data[1, 9:119] = -.17 + .21 * np.cos(t[9:119] / 11)
    source, sha = _source(tmp_path, data)
    settings = WavProcessingConfig(
        packet_frames=packet_frames, trim_silence_dbfs=-40,
        remove_dc=True, dither_seed=105, output_bits=24, gain_db=-3,
    )
    reference = process_wav_file(
        source, tmp_path / f"reference-{packet_frames}.wav",
        config=settings, expected_source_sha256=sha,
    )
    verified = process_wav_file(
        source, tmp_path / f"verified-{packet_frames}.wav",
        config=settings, expected_source_sha256=sha,
        expected_output_sha256=reference.output_sha256,
    )
    assert verified.output_sha256 == reference.output_sha256
    assert verified.dc_removed and verified.dither_applied
    assert verified.retained_start_frame == 7
    assert verified.retained_end_frame == 121
    assert verified.frames == 114
    assert verified.output_rms_before_quantization > 0
    assert (tmp_path / f"verified-{packet_frames}.wav").read_bytes() == (
        tmp_path / f"reference-{packet_frames}.wav"
    ).read_bytes()


def test_all_quiet_input_rejected_without_publishing_or_creating_parent(tmp_path):
    source, sha = _source(tmp_path, np.full((2, 73), .004, dtype=np.float32))
    target = tmp_path / "not-created" / "blocked.wav"
    with pytest.raises(AudioLimitError, match="No WAV frames"):
        process_wav_file(
            source, target, expected_source_sha256=sha,
            config=WavProcessingConfig(trim_silence_dbfs=-40),
            create_parent=True,
        )
    assert not target.parent.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_all_quiet_input_preserves_previous_overwrite_destination(tmp_path):
    source, _ = _source(tmp_path, np.zeros((1, 41), dtype=np.float32))
    target = tmp_path / "preserve.wav"
    target.write_bytes(b"valuable pre-existing output")
    with pytest.raises(AudioLimitError, match="No WAV frames"):
        process_wav_file(
            source, target, overwrite=True,
            config=WavProcessingConfig(trim_silence_dbfs=-60),
        )
    assert target.read_bytes() == b"valuable pre-existing output"


def test_single_retained_24_bit_frame_produces_valid_odd_riff_padding(tmp_path):
    data = np.zeros((1, 7), dtype=np.float32)
    data[0, 3] = .5
    source, sha = _source(tmp_path, data, bits=24)
    result = process_wav_file(
        source, tmp_path / "one-frame.wav",
        config=WavProcessingConfig(trim_silence_dbfs=-60),
        expected_source_sha256=sha,
    )
    assert (result.retained_start_frame, result.retained_end_frame) == (3, 4)
    assert result.frames == 1
    assert result.output_bytes == 48  # 44-byte header, 3-byte PCM, 1-byte pad
    np.testing.assert_array_equal(_read(result.destination), _read(source)[:, 3:4])


@pytest.mark.parametrize("invalid", [-121, 0, 1, True, -40.5j, "silent", float("nan"), float("inf")])
def test_bad_silence_threshold_rejected(invalid):
    with pytest.raises(AudioContractError, match="WAV trim threshold"):
        WavProcessingConfig(trim_silence_dbfs=invalid)


def test_trim_is_computed_on_raw_multichannel_signal_not_per_channel(tmp_path):
    data = np.zeros((2, 20), dtype=np.float32)
    data[0, 3] = 0.5
    data[1, 16] = -0.25
    source, _ = _source(tmp_path, data)
    result = process_wav_file(
        source, tmp_path / "channels.wav",
        config=WavProcessingConfig(trim_silence_dbfs=-20),
    )
    assert result.retained_start_frame == 3
    assert result.retained_end_frame == 17
    assert result.frames == 14
    np.testing.assert_array_equal(_read(result.destination), _read(source)[:, 3:17])


def test_all_input_still_verified_after_trim_second_pass(tmp_path, monkeypatch):
    data = np.zeros((1, 119), dtype=np.float32)
    data[0, 30:45] = .6
    source, sha = _source(tmp_path, data)
    original_factory = wave_pipeline.WavPacketReader
    count = 0

    def mutate_on_second_open(path, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            with open(path, "r+b") as fp:
                fp.seek(44 + (100 * 2))  # trimmed tail, not exported
                old = fp.read(1)
                fp.seek(44 + (100 * 2))
                fp.write(bytes([old[0] ^ 1]))
        return original_factory(path, *args, **kwargs)

    monkeypatch.setattr(wave_pipeline, "WavPacketReader", mutate_on_second_open)
    target = tmp_path / "must-abort.wav"
    with pytest.raises(AudioIntegrityError, match="SHA256"):
        process_wav_file(
            source, target, expected_source_sha256=sha,
            config=WavProcessingConfig(trim_silence_dbfs=-60, packet_frames=13),
        )
    assert count == 2
    assert not target.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_cli_trim_reports_source_bounds_and_sha(tmp_path, capsys):
    data = np.zeros((1, 53), dtype=np.float32)
    data[0, 13:39] = .5
    source, sha = _source(tmp_path, data)
    target = tmp_path / "cli.wav"
    assert main([
        "wav-process", str(source), "--output", str(target),
        "--trim-silence-dbfs", "-40", "--expected-sha256", sha,
        "--packet-frames", "8",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source_frames"] == 53
    assert report["frames"] == 26
    assert report["retained_start_frame"] == 13
    assert report["retained_end_frame"] == 39
    assert report["trim_silence_dbfs"] == -40
    assert report["source_sha256"] == sha
    assert report["output_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
