"""End-to-end local Sonumio wav-process CLI checks."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sonumio.cli import main
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter


def _source(tmp_path):
    path = tmp_path / "approved.wav"
    values = np.sin(np.arange(700, dtype=np.float32) / 19) * .4
    with WavPacketWriter(
        path, WavPacketSpec(24000, 1, 700, QuantizationSettings(bits=16))
    ) as writer:
        writer.push(FramePacket(24000, 0, values[:200]))
        writer.push(FramePacket(24000, 200, values[200:]))
        result = writer.finish()
    return path, result.wav_sha256


def test_cli_wav_process_with_checked_digest_and_metrics(tmp_path, capsys):
    source, sha = _source(tmp_path)
    destination = tmp_path / "processed.wav"
    exit_code = main([
        "wav-process", str(source), "--output", str(destination),
        "--expected-sha256", sha, "--gain-db", "-3",
        "--peak-ceiling-dbfs", "-1", "--packet-frames", "53",
    ])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert not captured.err
    result = json.loads(captured.out)
    assert result["destination"] == str(destination)
    assert result["source_sha256"] == sha
    assert result["output_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert result["frames"] == 700
    assert result["original_model_inference"] is False
    assert result["effective_gain_db"] == pytest.approx(-3)
    assert result["clipped_samples"] == 0


def test_cli_mismatched_source_checksum_does_not_write(tmp_path, capsys):
    source, _ = _source(tmp_path)
    target = tmp_path / "new" / "blocked.wav"
    code = main([
        "wav-process", str(source), "--output", str(target),
        "--expected-sha256", "0" * 64, "--create-parent",
    ])
    assert code == 2
    assert "SHA256" in capsys.readouterr().err
    assert not target.parent.exists()


def test_cli_existing_destination_requires_explicit_overwrite(tmp_path, capsys):
    source, sha = _source(tmp_path)
    target = tmp_path / "existing.wav"
    target.write_bytes(b"do-not-overwrite")
    common = [
        "wav-process", str(source), "--output", str(target),
        "--expected-sha256", sha,
    ]
    assert main(common) == 2
    assert target.read_bytes() == b"do-not-overwrite"
    assert main(common + ["--overwrite"]) == 0
    assert json.loads(capsys.readouterr().out)["frames"] == 700
    assert target.read_bytes() == source.read_bytes()


def test_cli_rejects_out_of_range_gain_without_output(tmp_path, capsys):
    source, _ = _source(tmp_path)
    target = tmp_path / "invalid.wav"
    code = main([
        "wav-process", str(source), "--output", str(target),
        "--gain-db", "300",
    ])
    assert code == 2
    assert not target.exists()
    assert "WAV gain dB" in capsys.readouterr().err


def test_cli_help_requires_output_and_never_uses_default_overwrite(tmp_path):
    source, _ = _source(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["wav-process", str(source)])
    assert exc.value.code == 2
