"""Verified WAV analysis CLI: JSON receipts, immutable input and errors."""
import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.waveio import encode_wav_bytes
from sonumio.cli import main


def save_wav(tmp_path, data, rate=8000):
    path = tmp_path / "input.wav"
    path.write_bytes(encode_wav_bytes(make_buffer(data.astype(np.float32), rate)))
    return path


def sine(rate=8000, frequency=440., seconds=.7):
    frames = round(rate * seconds)
    return .35 * np.sin(2 * np.pi * frequency * np.arange(frames) / rate)


def clicks(rate=8000, seconds=8, bpm=120):
    data = np.zeros(rate * seconds)
    for t in np.arange(.1, seconds - .1, 60 / bpm):
        offset = round(t * rate)
        data[offset:offset + 32] = .6 * np.hanning(32)
    return data


@pytest.mark.parametrize("summary_only", [False, True])
def test_verified_pitch_cli_receipt(tmp_path, capsys, summary_only):
    path = save_wav(tmp_path, sine())
    args = ["wav-pitch", str(path)]
    if summary_only:
        args.append("--summary-only")
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["protocol"] == "sonumio-local-audio-analysis-v1"
    assert result["command"] == "wav-pitch"
    assert result["source"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["source"]["source_verified"] is True
    assert result["source_modified"] is False
    assert result["model_inference"] is False
    assert result["analysis"]["summary"]["voiced_frames"] > 0
    assert result["analysis"]["summary"]["median_pitch_hz"] == pytest.approx(440, abs=10)
    assert ("frames" not in result["analysis"]) is summary_only


@pytest.mark.parametrize("summary_only", [False, True])
def test_verified_rhythm_cli_receipt(tmp_path, capsys, summary_only):
    path = save_wav(tmp_path, clicks())
    args = ["wav-rhythm", str(path), "--min-bpm", "60",
            "--max-bpm", "200"]
    if summary_only:
        args.append("--summary-only")
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["analysis"]["tempo_is_hypothesis"] is True
    assert result["analysis"]["onset_count"] > 4
    assert result["analysis"]["tempo_candidate_count"] > 0
    assert ("onsets" not in result["analysis"]) is summary_only
    assert ("beat_frames" not in result["analysis"]) is summary_only


@pytest.mark.parametrize("command", ["wav-pitch", "wav-rhythm"])
def test_pinned_input_digest_replay_is_stable(tmp_path, capsys, command):
    path = save_wav(tmp_path, sine())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    args = [command, str(path), "--expected-sha256", digest, "--summary-only"]
    assert main(args) == 0
    result = capsys.readouterr().out
    assert main(args) == 0
    assert capsys.readouterr().out == result
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("command", ["wav-pitch", "wav-rhythm"])
def test_mismatched_sha_rejects_without_json(tmp_path, capsys, command):
    path = save_wav(tmp_path, sine())
    assert main([command, str(path), "--expected-sha256", "0" * 64]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
    assert path.read_bytes()[:4] == b"RIFF"


@pytest.mark.parametrize("command", ["wav-pitch", "wav-rhythm"])
def test_missing_source_fails_without_output(tmp_path, capsys, command):
    path = tmp_path / "missing.wav"
    assert main([command, str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Sonumio local analysis:" in captured.err
    assert not path.exists()


def test_invalid_pitch_option_fails_closed(tmp_path, capsys):
    path = save_wav(tmp_path, sine())
    assert main(["wav-pitch", str(path), "--min-hz", "1500",
                 "--max-hz", "200"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""


def test_invalid_rhythm_bpm_rejected(tmp_path, capsys):
    path = save_wav(tmp_path, clicks())
    assert main(["wav-rhythm", str(path), "--min-bpm", "180",
                 "--max-bpm", "60"]) == 2
    assert capsys.readouterr().out == ""


def test_source_unchanged_after_analysis(tmp_path, capsys):
    path = save_wav(tmp_path, sine())
    before = path.read_bytes()
    assert main(["wav-pitch", str(path), "--summary-only"]) == 0
    capsys.readouterr()
    assert main(["wav-rhythm", str(path), "--summary-only"]) == 0
    assert path.read_bytes() == before
