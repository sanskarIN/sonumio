"""Verified WAV chroma CLI: no source mutation and deterministic receipts."""
import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.waveio import encode_wav_bytes
from sonumio.cli import main


def chord_wav(tmp_path, *, rate=16000, frames=16000):
    n = np.arange(frames) / rate
    data = sum(.15 * np.sin(2 * np.pi * hz * n)
               for hz in (261.625565, 329.627557, 391.995436))
    path = tmp_path / "triad.wav"
    path.write_bytes(encode_wav_bytes(make_buffer(
        data.astype(np.float32), rate,
    )))
    return path


@pytest.mark.parametrize("summary_only", [False, True])
def test_verified_chroma_cli_and_known_triad(tmp_path, capsys, summary_only):
    path = chord_wav(tmp_path)
    command = [
        "wav-chroma", str(path), "--fft-frames", "4096",
        "--hop-frames", "512", "--minimum-score-gap", ".005",
    ]
    if summary_only:
        command.append("--summary-only")
    assert main(command) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["protocol"] == "sonumio-local-audio-analysis-v1"
    assert report["command"] == "wav-chroma"
    assert report["source"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["source_modified"] is False
    assert report["model_inference"] is False
    analysis = report["analysis"]
    assert analysis["protocol"] == "sonumio-chroma-triad-hypotheses-v1"
    assert analysis["chord_is_hypothesis"] is True
    assert "C" in [segment["symbol"] for segment in analysis["segments"]]
    assert len(analysis["overall_chroma"]) == 12
    assert ("chord_hypotheses" not in analysis) == summary_only


def test_pinned_source_sha_is_repeatable(tmp_path, capsys):
    path = chord_wav(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cmd = ["wav-chroma", str(path), "--expected-sha256", digest,
           "--summary-only"]
    assert main(cmd) == 0
    first = capsys.readouterr().out
    assert main(cmd) == 0
    assert capsys.readouterr().out == first


def test_invalid_source_hash_does_not_emit_false_receipt(tmp_path, capsys):
    path = chord_wav(tmp_path)
    assert main(["wav-chroma", str(path), "--expected-sha256", "0" * 64]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Sonumio local analysis:" in captured.err


def test_missing_source_does_not_create_files(tmp_path, capsys):
    path = tmp_path / "missing.wav"
    assert main(["wav-chroma", str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not path.exists()


@pytest.mark.parametrize("argument,value", [
    ("--fft-frames", "63"), ("--hop-frames", "0"),
    ("--lowest-hz", "8000"), ("--highest-hz", "10000"),
    ("--reference-a-hz", "0"), ("--minimum-chord-score", "2"),
    ("--minimum-triad-energy", "-1"), ("--minimum-score-gap", "3"),
])
def test_invalid_chroma_options_are_fail_closed(tmp_path, capsys, argument, value):
    path = chord_wav(tmp_path)
    assert main(["wav-chroma", str(path), argument, value]) == 2
    assert capsys.readouterr().out == ""


def test_unchanged_file_bytes_after_analysis(tmp_path, capsys):
    path = chord_wav(tmp_path)
    before = path.read_bytes()
    assert main(["wav-chroma", str(path), "--summary-only"]) == 0
    capsys.readouterr()
    assert path.read_bytes() == before
