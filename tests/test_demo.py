"""Reproducibility, provenance and atomic publication of the public demo."""
from __future__ import annotations

import hashlib
import json
import wave

import pytest

from sonumio.demo import ENGINE, PROMPT, SEED, SECONDS, create_demo, main
from sonumio.service import generate_audio


def test_demo_is_identical_to_existing_procedural_engine(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    report = create_demo(first)
    repeat = create_demo(second, expected_sha256=report["wav_sha256"])
    expected = generate_audio(PROMPT, SECONDS, model_id=ENGINE, seed=SEED)
    assert first.read_bytes() == second.read_bytes() == expected.wav_bytes
    assert report == repeat
    assert report["wav_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert report["original_model_inference"] is False
    assert report["third_party_recordings_used"] is False
    assert report["bytes_written"] == 576044
    verified = report["inspection"]
    assert verified["source_sha256"] == report["wav_sha256"]
    assert verified["source_verified"] is True
    assert verified["pcm_rails"]["positive_samples"] == 0
    assert verified["pcm_rails"]["negative_samples"] == 0
    assert verified["signal_health"]["clipping_fraction"] == 0
    assert "quiet_intervals" in verified
    assert "frame_continuity" in verified
    with wave.open(str(first), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 24000
        assert reader.getnframes() == 288000


def test_demo_digest_failure_never_creates_destination(tmp_path, capsys):
    destination = tmp_path / "new-dir" / "demo.wav"
    assert main(["--output", str(destination), "--expected-sha256", "0" * 64]) == 2
    result = capsys.readouterr()
    assert not result.out
    assert "SHA256" in result.err
    assert not destination.parent.exists()


def test_demo_cli_does_not_overwrite_without_explicit_opt_in(tmp_path, capsys):
    destination = tmp_path / "demo.wav"
    destination.write_bytes(b"original content")
    assert main(["--output", str(destination)]) == 2
    assert destination.read_bytes() == b"original content"
    result = capsys.readouterr()
    assert not result.out
    assert "--overwrite" in result.err
    assert main(["--output", str(destination), "--overwrite"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["wav_sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.partial"))
