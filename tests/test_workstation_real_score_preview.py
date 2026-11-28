"""Actual Sonumio audio_core synthesis / PCM publication integration checks.

Skip only when this optional workstation code is tested in isolation without
the existing Sonumio audio_core. In full-repository CI, these MUST execute.
"""
from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

pytest.importorskip("sonumio.audio_core.wavetable")
from sonumio.audio_core.waveio import read_wav_file
from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.score import Score, ScoreNote
from sonumio.midi_workstation.score_io import encode_score
from sonumio.midi_workstation.score_preview import (
    ScorePreviewOptions, export_score_preview,
)
from sonumio.midi_workstation.wavetable_bridge import WavetableBridgeSettings
from sonumio.midi_workstation.contracts import MidiWorkstationError


def make_score(*, muted=False):
    base = Score.empty(ppqn=480, end=480)
    track = replace(base.tracks[0], muted=muted,
                    notes=(ScoreNote("a440", 0, 240, 69, velocity=120),))
    return replace(base, tracks=(track,))


def opts(**changes):
    bridge = dict(sample_rate=8000, channels=1, waveform="sine",
                  attack_frames=0, decay_frames=0, sustain_level=1.,
                  release_frames=64, max_harmonics=8, master_gain=.12)
    bridge.update(changes)
    return ScorePreviewOptions(bridge=WavetableBridgeSettings(**bridge))


def test_real_score_preview_writes_hash_pinned_pcm_and_preserves_score(tmp_path):
    score = make_score()
    original_digest = score.sha256
    target = tmp_path / "tone.wav"
    report = export_score_preview(score, target, opts())
    content = target.read_bytes()
    assert content.startswith(b"RIFF") and content[8:12] == b"WAVE"
    assert report.wav_sha256 == sha256(content).hexdigest()
    assert report.source_score_sha256 == original_digest
    assert score.sha256 == original_digest
    assert report.output_frames == 4064
    assert report.output_channels == 1
    assert report.output_bits == 16
    assert report.rendered_notes == 1
    assert report.skipped_percussion_notes == 0
    assert report.peak > .02
    assert report.as_dict()["trained_model_inference"] is False
    decoded = read_wav_file(target)
    assert decoded.channels == 1 and decoded.frames == 4064
    assert float(np.max(np.abs(decoded.samples))) > .02


def test_real_preview_is_reproducible_with_explicit_dither(tmp_path):
    score = make_score()
    settings = replace(opts(), dither_seed=2026)
    a = export_score_preview(score, tmp_path / "a.wav", settings)
    b = export_score_preview(score, tmp_path / "b.wav", settings)
    assert a.wav_sha256 == b.wav_sha256
    assert a.pcm_float_sha256 == b.pcm_float_sha256
    assert (tmp_path / "a.wav").read_bytes() == (tmp_path / "b.wav").read_bytes()


def test_real_preview_refuses_unapproved_overwrite(tmp_path):
    target = tmp_path / "existing.wav"
    target.write_bytes(b"user previous output")
    with pytest.raises(MidiWorkstationError):
        export_score_preview(make_score(), target, opts())
    assert target.read_bytes() == b"user previous output"


def test_real_preview_refuses_wrong_output_hash_before_publishing(tmp_path):
    target = tmp_path / "pinned.wav"
    with pytest.raises(MidiWorkstationError):
        export_score_preview(make_score(), target,
            replace(opts(), expected_output_sha256="0" * 64))
    assert not target.exists()


def test_real_preview_render_of_muted_score_is_silent(tmp_path):
    target = tmp_path / "muted.wav"
    result = export_score_preview(make_score(muted=True), target, opts())
    assert result.rendered_notes == 0
    decoded = read_wav_file(target)
    assert float(np.max(np.abs(decoded.samples))) == 0.


def test_real_preview_does_not_silently_render_muted_track(tmp_path):
    result = export_score_preview(make_score(muted=True),
                                  tmp_path / "audible.wav",
                                  replace(opts(), bridge=replace(opts().bridge,
                                                                 include_muted=True)))
    assert result.rendered_notes == 1
    assert result.peak > .02


def test_real_preview_mono_rejects_noncentered_pan(tmp_path):
    score = make_score()
    track = replace(score.tracks[0], notes=(replace(score.notes[0], pan=0),))
    score = replace(score, tracks=(track,))
    with pytest.raises(MidiWorkstationError):
        export_score_preview(score, tmp_path / "bad.wav", opts())
    assert not (tmp_path / "bad.wav").exists()


def test_real_cli_score_preview_wav_full_path(tmp_path, capsys):
    score = make_score()
    src = tmp_path / "score.json"
    src.write_bytes(encode_score(score))
    target = tmp_path / "tone.wav"
    status = main(["score-preview-wav", str(src), "--output", str(target),
                   "--sample-rate", "8000", "--channels", "1",
                   "--waveform", "sine", "--master-gain", ".12",
                   "--attack-frames", "0", "--decay-frames", "0",
                   "--sustain-level", "1", "--release-frames", "64",
                   "--max-harmonics", "8", "--output-bits", "16"])
    captured = capsys.readouterr()
    assert status == 0, captured.err
    receipt = json.loads(captured.out)
    assert receipt["preview"]["wav_sha256"] == sha256(target.read_bytes()).hexdigest()
    assert receipt["source"]["canonical_sha256"] == score.sha256
    assert receipt["preview"]["rendered_notes"] == 1


def test_real_cli_preview_sha_pin_rejects_without_output(tmp_path, capsys):
    src = tmp_path / "score.json"
    src.write_bytes(encode_score(make_score()))
    target = tmp_path / "wrong.wav"
    status = main(["score-preview-wav", str(src), "--output", str(target),
                   "--sample-rate", "8000", "--channels", "1",
                   "--expected-output-sha256", "0" * 64])
    assert status == 2
    assert capsys.readouterr().out == ""
    assert not target.exists()
