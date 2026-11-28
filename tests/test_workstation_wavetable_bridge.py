"""Isolated synth-adapter contract tests; full Sonumio integration is separate."""
from dataclasses import dataclass, replace
import json
import sys
import types

import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.score import Score, ScoreNote, ScoreTrack
from sonumio.midi_workstation.wavetable_bridge import (
    WavetableBridgeSettings, build_wavetable_plan, render_score_wavetable,
)


@dataclass(frozen=True)
class FakeEnvelope:
    attack_frames: int
    decay_frames: int
    sustain_level: float
    release_frames: int


@dataclass(frozen=True)
class FakePlan:
    sample_rate: int
    frames: int
    channels: int
    waveform: str
    table_size: int
    max_harmonics: int
    master_gain: float
    envelope: FakeEnvelope


@dataclass(frozen=True)
class FakeNote:
    start_frame: int
    gate_frames: int
    midi_note: int
    velocity: float
    pan: float
    initial_phase: float


@dataclass(frozen=True)
class FakeAudio:
    frames: int
    channels: int
    sample_rate: int


@dataclass(frozen=True)
class FakeRenderStats:
    maximum_polyphony: int
    peak: float
    rms: float


@dataclass(frozen=True)
class FakeRenderResult:
    audio: FakeAudio
    report: FakeRenderStats


def _fake_renderer(plan, notes):
    assert all(note.start_frame + note.gate_frames + plan.envelope.release_frames
               <= plan.frames for note in notes)
    maximum_polyphony = max((sum(1 for n in notes
                                 if n.start_frame <= clock < n.start_frame + n.gate_frames)
                             for clock in {n.start_frame for n in notes}), default=0)
    return FakeRenderResult(
        FakeAudio(plan.frames, plan.channels, plan.sample_rate),
        FakeRenderStats(maximum_polyphony, .3, .08),
    )


@pytest.fixture
def stub_synthesis(monkeypatch):
    """Test that bridge forwards correct API parameters, not actual waveform DSP."""
    package = types.ModuleType("sonumio.audio_core")
    package.__path__ = []
    wavetable = types.ModuleType("sonumio.audio_core.wavetable")
    wavetable.WavetablePlan = FakePlan
    wavetable.WavetableNote = FakeNote
    wavetable.VoiceEnvelope = FakeEnvelope
    wavetable.render_wavetable = _fake_renderer
    monkeypatch.setitem(sys.modules, "sonumio.audio_core", package)
    monkeypatch.setitem(sys.modules, "sonumio.audio_core.wavetable", wavetable)
    return wavetable


def original_score():
    base = Score.empty(end=1920)
    one = ScoreTrack("melody", "Melody", 0, 0, 0, 0, (
        ScoreNote("c", 0, 240, 60, velocity=127, pan=64),
        ScoreNote("e", 240, 240, 64, velocity=64, pan=0),
    ))
    drums = ScoreTrack("drums", "Drums", 9, 0, 0, 0, (
        ScoreNote("kick", 0, 100, 36),
    ))
    return replace(base, tracks=(one, drums))


def test_midi_score_converts_to_expected_wavetable_note_clock(stub_synthesis):
    source = original_score()
    opts = WavetableBridgeSettings(sample_rate=24000, release_frames=512)
    plan, notes, schedule, skipped = build_wavetable_plan(source, opts)
    assert plan.sample_rate == 24000
    assert plan.channels == 2
    assert plan.waveform == "triangle"
    assert plan.envelope.release_frames == 512
    assert len(notes) == 2
    assert skipped == 1
    assert notes[0].start_frame == 0
    assert notes[0].gate_frames == 6000
    assert notes[0].velocity == 1.
    assert notes[0].pan == 0.
    assert notes[1].start_frame == 6000
    assert notes[1].midi_note == 64
    assert notes[1].pan == -1.
    assert plan.frames == schedule.output_frames
    assert plan.frames == 48000 + 512


def test_bridge_follows_exact_score_source_digest(stub_synthesis):
    source = original_score()
    result = render_score_wavetable(source)
    receipt = result.report
    assert receipt.source_score_sha256 == source.sha256
    assert receipt.rendered_notes == 2
    assert receipt.skipped_percussion_notes == 1
    assert receipt.output_frames == result.audio.frames
    assert receipt.output_channels == 2
    assert receipt.peak_polyphony == 1
    assert receipt.as_dict()["trained_model_inference"] is False
    assert receipt.as_dict()["external_samples_used"] is False
    json.dumps(receipt.as_dict(), allow_nan=False)


def test_percussion_is_skipped_without_removing_source(stub_synthesis):
    source = original_score()
    _, notes, _, skipped = build_wavetable_plan(source)
    assert all(note.midi_note > 48 for note in notes)
    assert skipped == 1
    assert source.note_count == 3


def test_muted_instrument_is_excluded_by_default(stub_synthesis):
    source = original_score()
    muted_melody = replace(source.tracks[0], muted=True)
    source = replace(source, tracks=(muted_melody, source.tracks[1]))
    _, notes, _, skipped = build_wavetable_plan(source)
    assert not notes
    assert skipped == 1
    _, notes_all, _, skipped_all = build_wavetable_plan(
        source, WavetableBridgeSettings(include_muted=True),
    )
    assert len(notes_all) == 2
    assert skipped_all == 1


def test_mono_rejects_uncentered_midi_pan(stub_synthesis):
    with pytest.raises(MidiWorkstationError):
        build_wavetable_plan(
            original_score(), WavetableBridgeSettings(channels=1),
        )


def test_limit_rejects_too_many_render_notes(stub_synthesis):
    with pytest.raises(MidiLimitError):
        build_wavetable_plan(
            original_score(), WavetableBridgeSettings(maximum_notes=1),
        )


@pytest.mark.parametrize("invalid", [
    dict(sample_rate=7999), dict(sample_rate=200000),
    dict(channels=3), dict(waveform="noise"),
    dict(table_size=333), dict(max_harmonics=0),
    dict(master_gain=-1), dict(attack_frames=-1),
    dict(decay_frames=-1), dict(sustain_level=2),
    dict(release_frames=-1), dict(maximum_notes=0),
    dict(include_muted=1),
])
def test_invalid_bridge_options_rejected(invalid):
    with pytest.raises(MidiWorkstationError):
        WavetableBridgeSettings(**invalid)


def test_invalid_score_or_options_rejected_before_synthesis(stub_synthesis):
    with pytest.raises(MidiWorkstationError):
        build_wavetable_plan(None)
    with pytest.raises(MidiWorkstationError):
        build_wavetable_plan(original_score(), None)
    with pytest.raises(MidiWorkstationError):
        render_score_wavetable(None)
