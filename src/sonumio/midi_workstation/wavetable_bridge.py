"""Opt-in adapter from Sonumio symbolic scores to existing audio-core synthesis.

This is a compatibility boundary, not a second synthesizer. It constructs
bounded WavetablePlan and WavetableNote values from the validated, exact
sample-clock schedule and delegates waveform generation to audio_core.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from .contracts import MidiLimitError, MidiWorkstationError, finite, integer
from .score import Score
from .transport import TransportSettings, build_schedule

if TYPE_CHECKING:
    from sonumio.audio_core.contracts import AudioBuffer

MAX_WAVETABLE_BRIDGE_NOTES = 2048


@dataclass(frozen=True, slots=True)
class WavetableBridgeSettings:
    sample_rate: int = 24000
    channels: int = 2
    waveform: str = "triangle"
    table_size: int = 2048
    max_harmonics: int = 64
    master_gain: float = .15
    attack_frames: int = 64
    decay_frames: int = 256
    sustain_level: float = .7
    release_frames: int = 512
    include_muted: bool = False
    maximum_notes: int = MAX_WAVETABLE_BRIDGE_NOTES

    def __post_init__(self):
        integer(self.sample_rate, "wavetable sample rate", 8000, 192000)
        integer(self.channels, "wavetable channels", 1, 2)
        if self.waveform not in ("sine", "saw", "square", "triangle"):
            raise MidiWorkstationError("Unsupported original Fourier waveform")
        integer(self.table_size, "wavetable table size", 256, 8192)
        if self.table_size & (self.table_size - 1):
            raise MidiWorkstationError("Wavetable size must be power-of-two")
        integer(self.max_harmonics, "maximum harmonics", 1, 256)
        finite(self.master_gain, "master gain", 0., 1.)
        integer(self.attack_frames, "attack frames", 0, 1_000_000)
        integer(self.decay_frames, "decay frames", 0, 1_000_000)
        finite(self.sustain_level, "sustain level", 0., 1.)
        integer(self.release_frames, "release frames", 0, 1_000_000)
        integer(self.maximum_notes, "bridge note limit", 1, MAX_WAVETABLE_BRIDGE_NOTES)
        if type(self.include_muted) is not bool:
            raise MidiWorkstationError("include_muted must be boolean")


@dataclass(frozen=True, slots=True)
class BridgeReceipt:
    source_score_sha256: str
    output_frames: int
    output_channels: int
    sample_rate: int
    rendered_notes: int
    skipped_percussion_notes: int
    peak_polyphony: int
    output_peak: float
    output_rms: float
    protocol: str = "sonumio-symbolic-wavetable-bridge-v1"

    def as_dict(self):
        result = asdict(self)
        result["original_mathematical_synthesis"] = True
        result["trained_model_inference"] = False
        result["perceptual_quality_established"] = False
        result["source_modified"] = False
        result["external_samples_used"] = False
        return result


@dataclass(frozen=True, slots=True)
class BridgeResult:
    audio: Any
    report: BridgeReceipt


def build_wavetable_plan(score: Score,
                         options: WavetableBridgeSettings = WavetableBridgeSettings()):
    """Construct Sonumio's own wavetable plan and MIDI notes, not a new renderer."""
    if not isinstance(score, Score) or not isinstance(options, WavetableBridgeSettings):
        raise MidiWorkstationError("Expected Score and WavetableBridgeSettings")
    from sonumio.audio_core.wavetable import (
        WavetablePlan, WavetableNote, VoiceEnvelope,
    )
    transport = TransportSettings(
        sample_rate=options.sample_rate,
        tail_frames=options.release_frames,
        include_muted=options.include_muted,
    )
    schedule = build_schedule(score, transport)
    pitched_channels = {t.id for t in score.tracks if t.pitched}
    selected = [n for n in schedule.notes if n.track_id in pitched_channels]
    skipped = len(schedule.notes) - len(selected)
    if len(selected) > options.maximum_notes:
        raise MidiLimitError("Score exceeds wavetable event limit")
    if options.channels == 1 and any(n.pan != 64 for n in selected):
        raise MidiWorkstationError("Mono wavetable output requires centered note pan")
    notes = []
    for item in selected:
        # The output clock is stable; note gate is the difference between
        # separately rounded endpoint frames, not rounded duration seconds.
        note = WavetableNote(
            start_frame=item.start_frame,
            gate_frames=item.frames,
            midi_note=item.pitch,
            velocity=float(item.velocity_fraction),
            pan=float(item.pan_signed),
            initial_phase=0.,
        )
        notes.append(note)
    envelope = VoiceEnvelope(
        attack_frames=options.attack_frames,
        decay_frames=options.decay_frames,
        sustain_level=options.sustain_level,
        release_frames=options.release_frames,
    )
    plan = WavetablePlan(
        sample_rate=options.sample_rate,
        frames=schedule.output_frames,
        channels=options.channels,
        waveform=options.waveform,
        table_size=options.table_size,
        max_harmonics=options.max_harmonics,
        master_gain=options.master_gain,
        envelope=envelope,
    )
    return plan, tuple(notes), schedule, skipped


def render_score_wavetable(score: Score,
                           options: WavetableBridgeSettings = WavetableBridgeSettings()) -> BridgeResult:
    if not isinstance(score, Score) or not isinstance(options, WavetableBridgeSettings):
        raise MidiWorkstationError("Expected Score and WavetableBridgeSettings")
    from sonumio.audio_core.wavetable import render_wavetable
    plan, notes, schedule, skipped = build_wavetable_plan(score, options)
    result = render_wavetable(plan, notes)
    return BridgeResult(result.audio, BridgeReceipt(
        score.sha256, result.audio.frames, result.audio.channels,
        result.audio.sample_rate, len(notes), skipped,
        result.report.maximum_polyphony, result.report.peak, result.report.rms,
    ))
