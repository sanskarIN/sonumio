"""Opt-in legacy Sonumio arrangement-to-workstation score conversion.

The legacy arrangement owns sections, harmonic mode, style, swing and sample
clock. A MIDI workstation score has note identity, fixed patches and PPQN.
The bridge never silently claims exact reconstruction of lost semantics.
It requires explicit channel/patch destinations and explicit permission for
7-bit velocity/pan quantization or swing removal.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from .contracts import MAX_PPQ, MidiWorkstationError, MidiLimitError, integer
from .score import Score, ScoreNote, ScoreTrack, NOTE_ID_RE, note_sort_key
from .tempo import TempoPoint, MeterPoint

IMPORT_PROTOCOL = "sonumio-arrangement-to-workstation-v1"
LEGACY_INSTRUMENTS = ("keys", "pad", "bass", "lead", "pluck", "kick", "snare", "hat")
PERCUSSION = frozenset(LEGACY_INSTRUMENTS[-3:])
MAX_IMPORT_EVENTS = 4096


@dataclass(frozen=True, slots=True)
class LegacyInstrumentRoute:
    instrument: str
    track_id: str
    channel: int
    program: int = 0
    bank_msb: int = 0
    bank_lsb: int = 0
    mute: bool = False

    def __post_init__(self):
        if type(self.instrument) is not str or self.instrument not in LEGACY_INSTRUMENTS:
            raise MidiWorkstationError("Unknown legacy arrangement instrument")
        if type(self.track_id) is not str or not NOTE_ID_RE.fullmatch(self.track_id):
            raise MidiWorkstationError("Invalid target workstation track ID")
        integer(self.channel, "MIDI channel", 0, 15)
        integer(self.program, "General MIDI program", 0, 127)
        integer(self.bank_msb, "MIDI bank MSB", 0, 127)
        integer(self.bank_lsb, "MIDI bank LSB", 0, 127)
        if type(self.mute) is not bool:
            raise MidiWorkstationError("Track mute flag must be bool")
        if (self.instrument in PERCUSSION) != (self.channel == 9):
            raise MidiWorkstationError("Drum instrument must use MIDI percussion channel 9")


@dataclass(frozen=True, slots=True)
class LegacyImportPolicy:
    routes: tuple[LegacyInstrumentRoute, ...]
    ppqn: int = 480
    allow_velocity_quantization: bool = False
    allow_pan_quantization: bool = False
    allow_swing_loss: bool = False
    maximum_events: int = MAX_IMPORT_EVENTS

    def __post_init__(self):
        if (type(self.routes) is not tuple or not 1 <= len(self.routes) <= len(LEGACY_INSTRUMENTS)
                or any(not isinstance(route, LegacyInstrumentRoute) for route in self.routes)):
            raise MidiWorkstationError("Expected explicit bounded legacy route tuple")
        instruments = [r.instrument for r in self.routes]
        track_ids = [r.track_id for r in self.routes]
        if instruments != sorted(set(instruments)) or len(set(track_ids)) != len(track_ids):
            raise MidiWorkstationError("Legacy routes must be unique and sorted by instrument")
        integer(self.ppqn, "MIDI PPQN", 8, MAX_PPQ)
        if self.ppqn % 8:
            raise MidiWorkstationError("MIDI PPQN must be a multiple of 8 for exact grid")
        for name, value in (
            ("velocity quantization", self.allow_velocity_quantization),
            ("pan quantization", self.allow_pan_quantization),
            ("swing-loss policy", self.allow_swing_loss),
        ):
            if type(value) is not bool:
                raise MidiWorkstationError(name + " must be boolean")
        integer(self.maximum_events, "maximum imported legacy events", 1, MAX_IMPORT_EVENTS)


def _round_nearest_positive(value: float) -> int:
    """Deterministic round half-up for nonnegative finite values."""
    if not math.isfinite(value) or value < 0:
        raise MidiWorkstationError("Invalid legacy quantization input")
    return int(math.floor(value + .5))


def _velocity(value: float) -> tuple[int, float]:
    midi = max(1, min(127, _round_nearest_positive(value * 127)))
    return midi, abs((midi / 127.) - value)


def _pan(value: float) -> tuple[int, float]:
    midi = _round_nearest_positive(64. + value * (63. if value >= 0 else 64.))
    midi = max(0, min(127, midi))
    recovered = ((midi - 64) / (63. if midi > 64 else 64.))
    return midi, abs(recovered - value)


def arrangement_to_score(arrangement, policy: LegacyImportPolicy) -> tuple[Score, dict]:
    from sonumio.arrangement import Arrangement, arrangement_digest
    if not isinstance(arrangement, Arrangement):
        raise MidiWorkstationError("Expected canonical Sonumio Arrangement")
    if not isinstance(policy, LegacyImportPolicy):
        raise MidiWorkstationError("Expected explicit LegacyImportPolicy")
    initial_digest = arrangement_digest(arrangement)
    if arrangement.plan.swing and not policy.allow_swing_loss:
        raise MidiWorkstationError("Arrangement swing loss requires explicit opt in")
    instruments = {event.track for event in arrangement.events}
    routes = {row.instrument: row for row in policy.routes}
    if set(routes) != instruments:
        raise MidiWorkstationError("Routes must cover exactly note-bearing legacy instruments")
    if len(arrangement.events) > policy.maximum_events:
        raise MidiLimitError("Legacy arrangement import exceeds event budget")
    timing_factor = policy.ppqn // 8
    score_end = arrangement.plan.ticks * timing_factor
    event_notes: dict[str, list[ScoreNote]] = {k: [] for k in routes}
    velocity_quantized = 0
    pan_quantized = 0
    maximum_velocity_error = 0.
    maximum_pan_error = 0.
    for index, event in enumerate(arrangement.events):
        velocity, ve = _velocity(event.velocity)
        pan, pe = _pan(event.pan)
        maximum_velocity_error = max(maximum_velocity_error, ve)
        maximum_pan_error = max(maximum_pan_error, pe)
        if ve > 1e-12:
            velocity_quantized += 1
            if not policy.allow_velocity_quantization:
                raise MidiWorkstationError(
                    "Legacy velocity cannot be encoded in 7-bit MIDI without quantization"
                )
        if pe > 1e-12:
            pan_quantized += 1
            if not policy.allow_pan_quantization:
                raise MidiWorkstationError(
                    "Legacy pan cannot be encoded in 7-bit MIDI without quantization"
                )
        start = event.tick * timing_factor
        duration = event.length * timing_factor
        if duration <= 0 or start + duration > score_end:
            raise MidiWorkstationError("Legacy note exceeds converted MIDI time")
        event_notes[event.track].append(ScoreNote(
            f"legacy-{index:05d}", start, duration, event.pitch,
            velocity, 0, pan,
        ))
    tracks = []
    for instrument in sorted(routes):
        row = routes[instrument]
        tracks.append(ScoreTrack(
            row.track_id, instrument.title(), row.channel,
            row.program, row.bank_msb, row.bank_lsb,
            tuple(sorted(event_notes[instrument], key=note_sort_key)),
            muted=row.mute,
        ))
    tempo = (TempoPoint(0, (60_000_000+arrangement.plan.bpm//2)//arrangement.plan.bpm),)
    meter = (MeterPoint(0, 4, 4),)
    score = Score(
        "Imported original legacy arrangement", policy.ppqn,
        tuple(tracks), tempo, meter, score_end,
    )
    if arrangement_digest(arrangement) != initial_digest:
        raise MidiWorkstationError("Legacy source changed during conversion")
    ledger = {
        "protocol": IMPORT_PROTOCOL,
        "source_arrangement_sha256": initial_digest,
        "score_sha256": score.sha256,
        "source_bpm": arrangement.plan.bpm,
        "score_tempo_us_per_quarter": tempo[0].microseconds_per_quarter,
        "ppqn": policy.ppqn,
        "events": len(arrangement.events),
        "notes": score.note_count,
        "velocity_quantized": velocity_quantized,
        "pan_quantized": pan_quantized,
        "maximum_velocity_error": maximum_velocity_error,
        "maximum_pan_error": maximum_pan_error,
        "swing_discarded": bool(arrangement.plan.swing),
        "sections_harmony_and_root_metadata_lost": True,
        "legacy_sample_frame_rounding_preserved": False,
        "release_velocity_default": 0,
        "routes": [
            {"instrument": r.instrument, "track_id": r.track_id,
             "channel": r.channel, "program": r.program,
             "bank_msb": r.bank_msb, "bank_lsb": r.bank_lsb,
             "muted": r.mute}
            for r in policy.routes
        ],
        "exact_original_score_reconstructed": False,
        "audio_rendered": False,
        "trained_model_used": False,
    }
    return score, ledger
