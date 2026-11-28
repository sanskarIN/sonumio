"""Opt-in, fail-closed interoperability with Sonumio's long-form arrangement.

The two score models have different clocks, identity models and metadata.
Never reinterpret a workstation score as an arrangement without an explicit
validated arrangement template and a complete track routing policy. Only
monotonic eighth-beat grids at one constant tempo in 4/4 are supported.

This bridge does not generate any music, write to disk, or mutate an input.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .contracts import MidiWorkstationError, MidiLimitError, integer
from .score import Score

BRIDGE_PROTOCOL = "sonumio-workstation-arrangement-bridge-v1"
MAX_ROUTES = 48
MAX_TEMPLATE_EVENTS = 4096
TARGETS = ("keys", "pad", "bass", "lead", "pluck", "kick", "snare", "hat")
PITCHED_TARGETS = frozenset(TARGETS[:5])
DRUM_TARGETS = frozenset(TARGETS[5:])


@dataclass(frozen=True, slots=True)
class ArrangementRoute:
    """Explicit source track-ID to legacy arrangement-instrument mapping."""

    source_track_id: str
    target_instrument: str
    transpose: int = 0

    def __post_init__(self) -> None:
        from .score import NOTE_ID_RE
        if type(self.source_track_id) is not str or not NOTE_ID_RE.fullmatch(self.source_track_id):
            raise MidiWorkstationError("Invalid bridge source track identifier")
        if type(self.target_instrument) is not str or self.target_instrument not in TARGETS:
            raise MidiWorkstationError("Unknown arrangement instrument destination")
        integer(self.transpose + 48 if type(self.transpose) is int else self.transpose,
                "arrangement transposition + 48", 24, 72)


@dataclass(frozen=True, slots=True)
class ArrangementBridgePolicy:
    """No implicit track inference, quantization, mode changes or note drops."""

    routes: tuple[ArrangementRoute, ...]
    include_muted: bool = False
    preserve_note_pan: bool = True
    maximum_events: int = MAX_TEMPLATE_EVENTS

    def __post_init__(self) -> None:
        if (type(self.routes) is not tuple or not 1 <= len(self.routes) <= MAX_ROUTES
                or any(not isinstance(route, ArrangementRoute) for route in self.routes)):
            raise MidiWorkstationError("Expected 1..48 explicit route mappings")
        sources = [route.source_track_id for route in self.routes]
        if sources != sorted(set(sources)):
            raise MidiWorkstationError("Bridge mappings must be unique and sorted by source track")
        if type(self.include_muted) is not bool or type(self.preserve_note_pan) is not bool:
            raise MidiWorkstationError("Bridge policy flags must be boolean")
        integer(self.maximum_events, "maximum bridged events", 1, MAX_TEMPLATE_EVENTS)


def _clock(score: Score, template) -> tuple[int, int]:
    """Reject every timing assumption that cannot be represented exactly."""
    from sonumio.arrangement import Arrangement
    if not isinstance(score, Score) or not isinstance(template, Arrangement):
        raise MidiWorkstationError("Expected validated workstation Score and Arrangement template")
    plan = template.plan
    if plan.swing != 0:
        raise MidiWorkstationError("Template swing cannot be represented by the straight score clock")
    if (len(score.tempo) != 1 or score.tempo[0].tick != 0
            or score.tempo[0].microseconds_per_quarter !=
            (60_000_000 + plan.bpm // 2) // plan.bpm):
        raise MidiWorkstationError("Score tempo must exactly equal template constant tempo")
    if (len(score.meter) != 1 or score.meter[0].tick != 0 or
            (score.meter[0].numerator, score.meter[0].denominator) != (4, 4)):
        raise MidiWorkstationError("Score meter must be constant 4/4")
    if score.end * 8 != plan.ticks * score.ppqn:
        raise MidiWorkstationError("Score and template must have identical exact musical duration")
    return score.ppqn, plan.ticks


def _score_tick(tick: int, ppqn: int, total: int) -> int:
    """Convert PPQN tick to eighth-beat grid without snapping or rounding."""
    numerator = tick * 8
    result, remainder = divmod(numerator, ppqn)
    if remainder or not 0 <= result <= total:
        raise MidiWorkstationError("Score note endpoint falls off arrangement eighth-beat grid")
    return result


def _score_pan(pan: int) -> float:
    """Map discrete 7-bit score pan to centered arrangement [-1,1] pan."""
    return (pan - 64) / (63. if pan > 64 else 64.)


def _active_tracks(score: Score, include_muted: bool) -> set[str]:
    if include_muted:
        return {track.id for track in score.tracks}
    return set(score.active_track_ids)


def bridge_to_arrangement(
    score: Score, template, policy: ArrangementBridgePolicy,
) -> tuple[object, dict]:
    """Convert strict score events using a supplied canonical template.

    Only the template's events change. Harmony, key, sections and render
    clock remain template-owned. Unsupported expression data is *reported*
    rather than claimed reconstructed. Validation is atomic; a malformed
    note or polyphony clash never produces a partial arrangement.
    """
    from sonumio.arrangement import (
        Arrangement, Event, TRACKS, arrangement_digest, event_key,
    )
    from sonumio.errors import GenerationError

    if not isinstance(policy, ArrangementBridgePolicy):
        raise MidiWorkstationError("Expected explicit ArrangementBridgePolicy")
    ppqn, ticks = _clock(score, template)
    original_score_digest = score.sha256
    template_digest = arrangement_digest(template)
    routes = {route.source_track_id: route for route in policy.routes}
    all_source_ids = {track.id for track in score.tracks}
    if not set(routes) <= all_source_ids:
        raise MidiWorkstationError("Bridge route references unknown source track")
    included = _active_tracks(score, policy.include_muted)
    note_bearing = {track.id for track in score.tracks if track.notes and track.id in included}
    # Explicit route for every included note-bearing track, no unused routes.
    if set(routes) != note_bearing:
        raise MidiWorkstationError("Bridge routes must cover exactly included note-bearing tracks")

    converted = []
    ignored_muted = 0
    event_by_target = {target: 0 for target in TRACKS}
    note_pan_values = 0
    for track in score.tracks:
        if track.id not in included:
            ignored_muted += len(track.notes)
            continue
        route = routes.get(track.id)
        if route is None:
            continue
        # A source MIDI drum kit cannot turn into a harmonic instrument and
        # vice versa without changing synthesis semantics.
        if track.channel == 9 and route.target_instrument not in DRUM_TARGETS:
            raise MidiWorkstationError("Percussion MIDI channel needs percussion arrangement target")
        if track.channel != 9 and route.target_instrument not in PITCHED_TARGETS:
            raise MidiWorkstationError("Pitched MIDI channel cannot map to arrangement drums")
        for note in track.notes:
            pitch = note.pitch + route.transpose
            if not 24 <= pitch <= 96:
                raise MidiWorkstationError("Bridged note pitch outside arrangement range 24..96")
            start = _score_tick(note.start, ppqn, ticks)
            stop = _score_tick(note.stop, ppqn, ticks)
            if start >= stop:
                raise MidiWorkstationError("Bridged note duration is non-positive")
            pan = _score_pan(note.pan) if policy.preserve_note_pan else 0.
            if note.pan != 64:
                note_pan_values += 1
            converted.append(Event(route.target_instrument, pitch, start, stop-start,
                                   note.velocity / 127., pan))
            event_by_target[route.target_instrument] += 1
            if len(converted) > policy.maximum_events:
                raise MidiLimitError("Bridged arrangement exceeds explicit event budget")
    if not converted:
        raise MidiWorkstationError("Legacy arrangement requires at least one converted note")
    try:
        result = replace(template, events=tuple(sorted(converted, key=event_key)))
    except GenerationError as exc:
        # This catches scale exclusions, same-pitch overlaps, polyphony,
        # excessive event duration and duplicate legacy event identities.
        raise MidiWorkstationError("Bridged arrangement violates legacy composition constraints: "
                                   + str(exc)) from exc
    if score.sha256 != original_score_digest or arrangement_digest(template) != template_digest:
        raise MidiWorkstationError("Input score or template mutated during conversion")
    receipt = {
        "protocol": BRIDGE_PROTOCOL,
        "source_score_sha256": original_score_digest,
        "template_arrangement_sha256": template_digest,
        "output_arrangement_sha256": arrangement_digest(result),
        "events": len(converted),
        "events_by_instrument": event_by_target,
        "skipped_muted_notes": ignored_muted,
        "routes": [
            {"source_track_id": route.source_track_id,
             "target_instrument": route.target_instrument,
             "transpose": route.transpose}
            for route in policy.routes
        ],
        "clock_source": "template constant BPM and 4/4",
        "exact_tick_grid": True,
        "harmony_source": "template",
        "sections_source": "template",
        "root_mode_source": "template",
        "velocity_source": "MIDI 7-bit velocity divided by 127",
        "pan_source": "MIDI 7-bit pan" if policy.preserve_note_pan else "explicit centered override",
        "note_pan_values": note_pan_values,
        "discarded_expression": [
            "stable_note_ids", "release_velocity", "MIDI_channel_and_program",
            "bank_select", "muted_and_solo_metadata", "score_revision",
        ],
        "exact_original_score_reconstructed": False,
        "trained_model_used": False,
    }
    return result, receipt
