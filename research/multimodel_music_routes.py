"""Explicit routing for the independently trained short, sustained and drum models.

This version is research-only. No model loading, downloads, or filesystem
publication. Routing uses the original score clock and excludes exact
training controls. Every unsupported note remains procedural.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from research.instrument_note_data import (
    INSTRUMENTS, MIN_PITCH, MAX_PITCH, MIN_VELOCITY, MAX_VELOCITY,
    MIN_FRAMES as SHORT_MIN, MAX_FRAMES as SHORT_MAX, NoteSpec,
)
from research.instrument_long_data import MIN_LENGTH as LONG_MIN, LongNote
from research.instrument_long_model import MAX_NOTE_FRAMES
from research.percussion_data import DRUMS, MIN_FRAMES as DRUM_MIN, MAX_FRAMES as DRUM_MAX
from research.hybrid_music_render import MAX_SCORE_EVENTS, MAX_SCORE_FRAMES, MAX_EVENT_SAMPLES
from sonumio.arrangement import Arrangement, Event, TRACKS, arrangement_digest

PROTOCOL = "sonumio-multimodel-score-routing-v1"
FAMILIES = ("short", "sustained", "percussion")
FALLBACKS = ("pitch", "velocity", "duration", "training_control", "training_seed")
MAX_EXCLUSIONS = 1024


def _digest_set(values):
    if (type(values) is not frozenset or len(values) > MAX_EXCLUSIONS
            or any(type(x) is not str or len(x) != 64
                   or any(c not in "0123456789abcdef" for c in x) for x in values)):
        raise ValueError("Invalid multi-model training identity exclusions")


@dataclass(frozen=True, slots=True)
class Exclusions:
    short: frozenset[str] = frozenset()
    sustained: frozenset[str] = frozenset()
    drum_seeds: frozenset[int] = frozenset()

    def __post_init__(self):
        _digest_set(self.short)
        _digest_set(self.sustained)
        if (type(self.drum_seeds) is not frozenset or len(self.drum_seeds) > MAX_EXCLUSIONS
                or any(type(s) is not int or not 0 <= s < 2**32 for s in self.drum_seeds)):
            raise ValueError("Invalid percussion training seed exclusions")


@dataclass(frozen=True, slots=True)
class Route:
    event: Event
    start: int
    stop: int
    family: str | None
    reason: str | None
    identity: str

    def __post_init__(self):
        if (not isinstance(self.event, Event) or type(self.start) is not int
                or type(self.stop) is not int or not 0 <= self.start < self.stop <= MAX_SCORE_FRAMES):
            raise ValueError("Invalid multi-model event geometry")
        if (self.family is None and self.reason not in FALLBACKS
                or self.family is not None and (self.family not in FAMILIES or self.reason is not None)):
            raise ValueError("Invalid multi-model routing decision")
        if self.family == "percussion" and self.event.track not in DRUMS:
            raise ValueError("Non-percussion event routed to drums")
        if self.family in ("short", "sustained") and self.event.track not in INSTRUMENTS:
            raise ValueError("Percussion routed to pitched model")
        if type(self.identity) is not str or len(self.identity) != 64 or any(c not in "0123456789abcdef" for c in self.identity):
            raise ValueError("Invalid multi-model event digest")

    @property
    def frames(self):
        return self.stop - self.start

    @property
    def velocity_milli(self):
        return round(self.event.velocity * 1000)


def _event_identity(score, event, start, stop):
    # The source score digest makes identities independent of traversal and
    # unrelated to the source/learned training excitation namespaces.
    payload = json.dumps([arrangement_digest(score), event.track, event.pitch,
                          event.tick, event.length, event.velocity, event.pan,
                          start, stop], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(PROTOCOL.encode("ascii") + b"\0" + payload.encode("ascii")).hexdigest()


def route_event(score, event, exclusions):
    if not isinstance(score, Arrangement) or not isinstance(event, Event) or not isinstance(exclusions, Exclusions):
        raise ValueError("Expected validated routing inputs")
    start = score.plan.frame(event.tick)
    stop = score.plan.frame(event.tick + event.length)
    n = stop - start
    velocity = round(event.velocity * 1000)
    family = reason = None
    if not MIN_VELOCITY <= velocity <= MAX_VELOCITY:
        reason = "velocity"
    elif event.track in DRUMS:
        if score.plan.seed in exclusions.drum_seeds:
            reason = "training_seed"
        elif DRUM_MIN <= n <= DRUM_MAX:
            family = "percussion"
        else:
            reason = "duration"
    elif not MIN_PITCH <= event.pitch <= MAX_PITCH:
        reason = "pitch"
    elif SHORT_MIN <= n <= SHORT_MAX:
        spec = NoteSpec(event.track, event.pitch, velocity, n)
        if spec.identity in exclusions.short:
            reason = "training_control"
        else:
            family = "short"
    elif LONG_MIN <= n <= MAX_NOTE_FRAMES:
        spec = LongNote(event.track, event.pitch, velocity, n)
        if spec.identity in exclusions.sustained:
            reason = "training_control"
        else:
            family = "sustained"
    else:
        # Do not quietly extrapolate into the unqualified gap between the
        # short and long research grids.
        reason = "duration"
    return Route(event, start, stop, family, reason, _event_identity(score, event, start, stop))


def route_score(score, exclusions=None):
    exclusions = Exclusions() if exclusions is None else exclusions
    if not isinstance(score, Arrangement) or not isinstance(exclusions, Exclusions):
        raise ValueError("Expected arrangement and training exclusions")
    frames = score.plan.frame(score.plan.ticks)
    if not 1 <= frames <= MAX_SCORE_FRAMES or len(score.events) > MAX_SCORE_EVENTS:
        raise ValueError("Multi-model score exceeds preview budget")
    routes = tuple(route_event(score, event, exclusions) for event in score.events)
    if sum(row.frames for row in routes) > MAX_EVENT_SAMPLES:
        raise ValueError("Multi-model event sample budget exceeded")
    if len({row.identity for row in routes}) != len(routes):
        raise ValueError("Duplicate multi-model score event identities")
    return routes


def describe_routes(routes):
    if type(routes) is not tuple or not routes or any(not isinstance(row, Route) for row in routes):
        raise ValueError("Expected canonical multi-model route tuple")
    counts = Counter(row.family or "procedural" for row in routes)
    samples = Counter()
    for row in routes:
        samples[row.family or "procedural"] += row.frames
    family_names = (*FAMILIES, "procedural")
    track_summary = {}
    for track in TRACKS:
        rows = tuple(row for row in routes if row.event.track == track)
        track_summary[track] = {
            "events": len(rows), "samples": sum(row.frames for row in rows),
            "families": {name: sum((row.family or "procedural") == name for row in rows) for name in family_names},
        }
    return {"events": len(routes), "event_samples": sum(row.frames for row in routes),
            "events_by_family": {name: counts[name] for name in family_names},
            "samples_by_family": {name: samples[name] for name in family_names},
            "fallback_reasons": dict(sorted(Counter(row.reason for row in routes if row.reason).items())),
            "tracks": track_summary}


def event_excitation(route, replica):
    if not isinstance(route, Route) or route.family != "percussion":
        raise ValueError("Expected routed percussion event")
    if type(replica) is not int or not 0 <= replica < 2**32:
        raise ValueError("Invalid score excitation replica")
    digest = hashlib.sha256(b"sonumio-score-excitation-v1\0" + bytes.fromhex(route.identity)
                            + replica.to_bytes(4, "little")).digest()
    rng = np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4").tolist()))
    return rng.uniform(-1., 1., route.frames + 2).astype(np.float32)
