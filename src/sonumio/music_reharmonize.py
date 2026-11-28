"""Conservative melody-conditioned diatonic reharmonization.

The score clock, lead, percussion, sections and accompaniment event geometry
are immutable. Each changed bar is revoiced using an injective ordered
mapping of the distinct pitches on each accompaniment track. This preserves
the original overlap graph, avoids duplicate simultaneous pitches, and
guarantees that every changed accompaniment note belongs to the new chord.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

from .arrangement import (
    Arrangement, arrangement_digest, chord_classes, event_key, integer,
)
from .errors import GenerationError
from .music import TICKS_PER_BAR, TICKS_PER_BEAT
from .music_automation import _object, _unique, _constant

PROTOCOL = "sonumio-melody-reharmonization-v1"
ACCOMPANIMENT = ("keys", "pad", "bass", "pluck")
MAX_DOCUMENT_BYTES = 32 * 1024
MAX_DISTINCT_PITCHES = 24
MAX_VOICE_SHIFT = 24


@dataclass(frozen=True, slots=True)
class HarmonyPlan:
    score_sha256: str
    first_bar: int = 0
    bars: int = 4
    degrees: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    maximum_shift: int = 12
    melody_weight: int = 4
    movement_weight: int = 1
    identity_weight: int = 2
    voice_weight: int = 1
    preserve_edges: bool = False
    dominant_cadence: bool = False

    def __post_init__(self):
        if (type(self.score_sha256) is not str or len(self.score_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.score_sha256)):
            raise GenerationError("Invalid reharmonization score SHA256")
        integer(self.first_bar, 0, 31, "harmony first bar")
        integer(self.bars, 1, 32, "harmony bar count")
        if (type(self.degrees) is not tuple or not self.degrees
                or any(type(d) is not int or not 0 <= d <= 6 for d in self.degrees)
                or self.degrees != tuple(sorted(set(self.degrees)))):
            raise GenerationError("Expected ordered unique diatonic degrees")
        integer(self.maximum_shift, 0, MAX_VOICE_SHIFT, "harmony maximum shift")
        for value in (self.melody_weight, self.movement_weight, self.identity_weight, self.voice_weight):
            integer(value, 0, 16, "harmony objective weight")
        if not any((self.melody_weight, self.movement_weight, self.identity_weight, self.voice_weight)):
            raise GenerationError("Harmony objective must have a positive weight")
        if type(self.preserve_edges) is not bool or type(self.dominant_cadence) is not bool:
            raise GenerationError("Harmony switches must be boolean")

    @property
    def sha256(self):
        return hashlib.sha256(encode_harmony_plan(self)).hexdigest()


def encode_harmony_plan(plan):
    if not isinstance(plan, HarmonyPlan):
        raise GenerationError("Expected HarmonyPlan")
    return json.dumps({"format": PROTOCOL, **asdict(plan)}, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def decode_harmony_plan(data):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_DOCUMENT_BYTES:
        raise GenerationError("Invalid harmony plan byte count")
    try:
        obj = _object(json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                 parse_constant=_constant),
                      ("format", "score_sha256"),
                      tuple(name for name in HarmonyPlan.__dataclass_fields__
                            if name != "score_sha256"))
        if obj["format"] != PROTOCOL:
            raise GenerationError("Unknown harmony plan format")
        fields = {name: value for name, value in obj.items() if name != "format"}
        if "degrees" in fields:
            if type(fields["degrees"]) is not list:
                raise GenerationError("Expected degree array")
            fields["degrees"] = tuple(fields["degrees"])
        return HarmonyPlan(**fields)
    except (UnicodeError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed harmony plan") from exc


def read_harmony_plan(path):
    with open(path, "rb") as stream:
        return decode_harmony_plan(stream.read(MAX_DOCUMENT_BYTES+1))


def ordered_voicing(pitches, pitch_classes, maximum_shift):
    """Find minimum total movement with strict order and distinct targets.

    Every distinct source pitch is weighted by its occurrence count.
    Strictly increasing target pitches preserve injectivity even when
    multiple notes overlap. A path state retains only a cost and a tuple
    of at most 24 pitches, and the domain has at most 73 MIDI pitches.
    """
    integer(maximum_shift, 0, MAX_VOICE_SHIFT, "harmony maximum shift")
    if (type(pitches) is not tuple or len(pitches) > MAX_DISTINCT_PITCHES
            or any(type(row) is not tuple or len(row) != 2 or
                   type(row[0]) is not int or not 24 <= row[0] <= 96 or
                   type(row[1]) is not int or not 1 <= row[1] <= 4096 for row in pitches)
            or tuple(p for p, _ in pitches) != tuple(sorted({p for p, _ in pitches}))):
        raise GenerationError("Invalid ordered voicing source pitches")
    if (type(pitch_classes) is not tuple or len(pitch_classes) != 3
            or len(set(pitch_classes)) != 3
            or any(type(p) is not int or not 0 <= p <= 11 for p in pitch_classes)):
        raise GenerationError("Expected three distinct chord pitch classes")
    if not pitches:
        return {}, 0
    states = {}
    for index, (source, count) in enumerate(pitches):
        candidates = [p for p in range(max(24, source-maximum_shift),
                                      min(96, source+maximum_shift)+1)
                      if p % 12 in pitch_classes]
        current = {}
        for pitch in candidates:
            movement = abs(pitch-source)*count
            if index == 0:
                current[pitch] = (movement, (pitch,))
            else:
                prior = [entry for key, entry in states.items() if key < pitch]
                if prior:
                    cost, path = min(prior)
                    current[pitch] = (cost+movement, (*path, pitch))
        if not current:
            return None, None
        states = current
    cost, path = min(states.values())
    return dict(zip((p for p, _ in pitches), path, strict=True)), cost


def _bar_events(score, bar, track):
    start, stop = bar*TICKS_PER_BAR, (bar+1)*TICKS_PER_BAR
    return tuple(e for e in score.events if e.track == track and
                 e.tick < stop and e.tick+e.length > start)


def melody_mismatch(score, bar, degree):
    """Duration-weighted nonchord lead ticks, including sustained notes."""
    classes = chord_classes(score.plan, degree)
    start, stop = bar*TICKS_PER_BAR, (bar+1)*TICKS_PER_BAR
    penalty = 0
    for event in _bar_events(score, bar, "lead"):
        if event.pitch % 12 not in classes:
            overlap = min(stop, event.tick+event.length)-max(start, event.tick)
            penalty += overlap
            if start <= event.tick < stop and event.tick % TICKS_PER_BEAT == 0:
                penalty += TICKS_PER_BEAT
    return penalty


def _candidate(score, bar, degree, plan):
    old = score.harmony[bar]
    if degree == old:
        # Keeping the chord must not rewrite existing accompaniment,
        # including imported or manually edited nonchord notes.
        return {}, 0
    start, stop = bar*TICKS_PER_BAR, (bar+1)*TICKS_PER_BAR
    maps = {}
    cost = 0
    classes = chord_classes(score.plan, degree)
    for track in ACCOMPANIMENT:
        events = _bar_events(score, bar, track)
        if any(e.tick < start or e.tick+e.length > stop for e in events):
            return None, None
        counts = {}
        for event in events:
            counts[event.pitch] = counts.get(event.pitch, 0)+1
        if len(counts) > MAX_DISTINCT_PITCHES:
            return None, None
        mapping, movement = ordered_voicing(tuple(sorted(counts.items())), classes, plan.maximum_shift)
        if mapping is None:
            return None, None
        maps[track] = mapping
        cost += movement
    return maps, cost


def _transition(plan, score, left, right):
    # Common tones reduce the change cost. The circular degree distance
    # is a small secondary preference, not a music-theory constraint.
    shared = len(set(chord_classes(score.plan, left)) & set(chord_classes(score.plan, right)))
    distance = min((left-right)%7, (right-left)%7)
    return plan.movement_weight * (3-shared+distance)


def _allowed(score, bar, plan):
    degrees = set(plan.degrees)
    if plan.preserve_edges and bar in (plan.first_bar, plan.first_bar+plan.bars-1):
        degrees &= {score.harmony[bar]}
    if bar == score.plan.bars-1:
        degrees &= {0}
    if plan.dominant_cadence and bar == score.plan.bars-2:
        degrees &= {4}
    return tuple(sorted(degrees))


def optimize_harmony(score, plan):
    """Exact finite shortest path, with lexicographic deterministic ties."""
    if not isinstance(score, Arrangement) or not isinstance(plan, HarmonyPlan):
        raise GenerationError("Expected score and harmony plan")
    if arrangement_digest(score) != plan.score_sha256:
        raise GenerationError("Harmony source score SHA256 mismatch")
    stop = plan.first_bar+plan.bars
    if stop > score.plan.bars:
        raise GenerationError("Harmony region exceeds score")
    candidates = {}
    states = {}
    counts = []
    for bar in range(plan.first_bar, stop):
        row = {}
        for degree in _allowed(score, bar, plan):
            maps, movement = _candidate(score, bar, degree, plan)
            if maps is None:
                continue
            local = (plan.melody_weight*melody_mismatch(score, bar, degree)
                     + plan.identity_weight*int(degree != score.harmony[bar])
                     + plan.voice_weight*movement)
            candidates[bar, degree] = (maps, movement, local)
            if bar == plan.first_bar:
                if bar:
                    local += _transition(plan, score, score.harmony[bar-1], degree)
                row[degree] = (local, (degree,))
            else:
                row[degree] = min((cost+local+_transition(plan, score, previous, degree),
                                   (*path, degree))
                                  for previous, (cost, path) in states.items())
        if not row:
            raise GenerationError(f"No feasible harmony candidate at bar {bar}")
        states = row
        counts.append(len(row))
    if stop < score.plan.bars:
        states = {degree: (cost+_transition(plan, score, degree, score.harmony[stop]), path)
                  for degree, (cost, path) in states.items()}
    cost, path = min(states.values())
    return path, cost, candidates, tuple(counts)


def reharmonize(score, plan):
    path, cost, candidates, counts = optimize_harmony(score, plan)
    harmony = list(score.harmony)
    events = list(score.events)
    voice_movement = 0
    changed_events = 0
    for bar, degree in enumerate(path, start=plan.first_bar):
        maps, movement, _ = candidates[bar, degree]
        voice_movement += movement
        harmony[bar] = degree
        for index, event in enumerate(score.events):
            if event.track not in maps or event.tick//TICKS_PER_BAR != bar:
                continue
            pitch = maps[event.track][event.pitch]
            if pitch != event.pitch:
                events[index] = replace(event, pitch=pitch)
                changed_events += 1
    result = replace(score, harmony=tuple(harmony), events=tuple(sorted(events, key=event_key)))
    # The immutable result is revalidated by Arrangement construction.
    # A successful transformation preserves all non-pitch event fields.
    return result, {
        "protocol": PROTOCOL, "score_sha256": plan.score_sha256,
        "harmony_plan_sha256": plan.sha256,
        "result_score_sha256": arrangement_digest(result),
        "first_bar": plan.first_bar, "bars": plan.bars,
        "original_degrees": list(score.harmony[plan.first_bar:plan.first_bar+plan.bars]),
        "result_degrees": list(path), "objective_cost": cost,
        "candidate_counts": list(counts), "voice_movement": voice_movement,
        "changed_events": changed_events,
        "changed_bars": sum(a != b for a,b in zip(score.harmony, harmony, strict=True)),
        "melody_preserved": True, "percussion_preserved": True,
        "rhythm_preserved": True, "learned_model": False,
        "external_recordings_used": False,
    }
