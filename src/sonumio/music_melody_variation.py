"""Constrained original melodic variation on the existing score clock.

A bounded dynamic program selects a monophonic pitch path. It changes
only pitches on one pitched monophonic track, preserving note rhythm,
articulation, accompaniment and harmony. All pitch and cadence constraints
are checked before the new immutable score is returned.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import hashlib
import json

from .arrangement import (
    Arrangement, arrangement_digest, chord_classes, event_key, integer, real,
)
from .music import MODES, TICKS_PER_BEAT, TICKS_PER_BAR
from .errors import GenerationError
from .music_automation import _object, _unique, _constant

PROTOCOL = "sonumio-constrained-melody-variation-v1"
TRACKS = ("bass", "lead", "pluck")
MODES_OF_CHANGE = ("original", "mirror", "arch", "ascending", "descending")
MAX_NOTES = 512
MAX_STATES = 73
MAX_TRANSITIONS = 12_000_000
MAX_DOCUMENT_BYTES = 32 * 1024


@dataclass(frozen=True, slots=True)
class MelodyPlan:
    score_sha256: str
    track: str = "lead"
    first_bar: int = 0
    bars: int = 4
    mode: str = "mirror"
    minimum_pitch: int = 48
    maximum_pitch: int = 84
    max_leap: int = 12
    max_shift: int = 12
    preserve_ends: bool = True
    strong_chord_tones: bool = True
    tonic_end: bool = False
    contour_weight: int = 3
    identity_weight: int = 1
    movement_weight: int = 1

    def __post_init__(self):
        if (type(self.score_sha256) is not str or len(self.score_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.score_sha256)):
            raise GenerationError("Invalid variation score SHA256")
        if type(self.track) is not str or self.track not in TRACKS:
            raise GenerationError("Variation requires a monophonic pitched track")
        if type(self.mode) is not str or self.mode not in MODES_OF_CHANGE:
            raise GenerationError("Unknown melodic variation mode")
        integer(self.first_bar, 0, 31, "variation first bar")
        integer(self.bars, 1, 32, "variation bar count")
        integer(self.minimum_pitch, 24, 96, "variation minimum pitch")
        integer(self.maximum_pitch, self.minimum_pitch, 96, "variation maximum pitch")
        integer(self.max_leap, 1, 24, "variation maximum leap")
        integer(self.max_shift, 0, 36, "variation maximum shift")
        for flag in (self.preserve_ends, self.strong_chord_tones, self.tonic_end):
            if type(flag) is not bool:
                raise GenerationError("Variation switches must be boolean")
        for weight in (self.contour_weight, self.identity_weight, self.movement_weight):
            integer(weight, 0, 16, "variation objective weight")
        if self.contour_weight+self.identity_weight+self.movement_weight == 0:
            raise GenerationError("Variation objective has no active weights")

    @property
    def sha256(self):
        return hashlib.sha256(encode_melody_plan(self)).hexdigest()


def encode_melody_plan(plan):
    if not isinstance(plan, MelodyPlan):
        raise GenerationError("Expected MelodyPlan")
    return json.dumps({"format": PROTOCOL, **asdict(plan)}, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("ascii")


def decode_melody_plan(data):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_DOCUMENT_BYTES:
        raise GenerationError("Invalid melody variation document byte count")
    try:
        obj = _object(json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                 parse_constant=_constant),
                      ("format", "score_sha256"),
                      tuple(key for key in MelodyPlan.__dataclass_fields__ if key != "score_sha256"))
        if obj["format"] != PROTOCOL:
            raise GenerationError("Unknown melodic variation document")
        return MelodyPlan(**{key: value for key, value in obj.items() if key != "format"})
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed melodic variation document") from exc


def read_melody_plan(path):
    with open(path, "rb") as stream:
        return decode_melody_plan(stream.read(MAX_DOCUMENT_BYTES+1))


def _select(score, plan):
    if not isinstance(score, Arrangement) or not isinstance(plan, MelodyPlan):
        raise GenerationError("Expected arrangement and melody plan")
    if arrangement_digest(score) != plan.score_sha256:
        raise GenerationError("Melodic variation score SHA256 mismatch")
    if plan.first_bar+plan.bars > score.plan.bars:
        raise GenerationError("Melodic variation region exceeds arrangement")
    first = plan.first_bar*TICKS_PER_BAR
    last = (plan.first_bar+plan.bars)*TICKS_PER_BAR
    indexed = [(i,event) for i,event in enumerate(score.events) if event.track==plan.track]
    if any(event.tick < first < event.tick+event.length or event.tick < last < event.tick+event.length
           for _,event in indexed):
        raise GenerationError("Melodic variation boundary cuts a note")
    selected = [(i,event) for i,event in indexed if first<=event.tick<last]
    if not 1<=len(selected)<=MAX_NOTES:
        raise GenerationError("Variation requires 1–512 notes in the selected region")
    before = next((e.pitch for _,e in reversed(indexed) if e.tick<first),None)
    after = next((e.pitch for _,e in indexed if e.tick>=last),None)
    return selected, before, after


def _targets(events, plan):
    pitches = [e.pitch for e in events]
    n = len(pitches)
    if plan.mode == "original":
        return pitches
    if plan.mode == "mirror":
        # Invert about the first note. End constraints remain independent.
        return [2*pitches[0]-pitch for pitch in pitches]
    if n == 1:
        return pitches
    lo = max(plan.minimum_pitch, min(pitches)-plan.max_shift)
    hi = min(plan.maximum_pitch, max(pitches)+plan.max_shift)
    if lo > hi:
        raise GenerationError("No contour target within pitch range")
    span = hi-lo
    if plan.mode == "ascending":
        return [lo+(span*i+(n-1)//2)//(n-1) for i in range(n)]
    if plan.mode == "descending":
        return [hi-(span*i+(n-1)//2)//(n-1) for i in range(n)]
    # Symmetric arch, with an exact integer target at each event.
    return [lo+(2*span*min(i,n-1-i)+(n-1)//2)//(n-1) for i in range(n)]


def _candidates(score, events, plan, before, after):
    scale = {(score.plan.root+p)%12 for p in MODES[score.plan.mode]}
    result = []
    for index,event in enumerate(events):
        allowed = scale
        if plan.strong_chord_tones and event.tick%TICKS_PER_BEAT==0:
            allowed = set(chord_classes(score.plan, score.harmony[event.tick//TICKS_PER_BAR]))
        if plan.tonic_end and index == len(events)-1:
            allowed = allowed & {score.plan.root%12}
        if plan.preserve_ends and index in (0,len(events)-1):
            options = (event.pitch,)
        else:
            options = tuple(range(max(plan.minimum_pitch,event.pitch-plan.max_shift),
                                  min(plan.maximum_pitch,event.pitch+plan.max_shift)+1))
        candidates = tuple(p for p in options
                           if plan.minimum_pitch<=p<=plan.maximum_pitch and p%12 in allowed
                           and (index!=0 or before is None or abs(p-before)<=plan.max_leap)
                           and (index!=len(events)-1 or after is None or abs(p-after)<=plan.max_leap))
        if not candidates:
            raise GenerationError(f"No legal pitch for melody note {index}")
        result.append(candidates)
    return tuple(result)


def _local_cost(pitch, original, target, plan):
    return (plan.contour_weight*abs(pitch-target)
            + plan.identity_weight*abs(pitch-original))


def _transition_cost(previous, pitch, desired_interval, plan):
    if abs(pitch-previous)>plan.max_leap:
        return None
    return (plan.movement_weight*abs(pitch-previous)
            + plan.contour_weight*abs((pitch-previous)-desired_interval))


def optimize_pitch_path(original, targets, candidates, plan, before=None, after=None):
    """Exact finite shortest path with deterministic lexicographic tie breaks.

    Inputs contain one candidate set per original event. Each transition
    connects legal consecutive MIDI pitches. Costs are nonnegative integers.
    Storing prefix ranks, rather than complete paths at each state, bounds
    memory by the number of events times candidate pitches.
    """
    if not isinstance(plan, MelodyPlan):
        raise GenerationError("Expected melody optimization plan")
    n = len(original)
    if not 1<=n<=MAX_NOTES or len(targets)!=n or len(candidates)!=n:
        raise GenerationError("Invalid melody optimization geometry")
    if any(type(p) is not int or not 24<=p<=96 for p in original):
        raise GenerationError("Invalid original optimization pitch")
    if any(type(p) is not int or not -128<=p<=224 for p in targets):
        raise GenerationError("Invalid contour target")
    if any(type(row) is not tuple or not 1<=len(row)<=MAX_STATES
           or any(type(p) is not int or not 24<=p<=96 for p in row)
           or row!=tuple(sorted(set(row))) for row in candidates):
        raise GenerationError("Invalid melody candidate lattice")
    for neighbor in (before,after):
        if neighbor is not None and (type(neighbor) is not int or not 24<=neighbor<=96):
            raise GenerationError("Invalid melody boundary pitch")
    transitions = sum(len(a)*len(b) for a,b in zip(candidates,candidates[1:]))
    if transitions>MAX_TRANSITIONS:
        raise GenerationError("Melody candidate lattice exceeds transition budget")
    costs = {}
    ranks = {}
    for pitch in candidates[0]:
        if before is not None and abs(pitch-before)>plan.max_leap:
            continue
        costs[pitch] = _local_cost(pitch,original[0],targets[0],plan)
        if before is not None:
            costs[pitch] += plan.movement_weight*abs(pitch-before)
        ranks[pitch] = len(ranks)
    back = []
    reachable = len(costs)
    for index in range(1,n):
        new_costs = {}
        parents = {}
        interval = targets[index]-targets[index-1]
        for pitch in candidates[index]:
            best = None
            for previous in costs:
                step = _transition_cost(previous,pitch,interval,plan)
                if step is None:
                    continue
                candidate = (costs[previous]+step+_local_cost(
                    pitch,original[index],targets[index],plan), ranks[previous], previous)
                if best is None or candidate<best:
                    best = candidate
            if best is not None:
                new_costs[pitch] = best[0]
                parents[pitch] = best[2]
        if not new_costs:
            raise GenerationError(f"No legal melodic path at note {index}")
        order = sorted(new_costs,key=lambda pitch:(ranks[parents[pitch]],pitch))
        ranks = {pitch:rank for rank,pitch in enumerate(order)}
        costs = new_costs
        back.append(parents)
        reachable += len(costs)
    endings = [(cost+(plan.movement_weight*abs(pitch-after) if after is not None else 0),
                ranks[pitch],pitch)
               for pitch,cost in costs.items()
               if after is None or abs(pitch-after)<=plan.max_leap]
    if not endings:
        raise GenerationError("No legal melody cadence")
    best = min(endings)
    path = [best[2]]
    for parents in reversed(back):
        path.append(parents[path[-1]])
    return tuple(reversed(path)), best[0], {"transitions_considered":transitions,
                                           "reachable_states":reachable}


def vary_melody(score, plan):
    selected,before,after = _select(score,plan)
    events = [event for _,event in selected]
    original = [event.pitch for event in events]
    targets = _targets(events,plan)
    candidates = _candidates(score,events,plan,before,after)
    pitches,cost,search = optimize_pitch_path(original,targets,candidates,plan,before,after)
    changed = list(score.events)
    for (index,event),pitch in zip(selected,pitches,strict=True):
        changed[index] = replace(event,pitch=pitch)
    result = replace(score,events=tuple(sorted(changed,key=event_key)))
    report = {
        "protocol":PROTOCOL, "source_score_sha256":plan.score_sha256,
        "variation_sha256":plan.sha256,"result_score_sha256":arrangement_digest(result),
        "track":plan.track,"first_bar":plan.first_bar,"bars":plan.bars,"mode":plan.mode,
        "notes":len(events),"changed_notes":sum(a!=b for a,b in zip(original,pitches)),
        "objective_cost":cost,**search,
        "original_pitches":original,"contour_targets":targets,"result_pitches":list(pitches),
        "candidate_counts":[len(row) for row in candidates],
        "maximum_leap":max((abs(a-b) for a,b in zip(pitches,pitches[1:])),default=0),
        "maximum_shift":max(abs(a-b) for a,b in zip(original,pitches)),
        "rhythm_preserved":True,"accompaniment_preserved":True,"harmony_preserved":True,
        "learned_model":False,"external_recordings_used":False,
    }
    return result,report
