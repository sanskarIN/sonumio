"""Immutable score-pinned sample-clock performance overlays.

A performance is a separate artifact. The symbolic score, MIDI events and
automation clock are not changed. Original percussion seeds remain tied to
the original event tick. The rendered sample intervals are independently
validated for bounds, ordering, instrument duration and polyphony.
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import hashlib
import json
import math

import numpy as np

from .arrangement import Arrangement, TRACKS, MAX_EVENTS, arrangement_digest
from .audio import SAMPLE_RATE, encode_wav_channels
from .errors import GenerationError
from .music_instruments import MAX_NOTE_FRAMES, render_event
from .music_mix import MAX_STEM_BYTES, MixPlan, mix_stems
from .music_automation import AutomationPlan, mix_automated_stems, _object, _unique, _constant

PROTOCOL = "sonumio-music-performance-v1"
MAX_DOCUMENT_BYTES = 512 * 1024
MAX_OFFSET = SAMPLE_RATE // 10
MAX_RENDER_NOTE_FRAMES = 12 * 60 * SAMPLE_RATE


def _integer(value, lo, hi, label):
    if type(value) is not int or not lo <= value <= hi:
        raise GenerationError("Invalid performance " + label)
    return value


def _real(value, lo, hi, label):
    if type(value) not in (int, float) or not lo <= value <= hi:
        raise GenerationError("Invalid performance " + label)
    return float(value)


def _digest(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise GenerationError("Invalid performance score SHA256")
    return value


@dataclass(frozen=True, slots=True)
class Expression:
    index: int
    onset_delta: int = 0
    end_delta: int = 0
    velocity_scale: float = 1.
    pan_delta: float = 0.

    def __post_init__(self):
        _integer(self.index, 0, MAX_EVENTS-1, "event index")
        _integer(self.onset_delta, -MAX_OFFSET, MAX_OFFSET, "onset offset")
        _integer(self.end_delta, -MAX_OFFSET, MAX_OFFSET, "end offset")
        object.__setattr__(self, "velocity_scale", _real(self.velocity_scale, .25, 2., "velocity scale"))
        object.__setattr__(self, "pan_delta", _real(self.pan_delta, -1., 1., "pan offset"))

    @property
    def identity(self):
        return (self.onset_delta == 0 and self.end_delta == 0
                and self.velocity_scale == 1. and self.pan_delta == 0.)


@dataclass(frozen=True, slots=True)
class Performance:
    score_sha256: str
    expressions: tuple[Expression, ...] = ()

    def __post_init__(self):
        _digest(self.score_sha256)
        if (type(self.expressions) is not tuple or len(self.expressions) > MAX_EVENTS
                or any(not isinstance(item, Expression) for item in self.expressions)):
            raise GenerationError("Invalid performance expression list")
        indices = tuple(item.index for item in self.expressions)
        if indices != tuple(sorted(set(indices))):
            raise GenerationError("Performance indices must strictly increase")
        # Identity rows have no semantic effect. A single canonical encoding
        # prevents them from producing different performance identifiers.
        object.__setattr__(self, "expressions", tuple(x for x in self.expressions if not x.identity))

    @property
    def sha256(self):
        return hashlib.sha256(encode_performance(self)).hexdigest()


def encode_performance(plan):
    if not isinstance(plan, Performance):
        raise GenerationError("Expected Performance")
    data = json.dumps({"format": PROTOCOL, "score_sha256": plan.score_sha256,
                       "expressions": [asdict(row) for row in plan.expressions]},
                      sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise GenerationError("Performance document exceeds byte budget")
    return data


def decode_performance(data):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_DOCUMENT_BYTES:
        raise GenerationError("Invalid performance document byte count")
    try:
        obj = _object(json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                 parse_constant=_constant),
                      ("format", "score_sha256", "expressions"))
        if obj["format"] != PROTOCOL:
            raise GenerationError("Unknown performance format")
        rows = obj["expressions"]
        if type(rows) is not list or len(rows) > MAX_EVENTS:
            raise GenerationError("Invalid performance expressions")
        expressions = tuple(Expression(**_object(row, ("index",),
                    ("onset_delta", "end_delta", "velocity_scale", "pan_delta")))
                    for row in rows)
        return Performance(obj["score_sha256"], expressions)
    except (ValueError, UnicodeError, RecursionError, TypeError, OverflowError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed performance document") from exc


def read_performance(path):
    with open(path, "rb") as stream:
        return decode_performance(stream.read(MAX_DOCUMENT_BYTES+1))


@dataclass(frozen=True, slots=True)
class PerformedNote:
    index: int
    start: int
    stop: int
    velocity: float
    pan: float


def _polyphony(score, notes):
    for track in TRACKS:
        boundaries = []
        for row in notes:
            event = score.events[row.index]
            if event.track == track:
                boundaries.append((row.start, 1, event.pitch))
                boundaries.append((row.stop, -1, event.pitch))
        active = {}
        count = 0
        for _, delta, pitch in sorted(boundaries):
            count += delta
            active[pitch] = active.get(pitch, 0) + delta
            if active[pitch] > 1 or count > (4 if track in ("keys", "pad") else 1):
                raise GenerationError("Performance exceeds sample-clock polyphony")


def compile_performance(score, performance):
    if not isinstance(score, Arrangement) or not isinstance(performance, Performance):
        raise GenerationError("Expected score and performance")
    if arrangement_digest(score) != performance.score_sha256:
        raise GenerationError("Performance score SHA256 mismatch")
    if any(row.index >= len(score.events) for row in performance.expressions):
        raise GenerationError("Performance references missing score event")
    frames = score.plan.frame(score.plan.ticks)
    if frames * len(TRACKS) * 8 > MAX_STEM_BYTES:
        raise GenerationError("Performance stem allocation exceeds budget")
    changes = {row.index: row for row in performance.expressions}
    notes = []
    for index, event in enumerate(score.events):
        row = changes.get(index)
        onset = score.plan.frame(event.tick)
        stop = score.plan.frame(event.tick+event.length)
        velocity, pan = event.velocity, event.pan
        if row is not None:
            onset += row.onset_delta
            stop += row.end_delta
            velocity = min(1., max(.001, velocity * row.velocity_scale))
            pan = min(1., max(-1., pan + row.pan_delta))
        if not 0 <= onset < stop <= frames or not 2 <= stop-onset <= MAX_NOTE_FRAMES:
            raise GenerationError("Performance note outside frame or duration bounds")
        notes.append(PerformedNote(index, onset, stop, velocity, pan))
    if sum(row.stop-row.start for row in notes) > MAX_RENDER_NOTE_FRAMES:
        raise GenerationError("Performance synthesis exceeds total note-frame budget")
    _polyphony(score, notes)
    return tuple(notes)


def describe_performance(score, performance):
    notes = compile_performance(score, performance)
    offsets = [row.start-score.plan.frame(score.events[row.index].tick) for row in notes]
    ends = [row.stop-score.plan.frame(score.events[row.index].tick+score.events[row.index].length)
            for row in notes]
    return {
        "protocol": PROTOCOL, "score_sha256": performance.score_sha256,
        "performance_sha256": performance.sha256, "events": len(notes),
        "expressions": len(performance.expressions),
        "frames": score.plan.frame(score.plan.ticks), "sample_rate": SAMPLE_RATE,
        "maximum_onset_offset": max(map(abs, offsets)),
        "maximum_end_offset": max(map(abs, ends)),
        "total_note_frames": sum(row.stop-row.start for row in notes),
        "symbolic_score_changed": False, "midi_reflects_performance": False,
        "rendered": False, "learned_model": False,
    }


def render_performance_stems(score, performance):
    notes = compile_performance(score, performance)
    frames = score.plan.frame(score.plan.ticks)
    stems = {track: np.zeros((2, frames), dtype=np.float32) for track in TRACKS}
    # Use original index order for float32 accumulation. Identity overlays
    # follow the original engine bit-for-bit, even for simultaneous voices.
    for row in notes:
        event = score.events[row.index]
        # Preserve tick/pitch for deterministic percussion noise; only
        # articulation and duration are substituted.
        voice = replace(event, velocity=row.velocity, pan=row.pan)
        mono = render_event(voice, row.stop-row.start, score.plan.seed)
        if (mono.shape != (row.stop-row.start,) or not np.isfinite(mono).all()
                or np.max(np.abs(mono)) > 1.000001):
            raise GenerationError("Invalid performance note waveform")
        pan = (row.pan+1)*math.pi/4
        stems[event.track][0, row.start:row.stop] += (mono*math.cos(pan)).astype(np.float32)
        stems[event.track][1, row.start:row.stop] += (mono*math.sin(pan)).astype(np.float32)
    for stem in stems.values():
        if not np.isfinite(stem).all():
            raise GenerationError("Invalid performance stem")
        stem.setflags(write=False)
    return stems


def generate_performance_music(score, performance, mix=None):
    report = describe_performance(score, performance)
    mix = MixPlan() if mix is None else mix
    if isinstance(mix, AutomationPlan):
        mix.validate_score(score)
    elif not isinstance(mix, MixPlan):
        raise GenerationError("Expected performance mix or automation")
    stems = render_performance_stems(score, performance)
    if isinstance(mix, AutomationPlan):
        audio, details = mix_automated_stems(score, stems, mix)
    else:
        audio, details = mix_stems(stems, mix)
    wav = encode_wav_channels(audio, SAMPLE_RATE)
    report.update({"mix": details, "rendered": True, "channels": 2,
                   "wav_sha256": hashlib.sha256(wav).hexdigest(),
                   "external_recordings_used": False})
    return wav, report
