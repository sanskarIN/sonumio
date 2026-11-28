"""Deterministic performance generation with collision-free note groups.

Connected intervals on the same track move together, including chords
whose notes overlap. Adjacent groups share a fixed midpoint boundary.
Only shortening is allowed, so the original score's polyphony proof is
preserved. There is no random retry or dependence on NumPy RNG versions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

from .arrangement import Arrangement, TRACKS, arrangement_digest
from .audio import SAMPLE_RATE
from .errors import GenerationError
from .music_performance import (
    Expression, Performance, compile_performance, MAX_OFFSET, _integer, _real,
)


@dataclass(frozen=True, slots=True)
class TrackFeel:
    track: str
    timing_frames: int = 0
    shortening_frames: int = 0
    velocity_variation: float = 0.
    pan_variation: float = 0.

    def __post_init__(self):
        if type(self.track) is not str or self.track not in TRACKS:
            raise GenerationError("Unknown humanization track")
        _integer(self.timing_frames, 0, MAX_OFFSET//2, "timing variation")
        _integer(self.shortening_frames, 0, MAX_OFFSET//2, "shortening variation")
        object.__setattr__(self, "velocity_variation",
                           _real(self.velocity_variation, 0., .5, "velocity variation"))
        object.__setattr__(self, "pan_variation",
                           _real(self.pan_variation, 0., .5, "pan variation"))


@dataclass(frozen=True, slots=True)
class HumanizePlan:
    seed: int
    tracks: tuple[TrackFeel, ...]

    def __post_init__(self):
        _integer(self.seed, 0, 2**32-1, "humanization seed")
        if (type(self.tracks) is not tuple or len(self.tracks) > len(TRACKS)
                or any(not isinstance(row, TrackFeel) for row in self.tracks)):
            raise GenerationError("Invalid humanization tracks")
        names = tuple(row.track for row in self.tracks)
        if names != tuple(name for name in TRACKS if name in names):
            raise GenerationError("Humanization tracks must be unique and canonical")


def _draw(seed, track, index, domain):
    # Domain separation ensures changing duration variation does not
    # consume or perturb the timing or velocity random stream.
    h = hashlib.sha256(b"sonumio-humanize-v1\0")
    h.update(struct.pack("<III", seed, TRACKS.index(track), index))
    h.update(domain.encode("ascii"))
    return int.from_bytes(h.digest()[:8], "little")


def _bounded(seed, track, index, domain, low, high):
    if low > high:
        raise GenerationError("Empty humanization interval")
    value = _draw(seed, track, index, domain)
    return low + ((value * (high-low+1)) >> 64)


def _signed(seed, track, index, domain):
    # 53 high bits map to an exactly representable binary fraction.
    return 2. * ((_draw(seed, track, index, domain) >> 11) / (1 << 53)) - 1.


@dataclass(frozen=True, slots=True)
class _Group:
    start: int
    stop: int
    indices: tuple[int, ...]


def _groups(score, track):
    rows = [(score.plan.frame(e.tick), score.plan.frame(e.tick+e.length), i)
            for i, e in enumerate(score.events) if e.track == track]
    rows.sort()
    groups = []
    start = stop = None
    indices = []
    for left, right, index in rows:
        if stop is not None and left >= stop:
            groups.append(_Group(start, stop, tuple(indices)))
            start = stop = None
            indices = []
        if stop is None:
            start, stop = left, right
        else:
            stop = max(stop, right)
        indices.append(index)
    if indices:
        groups.append(_Group(start, stop, tuple(indices)))
    return tuple(groups)


def _boundaries(groups, total):
    if not groups:
        return ()
    middle = tuple((left.stop+right.start)//2 for left, right in zip(groups, groups[1:]))
    return (0, *middle, total)


def humanize(score, plan):
    if not isinstance(score, Arrangement) or not isinstance(plan, HumanizePlan):
        raise GenerationError("Expected score and humanization plan")
    total = score.plan.frame(score.plan.ticks)
    expressions = []
    group_count = 0
    track_reports = {}
    for feel in plan.tracks:
        groups = _groups(score, feel.track)
        limits = _boundaries(groups, total)
        actual_offsets = []
        shortening = []
        for i, group in enumerate(groups):
            low = max(-feel.timing_frames, limits[i]-group.start)
            high = min(feel.timing_frames, limits[i+1]-group.stop)
            shift = _bounded(plan.seed, feel.track, i, "timing", low, high)
            actual_offsets.append(shift)
            for index in group.indices:
                event = score.events[index]
                duration = score.plan.frame(event.tick+event.length)-score.plan.frame(event.tick)
                shorten = _bounded(plan.seed, feel.track, index, "duration", 0,
                                   min(feel.shortening_frames, duration-2))
                shortening.append(shorten)
                velocity = 1.+feel.velocity_variation*_signed(plan.seed, feel.track, index, "velocity")
                pan = feel.pan_variation*_signed(plan.seed, feel.track, index, "pan")
                expressions.append(Expression(index, shift, shift-shorten,
                                              round(velocity, 12), round(pan, 12)))
        group_count += len(groups)
        track_reports[feel.track] = {
            "groups": len(groups), "notes": len(shortening),
            "minimum_offset_frames": min(actual_offsets, default=0),
            "maximum_offset_frames": max(actual_offsets, default=0),
            "maximum_shortening_frames": max(shortening, default=0),
        }
    result = Performance(arrangement_digest(score), tuple(sorted(expressions, key=lambda row: row.index)))
    compile_performance(score, result)
    return result, {
        "protocol": "sonumio-collision-free-humanization-v1",
        "score_sha256": result.score_sha256, "performance_sha256": result.sha256,
        "seed": plan.seed, "groups": group_count, "tracks": track_reports,
        "expressions": len(result.expressions), "sample_rate": SAMPLE_RATE,
        "polyphony_verified": True, "symbolic_score_changed": False,
    }
