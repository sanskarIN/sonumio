"""Inspectable quantitative diagnostics for symbolic Sonumio compositions.

These are fixed-score statistics, not subjective music ratings. Temporal
and polyphony analyses sweep integer note boundaries with no audio FFTs.
The algorithms are bounded by the score's validated note-count contracts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from statistics import median
from typing import Iterable

from .contracts import (
    MAX_ANALYSIS_BINS, MAX_TICK, MidiLimitError, MidiWorkstationError,
    integer,
)
from .score import Score, ScoreNote, ScoreTrack
from .harmony import ChordEvidence, estimate_chords


@dataclass(frozen=True, slots=True)
class PolyphonyPoint:
    tick: int
    active_notes: int

    def __post_init__(self):
        integer(self.tick, "polyphony tick", 0, MAX_TICK)
        integer(self.active_notes, "polyphony count", 0, 100000)


@dataclass(frozen=True, slots=True)
class PolyphonyStats:
    peak_voices: int
    mean_voices: float
    occupied_fraction: float
    voice_ticks: int
    occupied_ticks: int
    changes: tuple[PolyphonyPoint, ...]

    def as_dict(self) -> dict:
        value = asdict(self)
        value["changes"] = [asdict(item) for item in self.changes]
        return value


def polyphony_timeline(notes: Iterable[ScoreNote], *,
                       end: int, include_changes: bool = False) -> PolyphonyStats:
    integer(end, "polyphony end", 1, MAX_TICK)
    if type(include_changes) is not bool:
        raise MidiWorkstationError("include_changes must be bool")
    events: list[tuple[int, int]] = []
    for note in notes:
        if not isinstance(note, ScoreNote) or note.stop > end:
            raise MidiWorkstationError("Invalid note in polyphony timeline")
        events.extend(((note.start, 1), (note.stop, -1)))
        if len(events) > 2 * 60_000:
            raise MidiLimitError("Polyphony event budget exceeded")
    if not events:
        return PolyphonyStats(0, 0., 0., 0, 0, ())
    events.sort(key=lambda item: (item[0], item[1]))
    count = peak = integrated = occupied = 0
    cursor = 0
    points = []
    index = 0
    while index < len(events):
        tick = events[index][0]
        elapsed = tick - cursor
        integrated += elapsed * count
        occupied += elapsed if count > 0 else 0
        cursor = tick
        while index < len(events) and events[index][0] == tick:
            count += events[index][1]
            index += 1
        if count < 0:
            raise MidiWorkstationError("Negative polyphony after boundary sweep")
        peak = max(peak, count)
        if include_changes:
            points.append(PolyphonyPoint(tick, count))
            if len(points) > 2 * MAX_ANALYSIS_BINS:
                raise MidiLimitError("Polyphony timeline contains too many transitions")
    integrated += (end - cursor) * count
    occupied += (end - cursor) if count > 0 else 0
    return PolyphonyStats(peak, integrated / end, occupied / end,
                          integrated, occupied, tuple(points))


@dataclass(frozen=True, slots=True)
class TrackMetrics:
    track_id: str
    name: str
    midi_channel: int
    program: int
    notes: int
    pitch_min: int | None
    pitch_max: int | None
    median_pitch: float | None
    median_velocity: float | None
    median_duration_ticks: float | None
    pitch_classes: tuple[int, ...]
    total_note_ticks: int
    peak_polyphony: int
    mean_polyphony: float
    coverage: float
    muted: bool
    solo: bool

    def as_dict(self) -> dict:
        return asdict(self)


def measure_track(track: ScoreTrack, *, score_end: int) -> TrackMetrics:
    if not isinstance(track, ScoreTrack):
        raise MidiWorkstationError("Expected ScoreTrack")
    spans = tuple(track.notes)
    voices = polyphony_timeline(spans, end=score_end)
    return TrackMetrics(
        track.id, track.name, track.channel, track.program,
        len(spans), min((n.pitch for n in spans), default=None),
        max((n.pitch for n in spans), default=None),
        float(median(n.pitch for n in spans)) if spans else None,
        float(median(n.velocity for n in spans)) if spans else None,
        float(median(n.duration for n in spans)) if spans else None,
        tuple(sum(n.pitch % 12 == index for n in spans) for index in range(12)),
        sum(n.duration for n in spans), voices.peak_voices,
        voices.mean_voices, voices.occupied_fraction,
        track.muted, track.solo,
    )


@dataclass(frozen=True, slots=True)
class QuantizationStats:
    step_ticks: int
    note_count: int
    aligned_onsets: int
    aligned_releases: int
    mean_absolute_onset_error: float
    median_absolute_onset_error: float
    max_absolute_onset_error: int


def quantization_statistics(notes: tuple[ScoreNote, ...], *,
                            step_ticks: int, offset_ticks: int = 0) -> QuantizationStats:
    integer(step_ticks, "analysis grid step", 1, MAX_TICK)
    integer(offset_ticks, "analysis grid offset", 0, step_ticks - 1)
    if type(notes) is not tuple or any(not isinstance(n, ScoreNote) for n in notes):
        raise MidiWorkstationError("Expected typed symbolic note tuple")

    def error(tick):
        distance = (tick - offset_ticks) % step_ticks
        return min(distance, step_ticks - distance)

    onset = [error(note.start) for note in notes]
    release = [error(note.stop) for note in notes]
    return QuantizationStats(
        step_ticks, len(notes), sum(x == 0 for x in onset),
        sum(x == 0 for x in release),
        sum(onset) / len(onset) if onset else 0.,
        float(median(onset)) if onset else 0.,
        max(onset, default=0),
    )


@dataclass(frozen=True, slots=True)
class TimeWindow:
    start: int
    stop: int
    notes_started: int
    notes_active: int
    unique_pitch_classes: int
    peak_polyphony: int
    median_velocity: float | None
    best_chord: str | None
    chord_score: float | None

    def __post_init__(self):
        integer(self.start, "window start", 0, MAX_TICK)
        integer(self.stop, "window stop", self.start + 1, MAX_TICK)
        integer(self.notes_started, "started notes", 0)
        integer(self.notes_active, "active notes", 0)
        integer(self.unique_pitch_classes, "pitch classes", 0, 12)
        integer(self.peak_polyphony, "peak polyphony", 0)


def analyze_windows(score: Score, *,
                    window_ticks: int | None = None,
                    track_ids: tuple[str, ...] | None = None,
                    include_muted: bool = False) -> tuple[TimeWindow, ...]:
    if not isinstance(score, Score):
        raise MidiWorkstationError("Expected Score")
    if window_ticks is None:
        window_ticks = score.ppqn * 4
    integer(window_ticks, "analysis window ticks", 1, MAX_TICK)
    if type(include_muted) is not bool:
        raise MidiWorkstationError("include_muted flag must be bool")
    if track_ids is not None and (
        type(track_ids) is not tuple or not set(track_ids) <= {t.id for t in score.tracks}
    ):
        raise MidiWorkstationError("Invalid analysis track selection")
    windows = (score.end + window_ticks - 1) // window_ticks
    if windows > MAX_ANALYSIS_BINS:
        raise MidiLimitError("Analysis would produce too many timeline windows")
    eligible = set(score.active_track_ids) if not include_muted else {t.id for t in score.tracks}
    if track_ids is not None:
        eligible &= set(track_ids)
    notes = tuple(n for t in score.tracks if t.id in eligible for n in t.notes)
    result = []
    for index in range(windows):
        start = index * window_ticks
        stop = min(score.end, start + window_ticks)
        active = [n for n in notes if n.start < stop and n.stop > start]
        started = [n for n in notes if start <= n.start < stop]
        # Weight a pitch by how long it actually sounds inside the window.
        weighted = tuple((n.pitch, float(min(stop, n.stop) - max(start, n.start)))
                         for n in active)
        hypotheses = estimate_chords(weighted, candidates=1)
        best = hypotheses[0] if hypotheses else None
        events = []
        for note in active:
            events.append((max(start, note.start), 1))
            events.append((min(stop, note.stop), -1))
        events.sort(key=lambda e: (e[0], e[1]))
        current = peak = 0
        for _, direction in events:
            current += direction
            peak = max(peak, current)
        result.append(TimeWindow(
            start, stop, len(started), len(active),
            len({n.pitch % 12 for n in active}), peak,
            float(median(n.velocity for n in started)) if started else None,
            best.chord.symbol if best else None,
            best.score if best else None,
        ))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ScoreMetrics:
    source_sha256: str
    note_count: int
    active_note_count: int
    tracks: tuple[TrackMetrics, ...]
    global_peak_polyphony: int
    global_mean_polyphony: float
    duration_microseconds: int
    pitch_classes: tuple[int, ...]
    overlapping_same_pitch_pairs: int
    empty_tracks: int
    sound_coverage_fraction: float
    protocol: str = "sonumio-symbolic-music-metrics-v1"

    def as_dict(self) -> dict:
        data = asdict(self)
        data["trained_models_used"] = False
        data["musical_quality_established"] = False
        return data


def analyze_score(score: Score, *, include_muted: bool = False) -> ScoreMetrics:
    if not isinstance(score, Score) or type(include_muted) is not bool:
        raise MidiWorkstationError("Expected Score and include_muted flag")
    selected = (score.tracks if include_muted else
                tuple(t for t in score.tracks if t.id in score.active_track_ids))
    notes = tuple(n for track in selected for n in track.notes)
    voices = polyphony_timeline(notes, end=score.end)
    same_pitch: dict[tuple[int, int], list[ScoreNote]] = defaultdict(list)
    for track in selected:
        for note in track.notes:
            same_pitch[(track.channel, note.pitch)].append(note)
    overlap = 0
    for values in same_pitch.values():
        ordered = sorted(values, key=lambda n: (n.start, n.stop))
        ends: list[int] = []
        for note in ordered:
            ends = [end for end in ends if end > note.start]
            overlap += len(ends)
            ends.append(note.stop)
    return ScoreMetrics(
        score.sha256, score.note_count, len(notes),
        tuple(measure_track(track, score_end=score.end) for track in score.tracks),
        voices.peak_voices, voices.mean_voices,
        int(score.duration_microseconds),
        tuple(sum(n.pitch % 12 == i for n in notes) for i in range(12)),
        overlap, sum(not t.notes for t in score.tracks),
        voices.occupied_fraction,
    )


def note_density(score: Score, *, bins: int = 64,
                 include_muted: bool = False) -> tuple[int, ...]:
    if not isinstance(score, Score):
        raise MidiWorkstationError("Expected Score")
    integer(bins, "density bins", 1, MAX_ANALYSIS_BINS)
    if type(include_muted) is not bool:
        raise MidiWorkstationError("include_muted must be bool")
    active = set(score.active_track_ids) if not include_muted else {t.id for t in score.tracks}
    results = [0] * bins
    for track in score.tracks:
        if track.id not in active:
            continue
        for note in track.notes:
            index = min(bins - 1, (note.start * bins) // score.end)
            results[index] += 1
    return tuple(results)


def score_inspection(score: Score, *, window_ticks: int | None = None,
                     include_muted: bool = False) -> dict:
    metrics = analyze_score(score, include_muted=include_muted)
    windows = analyze_windows(score, window_ticks=window_ticks,
                              include_muted=include_muted)
    return {
        "protocol": "sonumio-symbolic-inspection-v1",
        "score": score.description(),
        "metrics": metrics.as_dict(),
        "windows": [asdict(window) for window in windows],
        "number_of_windows": len(windows),
        "model_inference": False,
        "subjective_quality_scored": False,
    }
