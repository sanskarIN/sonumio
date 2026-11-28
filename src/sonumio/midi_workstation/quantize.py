"""Deterministic score quantization, swing and controlled expressive deviation.

All transformations are explicit tick-domain operations with rational integer
rounding and digestable settings. They never change source Score objects.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from fractions import Fraction
from typing import Iterable

from .contracts import (
    MAX_TICK, MidiLimitError, MidiWorkstationError,
    boolean, integer,
)
from .score import Score, ScoreNote, ScoreTrack, note_sort_key


@dataclass(frozen=True, slots=True)
class QuantizeGrid:
    step_ticks: int
    offset_ticks: int = 0
    swing_percent: int = 0
    strength_percent: int = 100
    quantize_releases: bool = False
    min_note_ticks: int = 1

    def __post_init__(self):
        integer(self.step_ticks, "quantize grid step", 1, MAX_TICK)
        integer(self.offset_ticks, "quantize grid offset", 0, MAX_TICK)
        integer(self.swing_percent, "grid swing", 0, 75)
        integer(self.strength_percent, "quantize strength", 0, 100)
        boolean(self.quantize_releases, "quantize_releases")
        integer(self.min_note_ticks, "minimum note duration", 1, MAX_TICK)
        if self.offset_ticks >= self.step_ticks:
            raise MidiWorkstationError("Grid offset must be smaller than its step")

    def anchor(self, tick: int) -> int:
        integer(tick, "grid tick", 0, MAX_TICK)
        if tick <= self.offset_ticks:
            return self.offset_ticks
        logical = (tick - self.offset_ticks) // self.step_ticks
        low = self.tick_at_index(logical)
        high = self.tick_at_index(logical + 1)
        if tick > high:
            logical += (tick - high) // self.step_ticks
            low = self.tick_at_index(logical)
            high = self.tick_at_index(logical + 1)
        if abs(tick - low) <= abs(high - tick):
            return low
        return high

    def tick_at_index(self, index: int) -> int:
        integer(index, "grid step index", 0, MAX_TICK)
        # Swing delays every odd grid event by a percentage of half a step;
        # even events remain on the original metric subdivision.
        delay = ((self.step_ticks * self.swing_percent + 100) // 200
                 if index % 2 else 0)
        tick = self.offset_ticks + index * self.step_ticks + delay
        return integer(tick, "swing grid coordinate", 0, MAX_TICK)

    def apply_strength(self, tick: int, target: int) -> int:
        integer(tick, "quantized source tick", 0, MAX_TICK)
        integer(target, "quantized target tick", 0, MAX_TICK)
        shift = target - tick
        weighted = shift * self.strength_percent
        # Symmetric rounding avoids biasing negative corrections.
        delta = (abs(weighted) + 50) // 100
        if weighted < 0:
            delta = -delta
        return integer(tick + delta, "quantized tick", 0, MAX_TICK)


def quantize_note(note: ScoreNote, grid: QuantizeGrid,
                  *, score_end: int | None = None) -> ScoreNote:
    if not isinstance(note, ScoreNote) or not isinstance(grid, QuantizeGrid):
        raise MidiWorkstationError("Expected note and quantization grid")
    start = grid.apply_strength(note.start, grid.anchor(note.start))
    if grid.quantize_releases:
        finish = grid.apply_strength(note.stop, grid.anchor(note.stop))
        duration = max(grid.min_note_ticks, finish - start)
    else:
        duration = max(grid.min_note_ticks, note.duration)
    if start + duration > MAX_TICK:
        raise MidiLimitError("Quantized note exceeds MIDI timestamp limit")
    if score_end is not None:
        integer(score_end, "score end", 1, MAX_TICK)
        if start >= score_end:
            raise MidiWorkstationError("Quantized onset exceeds score duration")
        duration = min(duration, score_end - start)
        if duration < grid.min_note_ticks:
            raise MidiWorkstationError("Quantized note cannot fit minimum duration")
    return replace(note, start=start, duration=duration)


@dataclass(frozen=True, slots=True)
class QuantizeReport:
    note_count: int
    changed_notes: int
    changed_onsets: int
    changed_releases: int
    maximum_onset_shift: int
    maximum_release_shift: int
    original_sha256: str
    result_sha256: str
    protocol: str = "sonumio-score-quantization-v1"

    def as_dict(self):
        from dataclasses import asdict
        value = asdict(self)
        value["source_mutated"] = False
        value["model_inference"] = False
        return value


def quantize_score(score: Score, grid: QuantizeGrid,
                   *, tracks: tuple[str, ...] | None = None) -> tuple[Score, QuantizeReport]:
    if not isinstance(score, Score) or not isinstance(grid, QuantizeGrid):
        raise MidiWorkstationError("Expected Score and QuantizeGrid")
    if tracks is not None and (type(tracks) is not tuple
                               or any(type(t) is not str for t in tracks)
                               or not set(tracks) <= {t.id for t in score.tracks}):
        raise MidiWorkstationError("Unknown quantization track selection")
    eligible = {t.id for t in score.tracks} if tracks is None else set(tracks)
    changed = onset_changes = release_changes = max_onset = max_release = 0
    updated = []
    for track in score.tracks:
        if track.id not in eligible:
            updated.append(track)
            continue
        new_notes = []
        for note in track.notes:
            new = quantize_note(note, grid, score_end=score.end)
            new_notes.append(new)
            changed += int(new != note)
            onset_changes += int(new.start != note.start)
            release_changes += int(new.stop != note.stop)
            max_onset = max(max_onset, abs(new.start - note.start))
            max_release = max(max_release, abs(new.stop - note.stop))
        updated.append(track.replace_notes(tuple(new_notes)))
    candidate = score.with_tracks(tuple(updated))
    report = QuantizeReport(score.note_count, changed, onset_changes,
                            release_changes, max_onset, max_release,
                            score.sha256, candidate.sha256)
    return candidate, report


@dataclass(frozen=True, slots=True)
class HumanizeSettings:
    timing_ticks: int = 4
    velocity_units: int = 8
    duration_ticks: int = 3
    seed: int = 0
    preserve_simultaneous_chords: bool = True
    minimum_duration: int = 1

    def __post_init__(self):
        integer(self.timing_ticks, "humanize timing", 0, 4096)
        integer(self.velocity_units, "humanize velocity", 0, 64)
        integer(self.duration_ticks, "humanize duration", 0, 4096)
        integer(self.seed, "humanize random seed", 0, 2**64 - 1)
        boolean(self.preserve_simultaneous_chords, "preserve chord attacks")
        integer(self.minimum_duration, "minimum duration", 1, 4096)


def _signed_hash(seed: int, key: str, tag: str, radius: int) -> int:
    if radius == 0:
        return 0
    payload = f"{seed}:{key}:{tag}".encode("utf-8")
    digest = sha256(payload).digest()
    integer_range = 2 * radius + 1
    number = int.from_bytes(digest[:8], "big")
    return number % integer_range - radius


def humanize_score(score: Score, settings: HumanizeSettings,
                   *, tracks: tuple[str, ...] | None = None) -> tuple[Score, dict]:
    if not isinstance(score, Score) or not isinstance(settings, HumanizeSettings):
        raise MidiWorkstationError("Expected Score and HumanizeSettings")
    if tracks is not None and (type(tracks) is not tuple
                               or not set(tracks) <= {t.id for t in score.tracks}):
        raise MidiWorkstationError("Invalid humanization track list")
    selected = set(t.id for t in score.tracks) if tracks is None else set(tracks)
    modified = []
    changes = 0
    for track in score.tracks:
        if track.id not in selected:
            modified.append(track)
            continue
        notes = []
        for note in track.notes:
            # Group simultaneous notes by their onset to avoid destroying
            # human-meaningful chord attacks when explicitly requested.
            basis = (f"{track.id}:{note.start}"
                     if settings.preserve_simultaneous_chords
                     else f"{track.id}:{note.id}:{note.start}")
            timing = _signed_hash(settings.seed, basis, "time", settings.timing_ticks)
            dynamics = _signed_hash(settings.seed, f"{track.id}:{note.id}",
                                    "velocity", settings.velocity_units)
            articulation = _signed_hash(settings.seed, f"{track.id}:{note.id}",
                                        "length", settings.duration_ticks)
            start = max(0, min(score.end - 1, note.start + timing))
            duration = max(settings.minimum_duration, note.duration + articulation)
            duration = min(duration, score.end - start)
            if duration < settings.minimum_duration:
                raise MidiWorkstationError("Humanized note cannot fit score duration")
            velocity = max(1, min(127, note.velocity + dynamics))
            new = replace(note, start=start, duration=duration, velocity=velocity)
            changes += int(new != note)
            notes.append(new)
        modified.append(track.replace_notes(tuple(notes)))
    result = score.with_tracks(tuple(modified))
    return result, {
        "protocol": "sonumio-humanize-deterministic-v1",
        "seed": settings.seed, "notes": score.note_count,
        "changed": changes, "source_sha256": score.sha256,
        "result_sha256": result.sha256,
        "simultaneous_chords_preserved": settings.preserve_simultaneous_chords,
        "source_mutated": False, "model_inference": False,
    }
