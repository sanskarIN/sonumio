"""Conservative, immutable score-region transformations and clip assembly.

All boundaries use half-open integer MIDI ticks. When a note overlaps a region
without being fully contained, operations either explicitly trim it or fail;
none silently loses its note-on or note-off. Tempo/meter context is copied
at boundaries when cropping and concatenating score-derived clips.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .contracts import MAX_SCORE_NOTES, MAX_TICK, MidiLimitError, MidiWorkstationError, integer
from .score import Score, ScoreNote, ScoreTrack, note_sort_key
from .tempo import MeterPoint, TempoPoint


@dataclass(frozen=True, slots=True)
class TickRegion:
    start: int
    stop: int

    def __post_init__(self):
        integer(self.start, "region start tick", 0, MAX_TICK - 1)
        integer(self.stop, "region stop tick", self.start + 1, MAX_TICK)

    @property
    def ticks(self):
        return self.stop - self.start

    def contains(self, note: ScoreNote):
        return self.start <= note.start and note.stop <= self.stop

    def intersects(self, note: ScoreNote):
        return note.start < self.stop and note.stop > self.start


@dataclass(frozen=True, slots=True)
class RegionReport:
    operation: str
    input_sha256: str
    output_sha256: str
    selected: int
    created: int
    removed: int
    trimmed: int
    source_unchanged: bool = True
    protocol: str = "sonumio-workstation-region-edit-v1"

    def as_dict(self):
        return {**asdict(self), "model_inference": False,
                "remote_action_performed": False}


def _select(score, region, tracks):
    if not isinstance(score, Score) or not isinstance(region, TickRegion):
        raise MidiWorkstationError("Expected Score and half-open TickRegion")
    if region.stop > score.end:
        raise MidiWorkstationError("Region extends beyond score end")
    available = {track.id for track in score.tracks}
    if tracks is None:
        return available
    if type(tracks) is not tuple or any(type(x) is not str for x in tracks):
        raise MidiWorkstationError("Expected immutable track-ID selection")
    if len(tracks) != len(set(tracks)) or not set(tracks) <= available:
        raise MidiWorkstationError("Region selection contains unknown/duplicate tracks")
    return set(tracks)


def _overlap(note, region, policy):
    if policy == "reject" and region.intersects(note) and not region.contains(note):
        raise MidiWorkstationError("A note crosses an uneditable region boundary")


def _updated(score, tracks, op, selected, created=0, removed=0, trimmed=0):
    candidate = score.with_tracks(tuple(tracks))
    return candidate, RegionReport(op, score.sha256, candidate.sha256,
                                   selected, created, removed, trimmed)


def reverse_region(score: Score, region: TickRegion,
                   *, tracks: tuple[str, ...] | None = None,
                   overlapping: str = "reject"):
    """Reverse onset ordering, preserving pitch, length and velocity."""
    targets = _select(score, region, tracks)
    if overlapping not in ("reject", "preserve"):
        raise MidiWorkstationError("Unknown boundary-overlap policy")
    result = []
    affected = 0
    for track in score.tracks:
        if track.id not in targets:
            result.append(track)
            continue
        notes = []
        for note in track.notes:
            if overlapping == "reject":
                _overlap(note, region, overlapping)
            if region.contains(note):
                shifted = region.start + (region.stop - note.stop)
                notes.append(replace(note, start=shifted))
                affected += 1
            else:
                notes.append(note)
        result.append(track.replace_notes(tuple(notes)))
    return _updated(score, result, "reverse", affected)


def transpose_region(score: Score, region: TickRegion, semitones: int,
                     *, tracks: tuple[str, ...] | None = None,
                     overlapping: str = "reject"):
    """Transpose bounded fully contained notes without altering percussion tracks."""
    targets = _select(score, region, tracks)
    integer(semitones, "region transposition", -127, 127)
    if overlapping not in ("reject", "preserve"):
        raise MidiWorkstationError("Unknown boundary-overlap policy")
    result = []
    changed = 0
    for track in score.tracks:
        if track.id not in targets:
            result.append(track)
            continue
        notes = []
        for note in track.notes:
            if overlapping == "reject":
                _overlap(note, region, overlapping)
            if region.contains(note) and track.pitched:
                notes.append(note.transpose(semitones))
                changed += int(semitones != 0)
            else:
                notes.append(note)
        result.append(track.replace_notes(tuple(notes)))
    return _updated(score, result, "transpose", changed)


def stretch_region(score: Score, region: TickRegion,
                   numerator: int, denominator: int,
                   *, tracks: tuple[str, ...] | None = None,
                   overlapping: str = "reject"):
    """Scale *contained* note offsets/durations around region start.

    The score global clock and tempo/meter curves do not change.
    Rounding of both endpoints uses integer half-up; no drift accumulates.
    """
    targets = _select(score, region, tracks)
    integer(numerator, "stretch numerator", 1, 128)
    integer(denominator, "stretch denominator", 1, 128)
    if overlapping not in ("reject", "preserve"):
        raise MidiWorkstationError("Unknown boundary-overlap policy")
    result = []
    changed = 0
    for track in score.tracks:
        if track.id not in targets:
            result.append(track)
            continue
        notes = []
        for note in track.notes:
            if overlapping == "reject":
                _overlap(note, region, overlapping)
            if region.contains(note):
                start = region.start + ((note.start - region.start) * numerator
                                        + denominator // 2) // denominator
                finish = region.start + ((note.stop - region.start) * numerator
                                         + denominator // 2) // denominator
                if start >= score.end or finish > score.end or finish <= start:
                    raise MidiWorkstationError("Stretched note exceeds bounded score")
                next_note = replace(note, start=start, duration=finish - start)
                changed += int(next_note != note)
                notes.append(next_note)
            else:
                notes.append(note)
        result.append(track.replace_notes(tuple(notes)))
    return _updated(score, result, "stretch", changed)


def repeat_region(score: Score, region: TickRegion, copies: int,
                  *, tracks: tuple[str, ...] | None = None):
    """Duplicate a region to later positions, retaining original notes.

    Copied note IDs encode source ID and copy number. Destination must fit
    existing score timeline; the operation does not truncate other events.
    """
    targets = _select(score, region, tracks)
    integer(copies, "region duplicate count", 1, 32)
    if region.stop + copies * region.ticks > score.end:
        raise MidiWorkstationError("Repeated region would exceed score end")
    output = []
    created = selected = 0
    for track in score.tracks:
        if track.id not in targets:
            output.append(track)
            continue
        notes = list(track.notes)
        existing = {n.id for n in notes}
        originals = [n for n in track.notes if region.contains(n)]
        selected += len(originals)
        for copy in range(1, copies + 1):
            for note in originals:
                new_id = f"{note.id}-r{copy}"
                if len(new_id) > 64 or new_id in existing:
                    raise MidiWorkstationError("Repeated note identifier collides or overflows")
                existing.add(new_id)
                notes.append(replace(note, id=new_id,
                                     start=note.start + copy * region.ticks))
                created += 1
        output.append(track.replace_notes(tuple(notes)))
    return _updated(score, output, "repeat", selected, created)


def crop_score(score: Score, region: TickRegion,
               *, tracks: tuple[str, ...] | None = None,
               crossing: str = "trim"):
    """Extract a shorter independent Score with explicit tempo/meter context.

    A crop preserves the most recent tempo and meter *in force* at its
    beginning. Notes overlapping either boundary are trimmed or rejected.
    Neither raw events nor source track objects are mutated.
    """
    targets = _select(score, region, tracks)
    if crossing not in ("trim", "reject"):
        raise MidiWorkstationError("Unknown crop boundary policy")
    tracks_out = []
    count = trimmed = 0
    for track in score.tracks:
        if track.id not in targets:
            continue
        notes = []
        for note in track.notes:
            if not region.intersects(note):
                continue
            if crossing == "reject" and not region.contains(note):
                raise MidiWorkstationError("Crop boundary intersects a note")
            low = max(note.start, region.start)
            high = min(note.stop, region.stop)
            trimmed += int(high - low != note.duration)
            notes.append(replace(note, start=low - region.start,
                                 duration=high - low))
            count += 1
        tracks_out.append(track.replace_notes(tuple(notes)))
    if not tracks_out:
        raise MidiWorkstationError("A score crop needs at least one track")
    def slice_points(points):
        previous = max((p for p in points if p.tick <= region.start),
                       key=lambda p: p.tick)
        first = replace(previous, tick=0)
        later = [replace(p, tick=p.tick - region.start)
                 for p in points if region.start < p.tick < region.stop]
        return (first, *later)
    result = Score(score.title + " excerpt", score.ppqn,
                   tuple(tracks_out), slice_points(score.tempo),
                   slice_points(score.meter), region.ticks)
    return result, RegionReport("crop", score.sha256, result.sha256,
                                count, count, score.note_count - count, trimmed)


def concatenate_scores(parts: tuple[Score, ...], *, title: str | None = None):
    """Join score excerpts with identical track patches and PPQN.

    Each part has its own tempo/meter context and output note IDs are
    namespaced per part. Context changes at joins are preserved, and
    duplicate adjacent context points are collapsed safely.
    """
    if type(parts) is not tuple or not 1 <= len(parts) <= 16:
        raise MidiWorkstationError("Expected 1..16 typed score parts")
    if any(not isinstance(p, Score) for p in parts):
        raise MidiWorkstationError("Expected immutable Score parts")
    base = parts[0]
    if title is None:
        title = base.title
    if type(title) is not str:
        raise MidiWorkstationError("Invalid concatenated score title")
    for part in parts[1:]:
        if part.ppqn != base.ppqn or len(part.tracks) != len(base.tracks):
            raise MidiWorkstationError("Score parts require identical PPQN and track topology")
        for left, right in zip(base.tracks, part.tracks):
            if (left.id, left.channel, left.patches) != (right.id, right.channel, right.patches):
                raise MidiWorkstationError("Score part channel routing differs")
    end = sum(p.end for p in parts)
    if end > MAX_TICK:
        raise MidiLimitError("Concatenated score exceeds MIDI duration limit")
    if sum(p.note_count for p in parts) > MAX_SCORE_NOTES:
        raise MidiLimitError("Concatenated score exceeds note budget")
    output = []
    for index, track in enumerate(base.tracks):
        notes = []
        offset = 0
        for part_number, part in enumerate(parts):
            for note in part.tracks[index].notes:
                label = f"p{part_number}-{note.id}"
                if len(label) > 64:
                    raise MidiWorkstationError("Joined note identifier exceeds bound")
                notes.append(replace(note, id=label, start=note.start + offset))
            offset += part.end
        output.append(track.replace_notes(tuple(notes)))
    def joined_map(attribute):
        collected = []
        offset = 0
        for part in parts:
            for point in getattr(part, attribute):
                shifted = replace(point, tick=point.tick + offset)
                if collected and collected[-1].tick == shifted.tick:
                    collected[-1] = shifted
                elif collected and replace(collected[-1], tick=shifted.tick) == shifted:
                    continue
                else:
                    collected.append(shifted)
            offset += part.end
        return tuple(collected)
    result = Score(title, base.ppqn, tuple(output), joined_map("tempo"),
                   joined_map("meter"), end)
    return result, {
        "protocol": "sonumio-workstation-concatenation-v1",
        "source_digests": [part.sha256 for part in parts],
        "result_sha256": result.sha256,
        "part_count": len(parts), "notes": result.note_count,
        "source_unchanged": True, "remote_actions": False,
    }
