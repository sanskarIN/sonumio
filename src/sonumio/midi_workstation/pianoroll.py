"""Backend-neutral piano-roll viewport, selection and drag-edit contracts.

All coordinates are absolute symbolic ticks and integer MIDI pitches. No
pixel resolution, CSS layout, DOM or pointer device is presumed. The same
API can serve a desktop/web editor and a keyboard-first accessibility layer.
Every gesture is atomic and returns a fresh validated Score; the source is
never modified and stale client revisions are rejected by SHA-256.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from .contracts import (
    MAX_SCORE_NOTES, MAX_TICK, MidiIntegrityError, MidiLimitError,
    MidiWorkstationError, integer, sha256,
)
from .score import Score, ScoreNote

MAX_VISIBLE_NOTES = 4096
MAX_CLIPBOARD_NOTES = 1024
MAX_GESTURE_NOTES = 4096


@dataclass(frozen=True, slots=True)
class RollViewport:
    first_tick: int
    last_tick: int
    lowest_pitch: int = 36
    highest_pitch: int = 96
    maximum_visible: int = MAX_VISIBLE_NOTES
    track_ids: tuple[str, ...] | None = None
    include_muted: bool = True

    def __post_init__(self):
        integer(self.first_tick, "viewport start", 0, MAX_TICK - 1)
        integer(self.last_tick, "viewport end", self.first_tick + 1, MAX_TICK)
        integer(self.lowest_pitch, "viewport low pitch", 0, 127)
        integer(self.highest_pitch, "viewport high pitch", self.lowest_pitch, 127)
        integer(self.maximum_visible, "visible note limit", 1, MAX_VISIBLE_NOTES)
        if type(self.include_muted) is not bool:
            raise MidiWorkstationError("include_muted must be boolean")
        if self.track_ids is not None:
            if (type(self.track_ids) is not tuple
                    or any(type(t) is not str for t in self.track_ids)
                    or len(self.track_ids) != len(set(self.track_ids))):
                raise MidiWorkstationError("Viewport track IDs must be unique strings")


@dataclass(frozen=True, slots=True)
class RollNoteBox:
    track_id: str
    note_id: str
    pitch: int
    start: int
    stop: int
    clipped_start: int
    clipped_stop: int
    velocity: int
    pan: int
    selected: bool = False

    @property
    def hidden_left(self):
        return self.start < self.clipped_start

    @property
    def hidden_right(self):
        return self.stop > self.clipped_stop

    @property
    def visible_ticks(self):
        return self.clipped_stop - self.clipped_start

    def as_dict(self):
        return asdict(self) | {
            "hidden_left": self.hidden_left,
            "hidden_right": self.hidden_right,
        }


@dataclass(frozen=True, slots=True)
class RollView:
    viewport: RollViewport
    source_sha256: str
    score_revision: int
    boxes: tuple[RollNoteBox, ...]
    source_note_count: int
    visible_note_count: int
    tracks_visible: tuple[str, ...]
    protocol: str = "sonumio-workstation-piano-roll-v1"

    def as_dict(self):
        return {
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
            "score_revision": self.score_revision,
            "bounds": asdict(self.viewport),
            "visible_notes": self.visible_note_count,
            "score_notes": self.source_note_count,
            "tracks": list(self.tracks_visible),
            "boxes": [box.as_dict() for box in self.boxes],
            "source_mutated": False,
            "no_pixel_coordinates_inferred": True,
        }


def _selected(selection):
    if selection is None:
        return set()
    if (type(selection) is not tuple or len(selection) > MAX_GESTURE_NOTES
            or any(type(pair) is not tuple or len(pair) != 2
                   or any(type(x) is not str for x in pair) for pair in selection)
            or len(set(selection)) != len(selection)):
        raise MidiWorkstationError("Selection must be bounded stable track/note pairs")
    return set(selection)


def view_piano_roll(score: Score, viewport: RollViewport,
                    *, selection: tuple[tuple[str, str], ...] | None = None) -> RollView:
    if not isinstance(score, Score) or not isinstance(viewport, RollViewport):
        raise MidiWorkstationError("Expected typed Score and RollViewport")
    if viewport.last_tick > score.end:
        raise MidiWorkstationError("Viewport cannot exceed declared score duration")
    selected = _selected(selection)
    available = {track.id for track in score.tracks}
    if viewport.track_ids is not None and not set(viewport.track_ids) <= available:
        raise MidiWorkstationError("Unknown piano-roll track filter")
    if selected and any(track_id not in available for track_id, _ in selected):
        raise MidiWorkstationError("Selection refers to unknown score track")
    permitted = available if viewport.track_ids is None else set(viewport.track_ids)
    active = set(score.active_track_ids)
    result = []
    tracks = []
    for track in score.tracks:
        if track.id not in permitted or (
            not viewport.include_muted and track.id not in active
        ):
            continue
        tracks.append(track.id)
        for note in track.notes:
            if (note.start >= viewport.last_tick or note.stop <= viewport.first_tick
                    or not viewport.lowest_pitch <= note.pitch <= viewport.highest_pitch):
                continue
            result.append(RollNoteBox(
                track.id, note.id, note.pitch, note.start, note.stop,
                max(note.start, viewport.first_tick),
                min(note.stop, viewport.last_tick), note.velocity, note.pan,
                (track.id, note.id) in selected,
            ))
            if len(result) > viewport.maximum_visible:
                raise MidiLimitError("Piano-roll viewport exceeded visible note limit")
    result.sort(key=lambda box: (box.start, box.pitch, box.track_id, box.note_id))
    return RollView(viewport, score.sha256, score.revision, tuple(result),
                    score.note_count, len(result), tuple(tracks))


def rectangular_selection(view: RollView, *, first_tick: int, last_tick: int,
                          lowest_pitch: int, highest_pitch: int,
                          mode: str = "intersect") -> tuple[tuple[str, str], ...]:
    if not isinstance(view, RollView):
        raise MidiWorkstationError("Expected piano-roll view")
    integer(first_tick, "selection start", 0, MAX_TICK - 1)
    integer(last_tick, "selection stop", first_tick + 1, MAX_TICK)
    integer(lowest_pitch, "selection low pitch", 0, 127)
    integer(highest_pitch, "selection high pitch", lowest_pitch, 127)
    if mode not in ("intersect", "contained", "onset"):
        raise MidiWorkstationError("Unsupported rectangular selection mode")
    selected = []
    for box in view.boxes:
        if not lowest_pitch <= box.pitch <= highest_pitch:
            continue
        if mode == "intersect":
            accepted = box.start < last_tick and box.stop > first_tick
        elif mode == "contained":
            accepted = first_tick <= box.start and box.stop <= last_tick
        else:
            accepted = first_tick <= box.start < last_tick
        if accepted:
            selected.append((box.track_id, box.note_id))
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class RollGesture:
    base_sha256: str
    selection: tuple[tuple[str, str], ...]
    shift_ticks: int = 0
    transpose_semitones: int = 0
    resize_left_ticks: int = 0
    resize_right_ticks: int = 0
    velocity_delta: int = 0
    clamp_velocity: bool = False

    def __post_init__(self):
        sha256(self.base_sha256)
        selected = _selected(self.selection)
        if not selected:
            raise MidiWorkstationError("Piano-roll gesture needs at least one note")
        integer(self.shift_ticks, "gesture timeline shift", -MAX_TICK, MAX_TICK)
        integer(self.transpose_semitones, "gesture pitch shift", -127, 127)
        integer(self.resize_left_ticks, "gesture left edge", -MAX_TICK, MAX_TICK)
        integer(self.resize_right_ticks, "gesture right edge", -MAX_TICK, MAX_TICK)
        integer(self.velocity_delta, "gesture velocity delta", -127, 127)
        if type(self.clamp_velocity) is not bool:
            raise MidiWorkstationError("clamp_velocity must be bool")


@dataclass(frozen=True, slots=True)
class RollGestureReport:
    base_sha256: str
    result_sha256: str
    source_revision: int
    result_revision: int
    changed_notes: int
    protocol: str = "sonumio-workstation-piano-roll-edit-v1"

    def as_dict(self):
        return {**asdict(self), "source_mutated": False,
                "no_remote_writes": True, "model_inference": False}


def apply_roll_gesture(score: Score, gesture: RollGesture):
    if not isinstance(score, Score) or not isinstance(gesture, RollGesture):
        raise MidiWorkstationError("Expected Score and RollGesture")
    if gesture.base_sha256 != score.sha256:
        raise MidiIntegrityError("Piano-roll gesture uses stale score digest")
    selected = _selected(gesture.selection)
    changed = set()
    output = []
    for track in score.tracks:
        modified = []
        for note in track.notes:
            key = (track.id, note.id)
            if key not in selected:
                modified.append(note)
                continue
            left = note.start + gesture.shift_ticks + gesture.resize_left_ticks
            stop = note.stop + gesture.shift_ticks + gesture.resize_right_ticks
            if left < 0 or left >= score.end or stop > score.end or stop <= left:
                raise MidiWorkstationError("Piano-roll gesture exceeds score timeline")
            pitch = note.pitch + gesture.transpose_semitones
            integer(pitch, "gesture note pitch", 0, 127)
            velocity = note.velocity + gesture.velocity_delta
            if gesture.clamp_velocity:
                velocity = max(1, min(127, velocity))
            integer(velocity, "gesture note velocity", 1, 127)
            updated = replace(note, start=left, duration=stop-left,
                              pitch=pitch, velocity=velocity)
            changed.add(key)
            modified.append(updated)
        output.append(track.replace_notes(tuple(modified)))
    if changed != selected:
        raise MidiWorkstationError("Piano-roll selection contains unknown note IDs")
    result = score.with_tracks(tuple(output))
    return result, RollGestureReport(score.sha256, result.sha256,
                                     score.revision, result.revision, len(changed))


@dataclass(frozen=True, slots=True)
class NoteClipboard:
    """Detached note content with relative tick/pitch location and stable IDs."""

    notes: tuple[tuple[str, ScoreNote], ...]
    source_sha256: str
    origin_tick: int
    origin_pitch: int

    def __post_init__(self):
        if (type(self.notes) is not tuple or not 1 <= len(self.notes) <= MAX_CLIPBOARD_NOTES
                or any(type(pair) is not tuple or len(pair) != 2
                       or type(pair[0]) is not str or not isinstance(pair[1], ScoreNote)
                       for pair in self.notes)):
            raise MidiWorkstationError("Clipboard needs bounded typed note pairs")
        sha256(self.source_sha256)
        integer(self.origin_tick, "clipboard origin tick", 0, MAX_TICK)
        integer(self.origin_pitch, "clipboard origin pitch", 0, 127)


def copy_roll_notes(score: Score,
                    selection: tuple[tuple[str, str], ...]) -> NoteClipboard:
    if not isinstance(score, Score):
        raise MidiWorkstationError("Expected Score to copy")
    selected = _selected(selection)
    if not selected or len(selected) > MAX_CLIPBOARD_NOTES:
        raise MidiLimitError("Clipboard needs 1..1024 selected notes")
    copied = [(track_id, note) for track in score.tracks
              for note in track.notes
              for track_id in (track.id,) if (track.id, note.id) in selected]
    if len(copied) != len(selected):
        raise MidiWorkstationError("Cannot copy nonexistent note IDs")
    copied.sort(key=lambda row: (row[1].start, row[1].pitch, row[0], row[1].id))
    return NoteClipboard(tuple(copied), score.sha256,
                         min(note.start for _, note in copied),
                         min(note.pitch for _, note in copied))


def paste_roll_notes(score: Score, clipboard: NoteClipboard, *,
                     destination_tick: int, destination_pitch: int,
                     suffix: str = "paste"):
    if not isinstance(score, Score) or not isinstance(clipboard, NoteClipboard):
        raise MidiWorkstationError("Expected Score and NoteClipboard")
    integer(destination_tick, "paste tick", 0, score.end - 1)
    integer(destination_pitch, "paste pitch", 0, 127)
    if (type(suffix) is not str or not 1 <= len(suffix) <= 16
            or not suffix.replace("-", "").replace("_", "").isalnum()):
        raise MidiWorkstationError("Invalid paste note-ID suffix")
    groups: dict[str, list[ScoreNote]] = {
        track.id: list(track.notes) for track in score.tracks
    }
    existing = {track.id: {note.id for note in track.notes}
                for track in score.tracks}
    created = 0
    for track_id, note in clipboard.notes:
        if track_id not in groups:
            raise MidiWorkstationError("Clipboard references unknown destination track")
        first = destination_tick + note.start - clipboard.origin_tick
        pitch = destination_pitch + note.pitch - clipboard.origin_pitch
        if not 0 <= first < score.end or first + note.duration > score.end:
            raise MidiWorkstationError("Pasted note extends beyond score timeline")
        integer(pitch, "pasted pitch", 0, 127)
        label = f"{note.id}-{suffix}"
        if len(label) > 64 or label in existing[track_id]:
            raise MidiWorkstationError("Pasted note ID collides or exceeds length bound")
        existing[track_id].add(label)
        groups[track_id].append(replace(note, id=label, start=first, pitch=pitch))
        created += 1
    if score.note_count + created > MAX_SCORE_NOTES:
        raise MidiLimitError("Pasting exceeds score event limit")
    result = score.with_tracks(tuple(track.replace_notes(tuple(groups[track.id]))
                                     for track in score.tracks))
    return result, {
        "protocol": "sonumio-workstation-piano-roll-paste-v1",
        "source_sha256": score.sha256,
        "clipboard_sha256": clipboard.source_sha256,
        "result_sha256": result.sha256,
        "created": created,
        "source_mutated": False,
        "model_inference": False,
    }
