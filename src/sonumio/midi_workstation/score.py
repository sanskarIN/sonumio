"""Immutable, bounded multi-track symbolic composition for Sonumio.

A score keeps note events, explicit tempo/meter changes and program/bank
routing. It never contains executable code, external sample paths, trained
weights, or binary media. Its intended use is editing original symbolic
compositions without changing the existing strict Arrangement schema.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import re

from .contracts import (
    MAX_LABEL, MAX_NOTES, MAX_PPQ, MAX_SCORE_NOTES, MAX_SCORE_TRACKS,
    MAX_TICK, MidiFormatError, MidiLimitError, MidiWorkstationError,
    canonical_json, finite, integer, text_value,
)
from .model import (
    META_EOT, META_TRACK_NAME, MidiEvent, MidiFile, MidiTrack, TimeDivision,
)
from .performance import (
    PerformanceSettings, reconstruct_performance,
)
from .tempo import MeterMap, MeterPoint, TempoMap, TempoPoint

NOTE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")
MAX_SCORE_DURATION = MAX_TICK
SCORE_PROTOCOL = "sonumio-symbolic-workstation-score-v1"


@dataclass(frozen=True, slots=True)
class ScoreNote:
    id: str
    start: int
    duration: int
    pitch: int
    velocity: int = 100
    release_velocity: int = 0
    pan: int = 64

    def __post_init__(self):
        if type(self.id) is not str or not NOTE_ID_RE.fullmatch(self.id):
            raise MidiWorkstationError("Score note requires ASCII stable identifier")
        integer(self.start, "score note start", 0, MAX_TICK)
        integer(self.duration, "score note duration", 1, MAX_TICK)
        if self.start + self.duration > MAX_TICK:
            raise MidiWorkstationError("Score note extends past MIDI tick bound")
        integer(self.pitch, "note pitch", 0, 127)
        integer(self.velocity, "note velocity", 1, 127)
        integer(self.release_velocity, "note release velocity", 0, 127)
        integer(self.pan, "note pan", 0, 127)

    @property
    def stop(self) -> int:
        return self.start + self.duration

    def transpose(self, semitones: int) -> "ScoreNote":
        integer(semitones, "transposition", -127, 127)
        return replace(self, pitch=integer(self.pitch + semitones,
                                          "transposed note pitch", 0, 127))

    def shift(self, ticks: int) -> "ScoreNote":
        integer(ticks, "note shift", -MAX_TICK, MAX_TICK)
        return replace(self, start=integer(self.start + ticks, "shifted note", 0, MAX_TICK))

    def with_velocity(self, value: int) -> "ScoreNote":
        return replace(self, velocity=integer(value, "note velocity", 1, 127))

    def resized(self, duration: int) -> "ScoreNote":
        return replace(self, duration=integer(duration, "note duration", 1, MAX_TICK))


def note_sort_key(note: ScoreNote) -> tuple:
    return (note.start, note.pitch, note.duration, note.id)


@dataclass(frozen=True, slots=True)
class ScoreTrack:
    id: str
    name: str
    channel: int
    program: int
    bank_msb: int
    bank_lsb: int
    notes: tuple[ScoreNote, ...]
    muted: bool = False
    solo: bool = False

    def __post_init__(self):
        if type(self.id) is not str or not NOTE_ID_RE.fullmatch(self.id):
            raise MidiWorkstationError("Score track ID must be stable ASCII")
        text_value(self.name, "track name", MAX_LABEL)
        integer(self.channel, "track MIDI channel", 0, 15)
        integer(self.program, "track MIDI program", 0, 127)
        integer(self.bank_msb, "bank MSB", 0, 127)
        integer(self.bank_lsb, "bank LSB", 0, 127)
        if type(self.notes) is not tuple or len(self.notes) > MAX_SCORE_NOTES:
            raise MidiWorkstationError("Invalid score track notes")
        if any(not isinstance(note, ScoreNote) for note in self.notes):
            raise MidiWorkstationError("Expected ScoreNote objects")
        if list(self.notes) != sorted(self.notes, key=note_sort_key):
            raise MidiWorkstationError("Score notes must be in canonical sort order")
        note_ids = [note.id for note in self.notes]
        if len(note_ids) != len(set(note_ids)):
            raise MidiWorkstationError("Duplicate note identifier in score track")
        if type(self.muted) is not bool or type(self.solo) is not bool:
            raise MidiWorkstationError("Score mute/solo switches must be bool")

    @property
    def end(self) -> int:
        return max((note.stop for note in self.notes), default=0)

    @property
    def pitched(self) -> bool:
        return self.channel != 9

    @property
    def patches(self) -> tuple[int, int, int]:
        return self.bank_msb, self.bank_lsb, self.program

    def replace_notes(self, notes: tuple[ScoreNote, ...]) -> "ScoreTrack":
        if type(notes) is not tuple:
            raise MidiWorkstationError("Expected tuple of score notes")
        return replace(self, notes=tuple(sorted(notes, key=note_sort_key)))

    def active_notes(self, *, include_muted: bool = False) -> tuple[ScoreNote, ...]:
        if type(include_muted) is not bool:
            raise MidiWorkstationError("include_muted flag must be bool")
        return self.notes if include_muted or not self.muted else ()


@dataclass(frozen=True, slots=True)
class Score:
    title: str
    ppqn: int
    tracks: tuple[ScoreTrack, ...]
    tempo: tuple[TempoPoint, ...]
    meter: tuple[MeterPoint, ...]
    end: int
    revision: int = 0

    def __post_init__(self):
        text_value(self.title, "score title", MAX_LABEL)
        integer(self.ppqn, "score PPQN", 1, MAX_PPQ)
        if (type(self.tracks) is not tuple
                or not 1 <= len(self.tracks) <= MAX_SCORE_TRACKS
                or any(not isinstance(t, ScoreTrack) for t in self.tracks)):
            raise MidiWorkstationError("Score needs 1..48 typed tracks")
        track_ids = [track.id for track in self.tracks]
        if len(track_ids) != len(set(track_ids)):
            raise MidiWorkstationError("Score track IDs must be unique")
        if sum(len(track.notes) for track in self.tracks) > MAX_SCORE_NOTES:
            raise MidiLimitError("Score note count exceeds budget")
        if type(self.tempo) is not tuple or not self.tempo:
            raise MidiWorkstationError("Score requires tempo points")
        if type(self.meter) is not tuple or not self.meter:
            raise MidiWorkstationError("Score requires meter points")
        TempoMap(TimeDivision(self.ppqn), self.tempo)
        MeterMap(TimeDivision(self.ppqn), self.meter)
        integer(self.end, "score end tick", 1, MAX_TICK)
        integer(self.revision, "score revision", 0, 2**31 - 1)
        if any(track.end > self.end for track in self.tracks):
            raise MidiWorkstationError("Score note exceeds global end time")
        if any(point.tick > self.end for point in (*self.tempo, *self.meter)):
            raise MidiWorkstationError("Tempo or meter change outside score timeline")
        # Scores may use one MIDI channel across several layers only when their
        # programs and banks agree. Otherwise live MIDI playback would share
        # conflicting channel state.
        route: dict[int, tuple[int, int, int]] = {}
        for track in self.tracks:
            prior = route.setdefault(track.channel, track.patches)
            if prior != track.patches:
                raise MidiWorkstationError("Shared MIDI channel has conflicting patch setup")

    @classmethod
    def empty(cls, *, ppqn: int = 480, end: int = 1920,
              title: str = "Original composition") -> "Score":
        track = ScoreTrack("track-1", "Instrument 1", 0, 0, 0, 0, ())
        return cls(title, ppqn, (track,), (TempoPoint(0, 500000),),
                   (MeterPoint(0, 4, 4),), end)

    @property
    def division(self) -> TimeDivision:
        return TimeDivision(self.ppqn)

    @property
    def tempo_map(self) -> TempoMap:
        return TempoMap(self.division, self.tempo)

    @property
    def meter_map(self) -> MeterMap:
        return MeterMap(self.division, self.meter)

    @property
    def notes(self) -> tuple[ScoreNote, ...]:
        return tuple(note for track in self.tracks for note in track.notes)

    @property
    def note_count(self) -> int:
        return sum(len(track.notes) for track in self.tracks)

    @property
    def active_track_ids(self) -> tuple[str, ...]:
        soloed = [track.id for track in self.tracks if track.solo and not track.muted]
        if soloed:
            return tuple(soloed)
        return tuple(track.id for track in self.tracks if not track.muted)

    @property
    def active_notes(self) -> tuple[tuple[str, ScoreNote], ...]:
        active = set(self.active_track_ids)
        return tuple((track.id, note) for track in self.tracks if track.id in active
                     for note in track.notes)

    @property
    def duration_microseconds(self) -> Fraction:
        return self.tempo_map.tick_to_microseconds(self.end)

    def track_by_id(self, name: str) -> ScoreTrack:
        for track in self.tracks:
            if track.id == name:
                return track
        raise MidiWorkstationError("No score track with requested identifier")

    def note_by_id(self, track_id: str, note_id: str) -> ScoreNote:
        for note in self.track_by_id(track_id).notes:
            if note.id == note_id:
                return note
        raise MidiWorkstationError("No note with requested identifier")

    def with_tracks(self, tracks: tuple[ScoreTrack, ...]) -> "Score":
        return replace(self, tracks=tracks, revision=self.revision + 1)

    def with_tempo(self, tempo: tuple[TempoPoint, ...]) -> "Score":
        return replace(self, tempo=tempo, revision=self.revision + 1)

    def with_meter(self, meter: tuple[MeterPoint, ...]) -> "Score":
        return replace(self, meter=meter, revision=self.revision + 1)

    def with_end(self, end: int) -> "Score":
        return replace(self, end=end, revision=self.revision + 1)

    def to_json_obj(self) -> dict:
        return {
            "protocol": SCORE_PROTOCOL,
            "title": self.title,
            "ppqn": self.ppqn,
            "end": self.end,
            "revision": self.revision,
            "tempo": [asdict(point) for point in self.tempo],
            "meter": [asdict(point) for point in self.meter],
            "tracks": [asdict(track) for track in self.tracks],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_json_obj())).hexdigest()

    def description(self) -> dict:
        return {
            "protocol": "sonumio-workstation-score-summary-v1",
            "title": self.title, "ppqn": self.ppqn,
            "tempo_points": len(self.tempo), "meter_points": len(self.meter),
            "tracks": len(self.tracks), "active_tracks": len(self.active_track_ids),
            "notes": self.note_count, "end_tick": self.end,
            "duration_seconds": float(self.duration_microseconds / 1_000_000),
            "sha256": self.sha256, "revision": self.revision,
            "trained_models_used": False,
        }


def _stable_note_id(track: int, ordinal: int, note: object) -> str:
    return f"m{track:02d}-{ordinal:05d}"


def from_midi(
    midi: MidiFile, *, title: str = "Imported MIDI",
    settings: PerformanceSettings = PerformanceSettings(),
    audible_duration: bool = False,
    allow_patch_changes: bool = False,
) -> tuple[Score, dict]:
    """Import note-bearing MIDI into explicit score tracks, with loss ledger.

    Program changes are handled by grouping notes by the active bank/patch.
    SMPTE timing is not guessed into a PPQN score. Format-2 tracks have
    independent tempo clocks, so this conversion rejects them.
    """
    if not isinstance(midi, MidiFile) or midi.division.smpte or midi.format == 2:
        raise MidiWorkstationError("Score conversion requires synchronous PPQN MIDI")
    if type(audible_duration) is not bool or type(allow_patch_changes) is not bool:
        raise MidiWorkstationError("Import policy flags must be bool")
    performance = reconstruct_performance(midi, settings)
    tempo_map = TempoMap.from_midi(midi)
    meter_map = MeterMap.from_midi(midi)
    grouping: dict[tuple[int, int, int, int, int], list] = {}
    for span in performance.notes:
        key = (span.track, span.channel, span.bank_msb, span.bank_lsb, span.program)
        grouping.setdefault(key, []).append(span)
    route_patches: dict[int, set[tuple[int, int, int]]] = {}
    for (_, channel, msb, lsb, patch) in grouping:
        route_patches.setdefault(channel, set()).add((msb, lsb, patch))
    changing_channels = {
        channel: patches for channel, patches in route_patches.items()
        if len(patches) > 1
    }
    if changing_channels and not allow_patch_changes:
        raise MidiWorkstationError(
            "MIDI channel changes patches; explicitly enable channel "
            "reassignment to import this as fixed-patch score tracks"
        )
    # A fixed-patch score cannot reproduce multiple patches on one shared
    # MIDI channel. The opt-in policy allocates each *additional* patch
    # a free channel, retaining the initial patch on its original channel.
    # Do not steal channels used by unrelated source material or route a
    # pitched instrument onto General MIDI percussion channel 9.
    reassigned: dict[tuple[int, tuple[int, int, int]], int] = {}
    if changing_channels:
        free_channels = [ch for ch in range(16)
                         if ch != 9 and ch not in route_patches]
        for channel in sorted(changing_channels):
            if channel == 9:
                raise MidiWorkstationError(
                    "Changing drum kits on MIDI channel 9 cannot be rechanneled "
                    "without altering percussion meaning"
                )
            first_seen: dict[tuple[int, int, int], tuple[int, int]] = {}
            for span in performance.notes:
                if span.channel == channel:
                    patch = (span.bank_msb, span.bank_lsb, span.program)
                    location = (span.start, span.track)
                    if patch not in first_seen or location < first_seen[patch]:
                        first_seen[patch] = location
            ordered = sorted(changing_channels[channel],
                             key=lambda patch: (first_seen[patch], patch))
            for patch in ordered[1:]:
                if not free_channels:
                    raise MidiLimitError(
                        "Not enough unoccupied MIDI channels for patch reassignment"
                    )
                reassigned[(channel, patch)] = free_channels.pop(0)
    remapping_ledger = [
        {
            "source_channel": channel,
            "output_channel": output_channel,
            "bank_msb": patch[0],
            "bank_lsb": patch[1],
            "program": patch[2],
        }
        for (channel, patch), output_channel in sorted(reassigned.items())
    ]
    tracks = []
    truncated = 0
    for (track_idx, channel, msb, lsb, program), spans in sorted(grouping.items()):
        output_channel = reassigned.get((channel, (msb, lsb, program)), channel)
        name = midi.tracks[track_idx].name or f"Track {track_idx + 1}"
        notes = []
        for ordinal, span in enumerate(spans):
            stop = span.sound_stop if audible_duration else span.key_release
            if stop <= span.start:
                truncated += 1
                continue
            notes.append(ScoreNote(
                _stable_note_id(track_idx, ordinal, span), span.start,
                stop - span.start, span.pitch, span.velocity,
                span.release_velocity,
            ))
        tracks.append(ScoreTrack(
            f"t{track_idx}-{output_channel}-{program}", name[:MAX_LABEL],
            output_channel, program, msb, lsb,
            tuple(sorted(notes, key=note_sort_key)),
        ))
    if not tracks:
        tracks = [ScoreTrack("empty", "Empty MIDI", 0, 0, 0, 0, ())]
    if len(tracks) > MAX_SCORE_TRACKS:
        raise MidiLimitError("Too many MIDI program/channel layers for score")
    end = max(1, midi.end)
    score = Score(title, midi.division.ppqn, tuple(tracks), tempo_map.points,
                  meter_map.points, end)
    ledger = {
        "protocol": "sonumio-workstation-midi-import-v1",
        "score_sha256": score.sha256,
        "notes_imported": score.note_count,
        "zero_duration_omitted": truncated,
        "original_sustain_applied": audible_duration,
        "performance_controller_messages_not_represented": True,
        "source_program_changes_supported": bool(reassigned),
        "source_program_changes_rechanneled": bool(reassigned),
        "program_channel_reassignments": remapping_ledger,
        "opaque_sysex_preserved": False,
        "key_signature_preserved": False,
        "source_audio_rendered": False,
    }
    return score, ledger


def to_midi(score: Score, *, include_muted: bool = True) -> MidiFile:
    """Export a portable format-1 MIDI arrangement with conductor track."""
    if not isinstance(score, Score) or type(include_muted) is not bool:
        raise MidiWorkstationError("Expected Score and include_muted flag")
    conductor = [
        MidiEvent.meta(0, META_TRACK_NAME, score.title.encode("utf-8")),
        *score.tempo_map.to_midi_events(),
        *score.meter_map.to_midi_events(),
    ]
    tracks = [MidiTrack.from_events(tuple(sorted(conductor, key=lambda e: e.tick)),
                                    end=score.end)]
    active = set(score.active_track_ids)
    for track in score.tracks:
        if not include_muted and track.id not in active:
            continue
        events: list[tuple[int, int, MidiEvent]] = [
            (0, 0, MidiEvent.meta(0, META_TRACK_NAME, track.name.encode("utf-8"))),
            (0, 1, MidiEvent.channel(0, 0xB, track.channel, 0, track.bank_msb)),
            (0, 2, MidiEvent.channel(0, 0xB, track.channel, 32, track.bank_lsb)),
            (0, 3, MidiEvent.channel(0, 0xC, track.channel, track.program)),
        ]
        for note in track.notes:
            events.append((note.stop, 4, MidiEvent.channel(
                note.stop, 0x8, track.channel, note.pitch, note.release_velocity)))
            events.append((note.start, 5, MidiEvent.channel(
                note.start, 0x9, track.channel, note.pitch, note.velocity)))
        events.sort(key=lambda entry: (entry[0], entry[1]))
        tracks.append(MidiTrack.from_events(tuple(item[2] for item in events), end=score.end))
    if len(tracks) > 64:
        raise MidiLimitError("Too many MIDI output tracks")
    return MidiFile(1, score.division, tuple(tracks))
