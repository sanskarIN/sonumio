"""Expressive MIDI note reconstruction with explicitly selected semantics.

Unlike a strict key-on/key-off-only importer, this optional decoder can pair
repeated same-pitch notes and preserve physical and sustain-held durations.
MIDI channel state is scoped to source tracks by default; this is a deliberate
choice to avoid inventing routing between unrelated source tracks.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterator

from .contracts import (
    MAX_NOTES, MAX_TICK, MidiFormatError, MidiLimitError,
    MidiWorkstationError, integer,
)
from .model import MidiEvent, MidiFile


@dataclass(frozen=True, slots=True)
class NoteSpan:
    track: int
    channel: int
    pitch: int
    velocity: int
    start: int
    key_release: int
    sound_stop: int
    release_velocity: int
    program: int
    bank_msb: int
    bank_lsb: int
    sustained: bool

    def __post_init__(self):
        for name, value, high in (
            ("track", self.track, 63), ("channel", self.channel, 15),
            ("pitch", self.pitch, 127), ("velocity", self.velocity, 127),
            ("release_velocity", self.release_velocity, 127),
            ("program", self.program, 127), ("bank_msb", self.bank_msb, 127),
            ("bank_lsb", self.bank_lsb, 127),
        ):
            integer(value, name, 0, high)
        integer(self.start, "note onset tick", 0, MAX_TICK)
        integer(self.key_release, "physical key release tick", self.start, MAX_TICK)
        integer(self.sound_stop, "audible release tick", self.key_release, MAX_TICK)
        if type(self.sustained) is not bool:
            raise MidiWorkstationError("Note sustained flag must be bool")

    @property
    def key_frames(self) -> int:
        return self.key_release - self.start

    @property
    def sound_frames(self) -> int:
        return self.sound_stop - self.start

    @property
    def key_duration(self) -> int:
        return self.key_release - self.start

    @property
    def audible_duration(self) -> int:
        return self.sound_stop - self.start

    @property
    def bank(self) -> int:
        return self.bank_msb * 128 + self.bank_lsb

    def as_dict(self) -> dict:
        return {
            "track": self.track, "channel": self.channel,
            "pitch": self.pitch, "velocity": self.velocity,
            "start": self.start, "key_release": self.key_release,
            "sound_stop": self.sound_stop,
            "key_duration_ticks": self.key_duration,
            "audible_duration_ticks": self.audible_duration,
            "release_velocity": self.release_velocity,
            "program": self.program, "bank_msb": self.bank_msb,
            "bank_lsb": self.bank_lsb, "sustained": self.sustained,
        }


@dataclass(frozen=True, slots=True)
class PerformanceSettings:
    same_pitch: str = "fifo"
    unmatched_note_off: str = "reject"
    dangling_note_on: str = "reject"
    sustain: bool = True
    isolate_tracks: bool = True
    all_notes_off: bool = True
    count_zero_duration: bool = False

    def __post_init__(self):
        if self.same_pitch not in ("fifo", "lifo", "reject"):
            raise MidiWorkstationError("Invalid same-pitch overlap policy")
        if self.unmatched_note_off not in ("reject", "ignore"):
            raise MidiWorkstationError("Invalid unmatched note-off policy")
        if self.dangling_note_on not in ("reject", "close"):
            raise MidiWorkstationError("Invalid dangling note policy")
        for name in ("sustain", "isolate_tracks", "all_notes_off", "count_zero_duration"):
            if type(getattr(self, name)) is not bool:
                raise MidiWorkstationError(f"{name} must be bool")


@dataclass(frozen=True, slots=True)
class PerformanceDiagnostics:
    tracks: int
    channel_messages: int
    physical_note_on_count: int
    physical_note_off_count: int
    audible_note_count: int
    overlapping_onsets: int
    ignored_note_offs: int
    forced_closures: int
    sustain_releases: int
    zero_duration_omissions: int
    auto_closed_at_end: bool
    protocol: str = "sonumio-midi-expressive-performance-v1"

    def as_dict(self) -> dict:
        from dataclasses import asdict
        value = asdict(self)
        value["controller_semantics_explicit"] = True
        value["audio_rendered"] = False
        value["model_inference"] = False
        return value


@dataclass(frozen=True, slots=True)
class Performance:
    notes: tuple[NoteSpan, ...]
    diagnostics: PerformanceDiagnostics

    def __post_init__(self):
        if type(self.notes) is not tuple or len(self.notes) > MAX_NOTES:
            raise MidiLimitError("Invalid bounded performance notes")
        if any(not isinstance(n, NoteSpan) for n in self.notes):
            raise MidiWorkstationError("Performance must contain NoteSpan")
        starts = [(n.start, n.track, n.channel, n.pitch, n.key_release, n.sound_stop)
                  for n in self.notes]
        if starts != sorted(starts):
            raise MidiWorkstationError("Performance notes must be deterministically ordered")

    @property
    def end(self) -> int:
        return max((note.sound_stop for note in self.notes), default=0)

    @property
    def tracks(self) -> tuple[int, ...]:
        return tuple(sorted({note.track for note in self.notes}))

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(sorted({note.channel for note in self.notes}))

    @property
    def pitches(self) -> tuple[int, ...]:
        return tuple(sorted({note.pitch for note in self.notes}))

    def filter(self, *, track: int | None = None, channel: int | None = None,
               minimum_pitch: int = 0, maximum_pitch: int = 127) -> "Performance":
        if track is not None:
            integer(track, "track filter", 0, 63)
        if channel is not None:
            integer(channel, "channel filter", 0, 15)
        integer(minimum_pitch, "minimum pitch", 0, 127)
        integer(maximum_pitch, "maximum pitch", minimum_pitch, 127)
        selected = tuple(n for n in self.notes if (
            (track is None or n.track == track)
            and (channel is None or n.channel == channel)
            and minimum_pitch <= n.pitch <= maximum_pitch
        ))
        return Performance(selected, self.diagnostics)


@dataclass(slots=True)
class _OpenNote:
    origin_track: int
    start: int
    pitch: int
    velocity: int
    program: int
    bank_msb: int
    bank_lsb: int
    release: int | None = None
    release_velocity: int = 0


class _ChannelState:
    __slots__ = ("program", "bank_msb", "bank_lsb", "sustain", "active", "held")

    def __init__(self):
        self.program = 0
        self.bank_msb = 0
        self.bank_lsb = 0
        self.sustain = False
        self.active: dict[int, deque[_OpenNote]] = defaultdict(deque)
        self.held: list[_OpenNote] = []


def _ordered_messages(midi: MidiFile, *, isolate_tracks: bool) -> list[tuple[int, int, int, MidiEvent]]:
    messages = []
    for track_index, track in enumerate(midi.tracks):
        for sequence, event in enumerate(track.events):
            if event.is_channel:
                messages.append((event.tick, track_index, sequence, event))
    messages.sort(key=lambda item: (item[0], item[1], item[2]))
    return messages


def reconstruct_performance(
    midi: MidiFile,
    settings: PerformanceSettings = PerformanceSettings(),
) -> Performance:
    """Match physical note and sustain-held intervals without mutating MIDI.

    A fixed event-order rule uses the original within-track order and sorted
    track number at simultaneous timestamps. A format-2 file needs isolated
    track states because its tracks are independent temporal sequences.
    """
    if not isinstance(midi, MidiFile) or not isinstance(settings, PerformanceSettings):
        raise MidiWorkstationError("Expected MIDI and PerformanceSettings")
    if midi.format == 2 and not settings.isolate_tracks:
        raise MidiWorkstationError("Independent format-2 tracks cannot share channels")
    channel_states: dict[tuple[int, int], _ChannelState] = defaultdict(_ChannelState)
    notes: list[NoteSpan] = []
    overlapping = ignored = forced = sustain_releases = zero = 0
    physical_on = physical_off = 0
    messages = _ordered_messages(midi, isolate_tracks=settings.isolate_tracks)
    tracks = {index: track.end for index, track in enumerate(midi.tracks)}

    def state_key(track: int, event: MidiEvent) -> tuple[int, int]:
        return (track if settings.isolate_tracks else 0, event.channel_number)

    def emit(channel: int, entry: _OpenNote, stop: int) -> None:
        nonlocal zero
        release = stop if entry.release is None else entry.release
        if stop < entry.start or release < entry.start:
            raise MidiFormatError("Note release precedes note onset")
        if stop == entry.start and not settings.count_zero_duration:
            zero += 1
            return
        # Shared channel state must not erase the track that actually
        # originated the note: its key-off, pedal or forced close may
        # legitimately arrive in another format-1 MIDI track.
        notes.append(NoteSpan(entry.origin_track, channel, entry.pitch, entry.velocity,
                              entry.start, release, stop, entry.release_velocity,
                              entry.program, entry.bank_msb, entry.bank_lsb,
                              entry.release is not None and stop > entry.release))
        if len(notes) > MAX_NOTES:
            raise MidiLimitError("Performance contains too many notes")

    def release_held(key: tuple[int, int], tick: int) -> None:
        nonlocal sustain_releases
        state = channel_states[key]
        if state.held:
            sustain_releases += len(state.held)
        for entry in state.held:
            emit(key[1], entry, tick)
        state.held.clear()

    for tick, track, _, event in messages:
        channel = event.channel_number
        kind = event.kind
        key = state_key(track, event)
        state = channel_states[key]
        if kind == 0xC:
            state.program = event.data[0]
        elif kind == 0xB:
            controller, value = event.data
            if controller == 0:
                state.bank_msb = value
            elif controller == 32:
                state.bank_lsb = value
            elif controller == 64 and settings.sustain:
                lowered = value < 64
                if lowered and state.sustain:
                    release_held(key, tick)
                state.sustain = not lowered
            elif controller in (120, 123) and settings.all_notes_off:
                for pitch, entries in list(state.active.items()):
                    for entry in entries:
                        entry.release = tick
                        emit(channel, entry, tick)
                        forced += 1
                state.active.clear()
                release_held(key, tick)
                if controller == 120:
                    state.sustain = False
            elif controller == 121 and settings.sustain:
                if state.sustain:
                    release_held(key, tick)
                state.sustain = False
        elif kind == 0x9 and event.data[1]:
            physical_on += 1
            pitch, velocity = event.data
            active = state.active[pitch]
            if active:
                overlapping += 1
                if settings.same_pitch == "reject":
                    raise MidiFormatError("Overlapping same-pitch note requires overlap policy")
            active.append(_OpenNote(track, tick, pitch, velocity, state.program,
                                    state.bank_msb, state.bank_lsb))
        elif kind == 0x8 or kind == 0x9 and event.data[1] == 0:
            physical_off += 1
            pitch = event.data[0]
            release_velocity = event.data[1]
            entries = state.active[pitch]
            if not entries:
                if settings.unmatched_note_off == "reject":
                    raise MidiFormatError("Note-off without a matching active note")
                ignored += 1
                continue
            entry = entries.pop() if settings.same_pitch == "lifo" else entries.popleft()
            entry.release = tick
            entry.release_velocity = release_velocity
            if state.sustain:
                state.held.append(entry)
            else:
                emit(channel, entry, tick)
            if not entries:
                del state.active[pitch]

    # Track end is a conservative default for outstanding key-on events.
    # The policy must be explicit because missing note-offs are otherwise
    # ambiguous. Held notes are closed even if sustain pedal never releases.
    auto_closed = False
    for key, state in sorted(channel_states.items()):
        track, channel = key
        end = tracks[track] if settings.isolate_tracks else midi.end
        if any(state.active.values()) or state.held:
            if settings.dangling_note_on == "reject":
                raise MidiFormatError("Unterminated MIDI performance note")
            auto_closed = True
        for entries in state.active.values():
            for entry in entries:
                entry.release = end
                emit(channel, entry, end)
                forced += 1
        for entry in state.held:
            emit(channel, entry, end)
            forced += 1

    ordered = tuple(sorted(notes, key=lambda n: (
        n.start, n.track, n.channel, n.pitch, n.key_release, n.sound_stop,
    )))
    return Performance(ordered, PerformanceDiagnostics(
        len(midi.tracks), len(messages), physical_on, physical_off,
        len(ordered), overlapping, ignored, forced, sustain_releases,
        zero, auto_closed,
    ))


def performance_report(performance: Performance) -> dict:
    if not isinstance(performance, Performance):
        raise MidiWorkstationError("Expected Performance")
    return {
        "protocol": "sonumio-midi-performance-report-v1",
        "notes": len(performance.notes),
        "tracks": list(performance.tracks),
        "channels": list(performance.channels),
        "range": [min(performance.pitches), max(performance.pitches)]
        if performance.pitches else None,
        "audible_end_tick": performance.end,
        "diagnostics": performance.diagnostics.as_dict(),
        "physical_key_release_preserved": True,
        "sustain_controls_applied": True,
        "audio_rendered": False,
    }
