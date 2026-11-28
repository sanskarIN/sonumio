"""Read-only, bounded MIDI semantic diagnostics for interchange/research.

The binary decoder enforces structural validity; linting adds observations
about musical performance semantics that the MIDI standard itself allows:
missing note releases, ambiguous same-pitch overlaps, controller termination,
conflicting conductor metadata, and patch switches during sounding notes.
No suggested 'repair' is silently performed, and opaque MIDI/SysEx data is
never interpreted as commands. Lint receipts are deterministic and bounded.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Literal

from .contracts import MidiWorkstationError, integer, digest
from .model import MidiFile
from .smf import encode_smf

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True, slots=True)
class LintSettings:
    maximum_findings: int = 256
    include_information: bool = True

    def __post_init__(self):
        integer(self.maximum_findings, "lint finding budget", 1, 2048)
        if type(self.include_information) is not bool:
            raise MidiWorkstationError("lint include_information must be bool")


@dataclass(frozen=True, slots=True)
class LintFinding:
    code: str
    severity: Severity
    track: int
    tick: int
    message: str
    channel: int | None = None
    pitch: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LintReport:
    midi_sha256: str
    format: int
    track_count: int
    source_events: int
    channel_events: int
    note_ons: int
    note_offs: int
    sysex_events: int
    tempo_events: int
    meter_events: int
    matched_notes: int
    unclosed_notes: int
    total_findings: int
    findings_omitted: int
    severity_counts: dict[str, int]
    issue_counts: dict[str, int]
    findings: tuple[LintFinding, ...]
    status: str
    protocol: str = "sonumio-midi-lint-v1"

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "findings": [finding.as_dict() for finding in self.findings],
            "no_input_changes": True,
            "source_audio_rendered": False,
            "opaque_sysex_not_interpreted": True,
            "trained_model_inference": False,
        }


def lint_midi(midi: MidiFile, settings: LintSettings = LintSettings()) -> LintReport:
    """Inspect format 0/1 shared-channel or format 2 independent sequences.

    The report includes all issue totals even if individual results are
    truncated at maximum_findings, keeping memory bounded for hostile files.
    Note-on/off pairing follows FIFO as a conservative diagnostic policy;
    it does not claim a specific synthesizer's voice-stealing behavior.
    """
    if not isinstance(midi, MidiFile) or not isinstance(settings, LintSettings):
        raise MidiWorkstationError("Expected typed MidiFile and LintSettings")

    findings: list[LintFinding] = []
    issues = Counter()
    severity = Counter()
    total = 0

    def record(code: str, level: Severity, track: int, tick: int,
               message: str, *, channel: int | None = None,
               pitch: int | None = None):
        nonlocal total
        if level == "info" and not settings.include_information:
            return
        total += 1
        issues[code] += 1
        severity[level] += 1
        if len(findings) < settings.maximum_findings:
            findings.append(LintFinding(code, level, track, tick, message,
                                        channel, pitch))

    # Each note-on has a stable provenance even if another type-1 track
    # releases it later. For type-2 the tracks are independent sequences.
    active: dict[tuple[int, int, int], deque[tuple[int, int]]] = defaultdict(deque)
    note_on_per_channel: dict[tuple[int, int], int] = defaultdict(int)
    active_keys: dict[tuple[int, int], set[tuple[int, int, int]]] = defaultdict(set)
    sustained: dict[tuple[int, int], bool] = defaultdict(bool)
    last_pedal_track: dict[tuple[int, int], int] = {}
    matched = 0
    notes_on = notes_off = sysex = channels = tempos = meters = 0

    events = [
        (event.tick, track_index, sequence, event)
        for track_index, track in enumerate(midi.tracks)
        for sequence, event in enumerate(track.events)
    ]
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    conductors: dict[tuple[int, int, int], tuple[bytes, int]] = {}
    tempo_at_zero = False
    for tick, track, _, event in events:
        namespace = track if midi.format == 2 else 0
        if event.is_tempo or event.is_meter:
            kind = event.meta_type
            if event.is_tempo:
                tempos += 1
                tempo_at_zero |= tick == 0
            else:
                meters += 1
            marker = (namespace, tick, kind)
            previous = conductors.get(marker)
            if previous is None:
                conductors[marker] = (event.data, track)
            elif previous[0] != event.data:
                record(
                    "conflicting_conductor", "error", track, tick,
                    "Simultaneous tempo or meter events disagree across tracks",
                )
            else:
                record(
                    "duplicate_conductor", "info", track, tick,
                    "Identical conductor metadata is repeated at the same tick",
                )
        if event.is_sysex:
            sysex += 1
            continue
        if not event.is_channel:
            continue
        channels += 1
        channel = event.channel_number
        key = (namespace, channel)
        status = event.kind
        if status == 0x9 and event.data[1]:
            notes_on += 1
            pitch = event.data[0]
            note_key = (*key, pitch)
            if active[note_key]:
                record("overlapping_same_pitch", "warning", track, tick,
                       "Another identical MIDI pitch is still held on this channel",
                       channel=channel, pitch=pitch)
            active[note_key].append((tick, track))
            active_keys[key].add(note_key)
            note_on_per_channel[key] += 1
        elif status == 0x8 or (status == 0x9 and event.data[1] == 0):
            notes_off += 1
            pitch = event.data[0]
            note_key = (*key, pitch)
            if not active[note_key]:
                record("unmatched_note_off", "error", track, tick,
                       "No held note-on matches this note-off",
                       channel=channel, pitch=pitch)
                continue
            start, origin_track = active[note_key].popleft()
            note_on_per_channel[key] -= 1
            if not active[note_key]:
                active_keys[key].discard(note_key)
            matched += 1
            if tick == start:
                record("zero_length_note", "warning", track, tick,
                       "Note release is at the same tick as note onset",
                       channel=channel, pitch=pitch)
            if midi.format == 1 and track != origin_track:
                record("cross_track_note_off", "info", track, tick,
                       "Note release came from a different format-1 MIDI track",
                       channel=channel, pitch=pitch)
        elif status == 0xB:
            control, value = event.data
            if control == 64:
                sustained[key] = value >= 64
                last_pedal_track[key] = track
            elif control in (120, 123) and note_on_per_channel[key]:
                # MIDI channel-mode all-sounds-off / all-notes-off may
                # stop held voices. The actual audible tail is synthesis
                # dependent; document the ambiguity, then clear note keys.
                record("channel_mode_closes_notes", "warning", track, tick,
                       "Channel mode message closes held notes without key-offs",
                       channel=channel)
                for note_key in active_keys[key]:
                    active[note_key].clear()
                active_keys[key].clear()
                note_on_per_channel[key] = 0
            elif control == 121 and sustained[key]:
                record("controllers_reset_while_sustaining", "warning", track,
                       tick, "Controller reset changes held sustain-pedal state",
                       channel=channel)
                sustained[key] = False
        elif status == 0xC and note_on_per_channel[key]:
            record("patch_change_during_held_notes", "warning", track, tick,
                   "Program change occurs while MIDI notes are still held",
                   channel=channel)
    if not midi.division.smpte and midi.format != 2 and not tempo_at_zero:
        record("implicit_initial_tempo", "info", 0, 0,
               "No tempo at tick zero: playback defaults to 500000 microseconds per quarter")
    unclosed = 0
    for (namespace, channel, pitch), held in sorted(active.items()):
        for start, origin_track in held:
            unclosed += 1
            record("unclosed_note_on", "error", origin_track, start,
                   "Note-on has no matching note-off or channel-wide closure",
                   channel=channel, pitch=pitch)
    for key in sorted(sustained):
        if sustained[key]:
            record("sustain_left_down", "info", last_pedal_track[key],
                   midi.tracks[key[0]].end if midi.format == 2 else midi.end,
                   "Sustain pedal remains down at the end of the MIDI sequence",
                   channel=key[1])
    findings.sort(key=lambda item: (
        item.tick, item.track, item.severity, item.code,
        -1 if item.channel is None else item.channel,
        -1 if item.pitch is None else item.pitch,
    ))
    status = "errors" if severity["error"] else (
        "warnings" if severity["warning"] else "clean"
    )
    return LintReport(
        midi_sha256=digest(encode_smf(midi)), format=midi.format,
        track_count=len(midi.tracks), source_events=len(midi.events),
        channel_events=channels, note_ons=notes_on, note_offs=notes_off,
        sysex_events=sysex, tempo_events=tempos, meter_events=meters,
        matched_notes=matched, unclosed_notes=unclosed,
        total_findings=total, findings_omitted=total-len(findings),
        severity_counts={name:severity[name] for name in ("error", "warning", "info")},
        issue_counts=dict(sorted(issues.items())),
        findings=tuple(findings), status=status,
    )
