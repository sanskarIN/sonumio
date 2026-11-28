"""Loss-conscious, bounded transforms on native Standard MIDI File events.

This is intentionally NOT an arrangement importer: arbitrary metadata,
SysEx, bank, controller, tempo, meter and timing information is preserved
byte-for-byte at the decoded event level. Only selected note pitch, poly
pressure key number, and nonzero key-on velocity may be edited. Key-signature
transposition is a separate, explicit, whole-score operation.

The note edits are transactional: overflow or accidental same-pitch note
collisions abort the operation rather than silently clipping or changing
how a synthesizer pairs note-on and note-off events.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict, deque

from .contracts import (
    MidiIntegrityError, MidiWorkstationError,
    integer, sha256,
)
from .model import META_KEY, MidiEvent, MidiFile, MidiTrack
from .smf import encode_smf
from .contracts import digest


@dataclass(frozen=True, slots=True)
class MidiTransformSettings:
    """Explicit selection and exact-rational note-expression operations.

    ``velocity_numerator / velocity_denominator`` is rounded to nearest
    integer, ties upwards. The signed offset is added *after* scaling.
    Nonzero key-on velocity is clamped to [1, 127], never converted into
    a note-off. Note-off/release velocity is always preserved.
    """

    semitones: int = 0
    velocity_numerator: int = 1
    velocity_denominator: int = 1
    velocity_offset: int = 0
    pitch_low: int = 0
    pitch_high: int = 127
    tracks: tuple[int, ...] | None = None
    channels: tuple[int, ...] | None = None
    include_percussion: bool = False
    transpose_key_signatures: bool = False
    reject_new_collisions: bool = True

    def __post_init__(self):
        integer(self.semitones, "transpose semitones", -48, 48)
        integer(self.velocity_numerator, "velocity numerator", 1, 1024)
        integer(self.velocity_denominator, "velocity denominator", 1, 1024)
        integer(self.velocity_offset, "velocity offset", -126, 126)
        integer(self.pitch_low, "pitch selection low", 0, 127)
        integer(self.pitch_high, "pitch selection high", self.pitch_low, 127)
        for name, selection, maximum in (("tracks", self.tracks, 63),
                                          ("channels", self.channels, 15)):
            if selection is None:
                continue
            if (type(selection) is not tuple or not selection
                    or any(type(item) is not int or not 0 <= item <= maximum
                           for item in selection)
                    or tuple(sorted(set(selection))) != selection):
                raise MidiWorkstationError(
                    f"{name} must be a nonempty canonical tuple of unique indices"
                )
        if any(type(flag) is not bool for flag in (
                self.include_percussion, self.transpose_key_signatures,
                self.reject_new_collisions)):
            raise MidiWorkstationError("MIDI transform switches must be boolean")
        if self.transpose_key_signatures and (
                self.tracks is not None or self.channels is not None
                or self.pitch_low != 0 or self.pitch_high != 127):
            raise MidiWorkstationError(
                "Key signature changes require a whole-score pitch selection"
            )


@dataclass(frozen=True, slots=True)
class MidiTransformReport:
    source_sha256: str
    result_sha256: str
    source_events: int
    selected_note_ons: int
    selected_note_offs: int
    changed_pitch_events: int
    changed_velocity_events: int
    clamped_velocities: int
    changed_key_signatures: int
    untouched_percussion_events: int
    unchanged_nonpitch_events: int
    key_signatures_preserved_without_transposition: int
    output_bytes: int
    collisions_before: int
    collisions_after: int
    protocol: str = "sonumio-midi-event-transform-v1"

    def as_dict(self):
        return {
            **asdict(self),
            "source_mutated": False,
            "opaque_meta_sysex_preserved": True,
            "controllers_preserved": True,
            "source_audio_rendered": False,
            "trained_model_inference": False,
            "timing_changed": False,
            "release_velocity_changed": False,
        }


def _key_signature_shift(data: bytes, semitones: int) -> bytes:
    """Transpose the declared tonic, choosing a minimal-accidental spelling.

    MIDI key signature stores sharps/flats count and major/minor mode,
    not a specific enhancer spelling. Enharmonic reassignment is therefore
    unavoidable in some keys and is explicitly reported to callers.
    """
    fifths = int.from_bytes(data[:1], "big", signed=True)
    minor = data[1]
    original_root = (7 * fifths + (9 if minor else 0)) % 12
    target_root = (original_root + semitones) % 12
    choices = [value for value in range(-7, 8)
               if (7 * value + (9 if minor else 0)) % 12 == target_root]
    if not choices:
        raise MidiWorkstationError("No MIDI key-signature spelling available")
    new_fifths = min(choices, key=lambda value: (
        abs(value), abs(value - fifths),
        (value < 0) if fifths >= 0 else (value > 0),
    ))
    return bytes((new_fifths & 255, minor))


def _collisions(midi: MidiFile) -> frozenset[tuple[int, int]]:
    """Track event identities whose onset introduced same-key ambiguity.

    Format-0/1 channels are global across tracks; format-2 tracks are
    independent sequences. Event identity (track, original event offset)
    survives transposition and avoids false alarms when an *existing*
    overlap is shifted uniformly to a different MIDI key.

    Holding only onset identities is bounded by MAX_EVENTS. A different
    overlap replacing an old one cannot be hidden by a net collision count.
    """
    events = [
        (event.tick, track_index, seq, event)
        for track_index, track in enumerate(midi.tracks)
        for seq, event in enumerate(track.events)
        if event.is_note_on or event.is_note_off
    ]
    events.sort(key=lambda item: (item[0], item[1], item[2]))
    active: dict[tuple[int, int, int], deque[tuple[int, int]]] = defaultdict(deque)
    ambiguous = set()
    for _, track_index, seq, event in events:
        namespace = track_index if midi.format == 2 else 0
        key = (namespace, event.channel_number, event.data[0])
        held = active[key]
        if event.is_note_on:
            if held:
                ambiguous.add((track_index, seq))
            held.append((track_index, seq))
        elif held:
            held.popleft()
    return frozenset(ambiguous)


def transform_midi(midi: MidiFile,
                   settings: MidiTransformSettings = MidiTransformSettings(),
                   *, expected_source_sha256: str | None = None
                   ) -> tuple[MidiFile, MidiTransformReport]:
    """Apply one atomic transformation and return immutable MIDI + audit.

    No event is reordered or retimed. Source MIDI is left unchanged even if
    a late resource limit, bad pitch, integrity failure or collision aborts.
    """
    if not isinstance(midi, MidiFile) or not isinstance(settings, MidiTransformSettings):
        raise MidiWorkstationError("Expected validated MidiFile and transform settings")
    if settings.tracks is not None and max(settings.tracks) >= len(midi.tracks):
        raise MidiWorkstationError("Selected MIDI track is absent")
    if expected_source_sha256 is not None:
        sha256(expected_source_sha256)
    before_bytes = encode_smf(midi)
    before_sha256 = digest(before_bytes)
    if expected_source_sha256 is not None and expected_source_sha256 != before_sha256:
        raise MidiIntegrityError("Canonical MIDI source digest differs from pinned digest")

    selected_on = selected_off = changed_pitch = changed_vel = clamped = 0
    percussion = unchanged = signature_changes = preserved_sig = 0
    result_tracks = []
    for track_index, track in enumerate(midi.tracks):
        result_events = []
        for event in track.events:
            if event.status == 0xFF and event.meta_type == META_KEY:
                if settings.transpose_key_signatures and settings.semitones:
                    updated = _key_signature_shift(event.data, settings.semitones)
                    result_events.append(MidiEvent.meta(event.tick, META_KEY, updated))
                    signature_changes += int(updated != event.data)
                else:
                    preserved_sig += 1
                    result_events.append(event)
                continue
            if not event.is_channel or event.kind not in (0x8, 0x9, 0xA):
                result_events.append(event)
                unchanged += 1
                continue
            if event.kind == 0x9 and event.data[1] == 0:
                is_off = True
            else:
                is_off = event.kind == 0x8
            channel = event.channel_number
            pitch = event.data[0]
            if not settings.include_percussion and channel == 9:
                result_events.append(event)
                percussion += 1
                continue
            if (settings.tracks is not None and track_index not in settings.tracks
                    or settings.channels is not None and channel not in settings.channels
                    or not settings.pitch_low <= pitch <= settings.pitch_high):
                result_events.append(event)
                unchanged += 1
                continue
            is_on = event.kind == 0x9 and not is_off
            if is_on:
                selected_on += 1
            elif is_off:
                selected_off += 1
            new_pitch = pitch + settings.semitones
            if not 0 <= new_pitch <= 127:
                raise MidiWorkstationError(
                    f"Transposition would move MIDI key {pitch} outside 0..127"
                )
            data = bytearray(event.data)
            data[0] = new_pitch
            changed_pitch += int(new_pitch != pitch)
            if is_on:
                # Integer arithmetic rather than float rounding preserves
                # exact deterministic behavior across Python versions.
                raw = ((2 * event.data[1] * settings.velocity_numerator
                        + settings.velocity_denominator)
                       // (2 * settings.velocity_denominator))
                raw += settings.velocity_offset
                velocity = max(1, min(127, raw))
                clamped += int(raw != velocity)
                data[1] = velocity
                changed_vel += int(velocity != event.data[1])
            result_events.append(MidiEvent(event.tick, event.status, bytes(data)))
        result_tracks.append(MidiTrack(tuple(result_events)))
    transformed = midi.with_tracks(tuple(result_tracks))
    original_ambiguity = _collisions(midi)
    resulting_ambiguity = _collisions(transformed)
    before_collisions = len(original_ambiguity)
    after_collisions = len(resulting_ambiguity)
    if settings.reject_new_collisions and resulting_ambiguity - original_ambiguity:
        raise MidiWorkstationError(
            "Transposition introduces overlapping notes at identical pitches; "
            "refine selected tracks/channels or use an explicit collision policy"
        )
    result_bytes = encode_smf(transformed)
    result_sha256 = digest(result_bytes)
    report = MidiTransformReport(
        source_sha256=before_sha256, result_sha256=result_sha256,
        source_events=len(midi.events), selected_note_ons=selected_on,
        selected_note_offs=selected_off,
        changed_pitch_events=changed_pitch, changed_velocity_events=changed_vel,
        clamped_velocities=clamped, changed_key_signatures=signature_changes,
        untouched_percussion_events=percussion,
        unchanged_nonpitch_events=unchanged,
        key_signatures_preserved_without_transposition=preserved_sig,
        output_bytes=len(result_bytes), collisions_before=before_collisions,
        collisions_after=after_collisions,
    )
    return transformed, report
