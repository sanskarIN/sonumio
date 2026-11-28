"""Deterministic MusicXML 4.0 score-partwise projection of bounded Sonumio scores.

This is a notation *projection*, not a promise of full Sonumio semantics.
No trained models, arbitrary file paths, or third-party notation renderer are
involved. The exporter rejects score concepts that would require inventing
musical structure and emits an explicit semantic-loss ledger.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterator
import xml.etree.ElementTree as ET

from .contracts import MidiFormatError, MidiLimitError, MidiWorkstationError
from .musicxml_common import (
    MAX_MEASURES, MAX_NOTE_FRAGMENTS, MAX_VOICES, MusicXmlReceipt,
    fingerprint, pitch_to_xml, serialize_xml, tempo_to_bpm, xml_tag,
)
from .score import Score, ScoreNote, ScoreTrack
from .tempo import MeterPoint, TempoPoint


@dataclass(frozen=True, slots=True)
class MeasureRange:
    index: int
    start: int
    stop: int
    meter: MeterPoint

    @property
    def duration(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class NoteFragment:
    note: ScoreNote
    start: int
    stop: int
    segment: int
    tied_from_previous: bool
    tied_to_next: bool

    @property
    def duration(self) -> int:
        return self.stop - self.start


def split_measures(score: Score) -> tuple[MeasureRange, ...]:
    """Bar layout is unambiguous only when meter changes start new bars."""
    if not isinstance(score, Score):
        raise MidiWorkstationError("Expected Sonumio score")
    changes = {point.tick: point for point in score.meter}
    position, result = 0, []
    while position < score.end:
        if len(result) >= MAX_MEASURES:
            raise MidiLimitError("MusicXML measure count exceeds limit")
        meter = score.meter_map.meter_at(position)
        span = meter.measure_ticks(score.ppqn)
        if span.denominator != 1:
            raise MidiFormatError("Score PPQN cannot represent whole metrical bars")
        span = int(span)
        if span <= 0:
            raise MidiFormatError("Zero-length MusicXML measure")
        stop = min(score.end, position + span)
        if any(position < tick < stop for tick in changes):
            raise MidiFormatError("Meter change in the middle of a measure is unsupported")
        result.append(MeasureRange(len(result), position, stop, meter))
        position = stop
    return tuple(result)


def split_fragments(score: Score, measures: tuple[MeasureRange, ...]) -> dict[str, dict[int, list[NoteFragment]]]:
    """Partition notes by measure in O(notes + fragments + log(measures))."""
    if not isinstance(score, Score) or not measures:
        raise MidiWorkstationError("Expected score and measure layout")
    starts = [m.start for m in measures]
    result = {track.id: {} for track in score.tracks}
    fragments = 0
    for track in score.tracks:
        for note in track.notes:
            index = bisect_right(starts, note.start) - 1
            if index < 0 or index >= len(measures):
                raise MidiFormatError("Note begins outside MusicXML measure layout")
            absolute_start = note.start
            part = 0
            while absolute_start < note.stop:
                if index >= len(measures):
                    raise MidiFormatError("Note exceeds MusicXML final measure")
                measure = measures[index]
                stop = min(measure.stop, note.stop)
                if stop <= absolute_start:
                    raise MidiFormatError("Invalid note subdivision across bars")
                result[track.id].setdefault(index, []).append(NoteFragment(
                    note, absolute_start - measure.start, stop - measure.start,
                    part, part > 0, stop < note.stop,
                ))
                fragments += 1
                if fragments > MAX_NOTE_FRAGMENTS:
                    raise MidiLimitError("MusicXML tied note fragment count exceeds limit")
                absolute_start = stop
                index += 1
                part += 1
    return result


def allocate_voices(fragments: list[NoteFragment]) -> tuple[tuple[NoteFragment, ...], ...]:
    """Interval partitioning into non-overlapping MusicXML voices.

    Identical note onsets occupy different voices, preserving timing without
    forcing unreliable chord interpretation. Reuse the lowest free voice.
    """
    ordered = sorted(fragments, key=lambda f: (
        f.start, f.stop, f.note.pitch, f.note.id, f.segment,
    ))
    voices: list[list[NoteFragment]] = []
    last_stop: list[int] = []
    for fragment in ordered:
        assigned = None
        for voice, end in enumerate(last_stop):
            if end <= fragment.start:
                assigned = voice
                break
        if assigned is None:
            if len(voices) >= MAX_VOICES:
                raise MidiLimitError("MusicXML polyphony exceeds 64 voice budget")
            assigned = len(voices)
            voices.append([])
            last_stop.append(0)
        voices[assigned].append(fragment)
        last_stop[assigned] = fragment.stop
    return tuple(tuple(voice) for voice in voices)


def _add_note(measure: ET.Element, fragment: NoteFragment, voice: int,
              part_idx: int) -> None:
    note = fragment.note
    element = xml_tag(measure, "note", id=f"sn_{note.id}_part{part_idx}_seg{fragment.segment}")
    step, alter, octave = pitch_to_xml(note.pitch)
    pitch = xml_tag(element, "pitch")
    xml_tag(pitch, "step", step)
    if alter:
        xml_tag(pitch, "alter", alter)
    xml_tag(pitch, "octave", octave)
    xml_tag(element, "duration", fragment.duration)
    xml_tag(element, "voice", voice)
    if fragment.tied_from_previous:
        xml_tag(element, "tie", type="stop")
    if fragment.tied_to_next:
        xml_tag(element, "tie", type="start")
    if fragment.tied_from_previous or fragment.tied_to_next:
        notations = xml_tag(element, "notations")
        if fragment.tied_from_previous:
            xml_tag(notations, "tied", type="stop")
        if fragment.tied_to_next:
            xml_tag(notations, "tied", type="start")


def _add_meter_and_divisions(part: ET.Element, measure: MeasureRange,
                             previous: MeterPoint | None, ppqn: int) -> None:
    if measure.index != 0 and measure.meter == previous:
        return
    attributes = xml_tag(part, "attributes")
    if measure.index == 0:
        xml_tag(attributes, "divisions", ppqn)
    time = xml_tag(attributes, "time")
    xml_tag(time, "beats", measure.meter.numerator)
    xml_tag(time, "beat-type", measure.meter.denominator)


def _add_tempo_directions(measure_node: ET.Element, range_: MeasureRange,
                          tempo: tuple[TempoPoint, ...], *, include_end: bool = False) -> None:
    # A MusicXML direction offset is relative to the current score cursor.
    # Directions precede all note voices, whose cursors are independent.
    for change in tempo:
        if range_.start <= change.tick < range_.stop or (include_end and change.tick == range_.stop):
            direction = xml_tag(measure_node, "direction")
            xml_tag(direction, "offset", change.tick - range_.start)
            xml_tag(direction, "sound", tempo=tempo_to_bpm(change.microseconds_per_quarter))


def _add_measure_voices(node: ET.Element, layout: MeasureRange,
                        fragments: list[NoteFragment], part_idx: int) -> int:
    voices = allocate_voices(fragments)
    for index, voice_fragments in enumerate(voices):
        if index:
            backup = xml_tag(node, "backup")
            xml_tag(backup, "duration", layout.duration)
        cursor = 0
        for fragment in voice_fragments:
            if fragment.start < cursor:
                raise MidiFormatError("Internal MusicXML voice overlap")
            if fragment.start > cursor:
                forward = xml_tag(node, "forward")
                xml_tag(forward, "duration", fragment.start - cursor)
            _add_note(node, fragment, index + 1, part_idx)
            cursor = fragment.stop
        if cursor < layout.duration:
            forward = xml_tag(node, "forward")
            xml_tag(forward, "duration", layout.duration - cursor)
    if not voices:
        forward = xml_tag(node, "forward")
        xml_tag(forward, "duration", layout.duration)
    return len(voices)


def encode_musicxml(score: Score) -> tuple[bytes, MusicXmlReceipt]:
    if not isinstance(score, Score):
        raise MidiWorkstationError("Expected bounded Sonumio score")
    if any(track.channel == 9 for track in score.tracks):
        raise MidiWorkstationError(
            "Pitched-note MusicXML projection cannot silently reinterpret drum notes"
        )
    if any(point.tick == score.end for point in score.meter):
        raise MidiFormatError("Meter changes exactly at score end are not representable")
    measures = split_measures(score)
    fragments = split_fragments(score, measures)
    root = ET.Element("score-partwise", version="4.0")
    xml_tag(root, "work")
    xml_tag(root, "movement-title", score.title)
    ident = xml_tag(root, "identification")
    misc = xml_tag(ident, "miscellaneous")
    xml_tag(misc, "miscellaneous-field", score.end, name="sonumio-end-tick")
    xml_tag(misc, "miscellaneous-field", score.sha256, name="sonumio-source-score-sha256")
    xml_tag(misc, "miscellaneous-field", "performance-subset", name="sonumio-projection")
    part_list = xml_tag(root, "part-list")
    for index, track in enumerate(score.tracks):
        part_id = f"sn-{track.id}"
        part = xml_tag(part_list, "score-part", id=part_id)
        xml_tag(part, "part-name", track.name)
        inst_id = f"I{index + 1}"
        instrument = xml_tag(part, "score-instrument", id=inst_id)
        xml_tag(instrument, "instrument-name", track.name)
        midi = xml_tag(part, "midi-instrument", id=inst_id)
        xml_tag(midi, "midi-channel", track.channel + 1)
        xml_tag(midi, "midi-program", track.program + 1)
    for index, track in enumerate(score.tracks):
        part = xml_tag(root, "part", id=f"sn-{track.id}")
        prior_meter: MeterPoint | None = None
        for layout in measures:
            node = xml_tag(part, "measure", number=layout.index + 1)
            _add_meter_and_divisions(node, layout, prior_meter, score.ppqn)
            prior_meter = layout.meter
            if index == 0:
                _add_tempo_directions(node, layout, score.tempo,
                                      include_end=layout.index == len(measures) - 1)
            _add_measure_voices(node, layout, fragments[track.id].get(layout.index, []), index)
    raw = serialize_xml(root)
    losses = ["note_velocity_and_release_velocity_not_exported",
              "note_pan_not_exported", "score_revision_not_exported",
              "track_mute_solo_not_exported", "MIDI_bank_selection_not_exported"]
    report = MusicXmlReceipt(score.sha256, fingerprint(raw), len(score.tracks),
                             len(measures), score.note_count, len(score.tempo),
                             len(score.meter), tuple(losses))
    return raw, report
