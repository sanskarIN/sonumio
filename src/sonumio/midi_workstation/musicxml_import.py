"""Strict import of the pitched-performance subset of MusicXML 3.1/4.0.

The parser never guesses arbitrary notation semantics. It accepts integer
MusicXML quarter divisions, exact note/rest duration, multiple independent
voices via backup/forward, bar-crossing ties, and global tempo/meter. Unsupported
repeat/navigation, grace, tuplet, transposed and microtonal notation is rejected
with a stable format error. Human engraving is not reconstructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import re
import xml.etree.ElementTree as ET

from .contracts import MAX_LABEL, MAX_PPQ, MAX_SCORE_NOTES, MAX_TICK, MidiFormatError, MidiLimitError, MidiWorkstationError
from .musicxml_common import (
    MUSICXML_PROTOCOL, MAX_MEASURES, MAX_NOTE_FRAGMENTS, MAX_PARTS, MAX_VOICES,
    MusicXmlReceipt, PART_ID, SONUMIO_ID, bounded_integer, bpm_to_tempo,
    child, children, fingerprint, integer_element, local_name,
    parse_musicxml, pitch_from_xml, required_text, validate_children,
    dynamics_to_velocity,
)
from .score import NOTE_ID_RE, Score, ScoreNote, ScoreTrack, note_sort_key
from .tempo import MeterPoint, TempoPoint


@dataclass(frozen=True, slots=True)
class _PartInfo:
    id: str
    name: str
    channel: int
    program: int
    bank_msb: int
    bank_lsb: int


@dataclass(frozen=True, slots=True)
class _NotePart:
    name: str
    pitch: int
    start: int
    stop: int
    voice: int
    tied_to_next: bool
    tied_from_previous: bool
    explicit_id: str | None
    velocity: int
    release_velocity: int


def _parse_parts(root: ET.Element) -> tuple[_PartInfo, ...]:
    part_list = child(root, "part-list", required=True)
    validate_children(part_list, {"score-part"}, context="part-list")
    result: list[_PartInfo] = []
    known_ids: set[str] = set()
    known_names: set[str] = set()
    for index, part in enumerate(children(part_list, "score-part")):
        if index >= MAX_PARTS:
            raise MidiLimitError("MusicXML part count exceeds 48")
        xml_id = part.get("id", "")
        if not 1 <= len(xml_id) <= 128 or xml_id in known_ids:
            raise MidiFormatError("Duplicate or invalid MusicXML part identifier")
        known_ids.add(xml_id)
        name = required_text(part, "part-name")
        if len(name) > MAX_LABEL:
            raise MidiLimitError("MusicXML part name exceeds label budget")
        matching = PART_ID.fullmatch(xml_id)
        stable_id = matching.group(1) if matching else f"part-{index + 1}"
        if stable_id in known_names:
            raise MidiFormatError("Part identifiers collapse to duplicate Sonumio names")
        known_names.add(stable_id)
        midi = child(part, "midi-instrument")
        # Imported part routing is explicit when available; missing routing
        # gets a conventional free pitched channel, never channel 10 drums.
        fallback_channels = [c for c in range(16) if c != 9]
        default = fallback_channels[index] if index < len(fallback_channels) else None
        if midi is None:
            if default is None:
                raise MidiFormatError("More than 15 MIDI parts require explicit channel assignments")
            channel = default
            program = 0
            bank_msb = bank_lsb = 0
        else:
            validate_children(midi, {"midi-channel", "midi-bank", "midi-program", "midi-name"},
                              context="MIDI instrument")
            raw_channel = child(midi, "midi-channel")
            raw_program = child(midi, "midi-program")
            if raw_channel is None or raw_program is None:
                raise MidiFormatError("MIDI instrument needs both channel and program")
            channel = integer_element(midi, "midi-channel", 1, 16) - 1
            program = integer_element(midi, "midi-program", 1, 128) - 1
            bank_node = child(midi, "midi-bank")
            bank = 1 if bank_node is None else integer_element(midi, "midi-bank", 1, 16384)
            bank_msb, bank_lsb = divmod(bank - 1, 128)
        if channel == 9:
            raise MidiFormatError("Drum channel 10 requires unsupported unpitched notation")
        result.append(_PartInfo(stable_id, name, channel, program, bank_msb, bank_lsb))
    if not result:
        raise MidiFormatError("Empty MusicXML part list")
    return tuple(result)


def _read_metadata(root: ET.Element) -> tuple[str, int | None]:
    title_node = child(root, "movement-title")
    title = "Imported MusicXML" if title_node is None else (title_node.text or "").strip()
    if not title or len(title) > MAX_LABEL:
        raise MidiFormatError("MusicXML title is empty or oversized")
    identification = child(root, "identification")
    end = None
    if identification is not None:
        misc = child(identification, "miscellaneous")
        if misc is not None:
            seen: set[str] = set()
            for field in children(misc, "miscellaneous-field"):
                key = field.get("name", "")
                if key in seen:
                    raise MidiFormatError("Duplicate MusicXML miscellaneous field")
                seen.add(key)
                if key == "sonumio-end-tick":
                    end = bounded_integer(field.text or "", "Sonumio end", 1, MAX_TICK)
    return title, end


def _parse_meter(attributes: ET.Element) -> MeterPoint | None:
    time = child(attributes, "time")
    if time is None:
        return None
    validate_children(time, {"beats", "beat-type"}, context="time signature")
    beats = integer_element(time, "beats", 1, 255)
    beat_type = integer_element(time, "beat-type", 1, 128)
    return MeterPoint(0, beats, beat_type)


def _parse_note(node: ET.Element, cursor: int, prior_start: int | None,
                prior_voice: int | None, divisions: int, ppqn: int,
                part_index: int) -> tuple[int, _NotePart | None, int, int]:
    supported = {"pitch", "duration", "rest", "chord", "voice", "tie",
                 "notations", "type", "dot", "accidental", "stem", "beam"}
    validate_children(node, supported, context="note feature")
    if child(node, "grace") is not None:
        raise MidiFormatError("Grace notes require unsupported playback timing")
    duration = integer_element(node, "duration", 1, MAX_TICK)
    exact = Fraction(duration * ppqn, divisions)
    if exact.denominator != 1:
        raise MidiFormatError("MusicXML note duration does not map exactly to score PPQN")
    length = int(exact)
    voice = 1
    voice_node = child(node, "voice")
    if voice_node is not None:
        voice = bounded_integer(voice_node.text or "", "voice", 1, MAX_VOICES)
    is_chord = child(node, "chord") is not None
    if is_chord and (prior_start is None or prior_voice != voice):
        raise MidiFormatError("MusicXML chord lacks a preceding compatible note")
    start = prior_start if is_chord else cursor
    end = start + length
    if child(node, "rest") is not None:
        if child(node, "pitch") is not None or is_chord or children(node, "tie"):
            raise MidiFormatError("MusicXML rest cannot have pitch/chord/tie")
        return (cursor if is_chord else cursor + length), None, start, voice
    pitch_node = child(node, "pitch", required=True)
    pitch = pitch_from_xml(pitch_node)
    dynamics_text = node.get("dynamics")
    end_dynamics_text = node.get("end-dynamics")
    velocity = 100 if dynamics_text is None else dynamics_to_velocity(dynamics_text, allow_zero=False)
    release_velocity = (0 if end_dynamics_text is None else
                        dynamics_to_velocity(end_dynamics_text, allow_zero=True))
    found_ties = [item.get("type") for item in children(node, "tie")]
    if len(found_ties) != len(set(found_ties)) or any(x not in ("start", "stop") for x in found_ties):
        raise MidiFormatError("Invalid MIDI playback tie annotations")
    notations = child(node, "notations")
    if notations is not None:
        validate_children(notations, {"tied"}, context="note notations")
        display_ties = [item.get("type") for item in children(notations, "tied")]
        if any(t not in ("start", "stop") for t in display_ties) or set(display_ties) != set(found_ties):
            raise MidiFormatError("Notation ties conflict with playback ties")
    source_id = node.get("id")
    explicit_id = None
    if source_id is not None:
        matching = SONUMIO_ID.fullmatch(source_id)
        if matching is not None:
            if int(matching.group(2)) != part_index:
                raise MidiFormatError("Sonumio note ID references the wrong MusicXML part")
            explicit_id = matching.group(1)
    return (cursor if is_chord else cursor + length), _NotePart(
        source_id or "", pitch, start, end, voice,
        "start" in found_ties, "stop" in found_ties, explicit_id,
        velocity, release_velocity,
    ), start, voice


def _convert_voices(parts: list[_NotePart], part_index: int, span_end: int) -> tuple[ScoreNote, ...]:
    """Combine exact adjacent tie chains; reject dangling or ambiguous ties."""
    active: dict[object, _NotePart] = {}
    notes: list[ScoreNote] = []
    ordinal = 0
    named: set[str] = set()
    def key_of(fragment: _NotePart):
        return ("id", fragment.explicit_id) if fragment.explicit_id else (
            "voice", fragment.pitch, fragment.voice)
    for fragment in sorted(parts, key=lambda f: (f.start, f.pitch, f.voice, f.stop)):
        if fragment.stop > span_end or fragment.stop <= fragment.start:
            raise MidiFormatError("Note fragment lies outside supported MusicXML score end")
        key = key_of(fragment)
        if fragment.tied_from_previous:
            prior = active.pop(key, None)
            if prior is None or prior.stop != fragment.start or prior.pitch != fragment.pitch:
                raise MidiFormatError("Unmatched or discontinuous MusicXML tie-stop")
            if fragment.explicit_id and prior.explicit_id != fragment.explicit_id:
                raise MidiFormatError("MusicXML tie changed stable note identity")
            if (fragment.velocity, fragment.release_velocity) != (prior.velocity, prior.release_velocity):
                raise MidiFormatError("Tied MusicXML fragments disagree on expressive dynamics")
            combined = _NotePart(prior.name, prior.pitch, prior.start, fragment.stop,
                                 prior.voice, fragment.tied_to_next, False, prior.explicit_id,
                                 prior.velocity, prior.release_velocity)
        else:
            if key in active:
                raise MidiFormatError("Conflicting MusicXML tie chains on one voice/pitch")
            combined = fragment
        if combined.tied_to_next:
            active[key] = combined
            continue
        note_id = combined.explicit_id or f"xml-p{part_index + 1}-n{ordinal:05d}"
        ordinal += 1
        if note_id in named or not NOTE_ID_RE.fullmatch(note_id):
            raise MidiFormatError("Duplicate or invalid MusicXML stable note ID")
        named.add(note_id)
        notes.append(ScoreNote(note_id, combined.start, combined.stop - combined.start,
                               combined.pitch, combined.velocity,
                               combined.release_velocity, 64))
        if len(notes) > MAX_SCORE_NOTES:
            raise MidiLimitError("MusicXML imported note count exceeds budget")
    if active:
        raise MidiFormatError("MusicXML contains unclosed playback ties")
    return tuple(sorted(notes, key=note_sort_key))


def decode_musicxml(raw: bytes) -> tuple[Score, MusicXmlReceipt]:
    root = parse_musicxml(raw)
    validate_children(root, {"work", "movement-title", "identification", "defaults",
                            "part-list", "part"}, context="score-partwise")
    part_info = _parse_parts(root)
    title, explicit_end = _read_metadata(root)
    xml_parts = children(root, "part")
    if len(xml_parts) != len(part_info) or any(p.get("id") != f"sn-{info.id}"
        and not (p.get("id") and not PART_ID.fullmatch(p.get("id")))
        for p, info in zip(xml_parts, part_info)):
        raise MidiFormatError("MusicXML part list disagrees with part bodies")
    if len(set(p.get("id") for p in xml_parts)) != len(xml_parts):
        raise MidiFormatError("Duplicate MusicXML part body ID")
    # Exact original reference IDs are resolved against part-list ordering.
    list_parts = children(child(root, "part-list", required=True), "score-part")
    if [p.get("id") for p in xml_parts] != [p.get("id") for p in list_parts]:
        raise MidiFormatError("MusicXML parts must match declared order")
    master_measure_starts: tuple[int, ...] | None = None
    master_measure_stops: tuple[int, ...] | None = None
    master_meters: dict[int, MeterPoint] = {0: MeterPoint(0, 4, 4)}
    master_tempos: dict[int, TempoPoint] = {0: TempoPoint(0, 500000)}
    track_rows = []
    ppqn = None
    total_fragments = 0
    for part_idx, (part, info) in enumerate(zip(xml_parts, part_info)):
        rows = children(part, "measure")
        if not 1 <= len(rows) <= MAX_MEASURES:
            raise MidiLimitError("MusicXML part has unsupported measure count")
        if part_idx > 0 and len(rows) != len(master_measure_starts):
            raise MidiFormatError("Parts use different MusicXML measure counts")
        fragments: list[_NotePart] = []
        measure_starts: list[int] = []
        measure_stops: list[int] = []
        position = 0
        current_meter = MeterPoint(0, 4, 4)
        division = None
        for measure_idx, measure in enumerate(rows):
            if measure.get("number", "") != str(measure_idx + 1):
                raise MidiFormatError("MusicXML requires consecutive numbered measures")
            measure_starts.append(position)
            cursor, prior_start, prior_voice = 0, None, None
            attributes_seen = False
            for item in measure:
                kind = local_name(item.tag)
                if kind == "attributes":
                    if cursor != 0 or attributes_seen:
                        raise MidiFormatError("MusicXML measure attributes must precede notes")
                    attributes_seen = True
                    validate_children(item, {"divisions", "time", "key", "clef"}, context="measure attributes")
                    candidate = child(item, "divisions")
                    if candidate is not None:
                        value = bounded_integer(candidate.text or "", "divisions", 1, MAX_PPQ)
                        if division is not None and value != division:
                            raise MidiFormatError("Changing MusicXML divisions is unsupported")
                        division = value
                        if ppqn is None:
                            ppqn = division
                        elif ppqn != division:
                            raise MidiFormatError("Part divisions do not match score PPQN")
                    meter = _parse_meter(item)
                    if meter is not None:
                        current_meter = MeterPoint(position, meter.numerator, meter.denominator)
                        if part_idx == 0:
                            previous = master_meters.get(position)
                            if previous is not None and previous != current_meter and position != 0:
                                raise MidiFormatError("Conflicting meter changes at same tick")
                            master_meters[position] = current_meter
                        else:
                            expected = master_meters.get(position)
                            if expected != current_meter:
                                raise MidiFormatError("A part has inconsistent global time signature")
                    if children(item, "key") or children(item, "clef"):
                        # These are visual notation metadata; pitches are
                        # always read as explicit step+alter+octave values.
                        pass
                elif kind == "direction":
                    validate_children(item, {"sound", "offset", "direction-type"}, context="direction")
                    sound = child(item, "sound")
                    direction_type = child(item, "direction-type")
                    if direction_type is not None:
                        validate_children(direction_type, {"metronome"},
                                          context="tempo direction-type")
                        metronome = child(direction_type, "metronome", required=True)
                        validate_children(metronome, {"beat-unit", "per-minute"},
                                          context="metronome")
                        if required_text(metronome, "beat-unit") != "quarter":
                            raise MidiFormatError("Only quarter-note metronome marks are supported")
                        marked_tempo = bpm_to_tempo(required_text(metronome, "per-minute"))
                    else:
                        marked_tempo = None
                    if sound is None or set(sound.attrib) != {"tempo"}:
                        raise MidiFormatError("Only MusicXML tempo sound directions are supported")
                    if marked_tempo is not None and marked_tempo != bpm_to_tempo(sound.attrib["tempo"]):
                        raise MidiFormatError("Metronome and playback tempo disagree")
                    if division is None or ppqn is None:
                        raise MidiFormatError("Tempo direction precedes divisions")
                    offset = child(item, "offset")
                    delta = 0 if offset is None else bounded_integer(offset.text or "", "direction offset", 0, MAX_TICK)
                    local_time = cursor + Fraction(delta * ppqn, division)
                    if local_time.denominator != 1:
                        raise MidiFormatError("MusicXML direction offset is off tick grid")
                    when = position + int(local_time)
                    if part_idx == 0:
                        val = TempoPoint(when, bpm_to_tempo(sound.attrib["tempo"]))
                        previous = master_tempos.get(when)
                        if previous is not None and previous != val and when != 0:
                            raise MidiFormatError("Conflicting simultaneous MusicXML tempo changes")
                        master_tempos[when] = val
                    else:
                        expected = master_tempos.get(when)
                        if expected is None or expected.microseconds_per_quarter != bpm_to_tempo(sound.attrib["tempo"]):
                            raise MidiFormatError("Non-conductor part has conflicting tempo")
                elif kind in ("backup", "forward"):
                    if division is None or ppqn is None:
                        raise MidiFormatError("MusicXML movement precedes divisions declaration")
                    validate_children(item, {"duration"}, context=kind)
                    raw_duration = integer_element(item, "duration", 1, MAX_TICK)
                    amount = Fraction(raw_duration * ppqn, division)
                    if amount.denominator != 1:
                        raise MidiFormatError("MusicXML cursor movement is off the score tick grid")
                    cursor += int(amount) if kind == "forward" else -int(amount)
                    if cursor < 0:
                        raise MidiFormatError("MusicXML backup crosses the measure boundary")
                    prior_start, prior_voice = None, None
                elif kind == "note":
                    if division is None or ppqn is None:
                        raise MidiFormatError("MusicXML note precedes divisions")
                    cursor, note, prior_start, prior_voice = _parse_note(
                        item, cursor, prior_start, prior_voice, division, ppqn, part_idx)
                    if note is not None:
                        fragments.append(_NotePart(note.name, note.pitch,
                            position + note.start, position + note.stop,
                            note.voice, note.tied_to_next, note.tied_from_previous,
                            note.explicit_id, note.velocity, note.release_velocity))
                        total_fragments += 1
                        if total_fragments > MAX_NOTE_FRAGMENTS:
                            raise MidiLimitError("MusicXML note fragment count exceeds budget")
                else:
                    raise MidiFormatError(f"Unsupported MusicXML measure feature: {kind}")
            if ppqn is None or division is None:
                raise MidiFormatError("First measure must declare note divisions")
            length = current_meter.measure_ticks(ppqn)
            if length.denominator != 1:
                raise MidiFormatError("Time signature has a fractional tick bar")
            # The last measure may be shorter: Sonumio embeds its true end
            # in a custom identification miscellaneous-field.
            stop = position + int(length)
            if explicit_end is not None and measure_idx == len(rows) - 1:
                if not position < explicit_end <= stop:
                    raise MidiFormatError("Embedded source end is outside final measure")
                stop = explicit_end
            if cursor > stop - position:
                raise MidiFormatError("MusicXML cursor exceeds the declared measure")
            position = stop
            measure_stops.append(stop)
            if part_idx > 0 and (
                    measure_starts[-1] != master_measure_starts[measure_idx]
                    or stop != master_measure_stops[measure_idx]):
                raise MidiFormatError("MusicXML parts disagree on measure boundaries")
        if part_idx == 0:
            master_measure_starts = tuple(measure_starts)
            master_measure_stops = tuple(measure_stops)
        tracks = _convert_voices(fragments, part_idx, position)
        track_rows.append(ScoreTrack(info.id, info.name, info.channel, info.program,
                                     info.bank_msb, info.bank_lsb, tracks))
    if ppqn is None or master_measure_stops is None:
        raise MidiFormatError("MusicXML has no valid quarter division")
    end = master_measure_stops[-1]
    # Drop repeated meter reassertions as they do not change performance time.
    meter_points = tuple(sorted((point for tick, point in master_meters.items()
                  if tick == 0 or point.numerator != master_meters.get(max(
                      (p for p in master_meters if p < tick), default=0), point).numerator
                  or point.denominator != master_meters.get(max(
                      (p for p in master_meters if p < tick), default=0), point).denominator),
                  key=lambda point: point.tick))
    tempo_points = tuple(point for _, point in sorted(master_tempos.items()))
    score = Score(title, ppqn, tuple(track_rows), tempo_points,
                  meter_points, end)
    losses = ("MusicXML_per_note_pan_not_represented_default_64",
              "notation_spelling_and_layout_not_retained",
              "articulation_and_engraving_not_retained",
              "absent_note_dynamics_default_to_100_and_release_0")
    receipt = MusicXmlReceipt(fingerprint(raw), score.sha256, len(track_rows),
                              len(master_measure_starts), score.note_count,
                              len(tempo_points), len(meter_points), losses)
    return score, receipt
