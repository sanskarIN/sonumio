"""Defensive MusicXML 4.0 interchange helpers.

MusicXML is a large engraving language. Sonumio deliberately supports a
clearly defined performance subset: score-partwise, integer divisions,
bar-aligned meter, explicit pitched notes, rest/forward/backup voice timing,
notes tied across bars, and sound tempo. Unsupported constructs fail closed.
There is no network, DTD, XML entity expansion, or external resource access.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from fractions import Fraction
from hashlib import sha256
import re
import xml.etree.ElementTree as ET

from .contracts import MidiFormatError, MidiLimitError, MidiWorkstationError, integer

MUSICXML_PROTOCOL = "sonumio-musicxml-performance-subset-v1"
MAX_MUSICXML_BYTES = 4 * 1024 * 1024
MAX_MUSICXML_ELEMENTS = 160_000
MAX_XML_DEPTH = 48
MAX_XML_TEXT = 16384
MAX_PARTS = 48
MAX_MEASURES = 512
MAX_NOTE_FRAGMENTS = 96_000
MAX_VOICES = 64
MAX_XML_NAME = 256
MAX_XML_DURATION = 0x0FFFFFFF
PITCH_STEPS = ("C", "D", "E", "F", "G", "A", "B")
PITCH_BASE = (0, 2, 4, 5, 7, 9, 11)
PITCH_NAMES = (("C", 0), ("C", 1), ("D", 0), ("D", 1),
               ("E", 0), ("F", 0), ("F", 1), ("G", 0),
               ("G", 1), ("A", 0), ("A", 1), ("B", 0))
SONUMIO_ID = re.compile(r"^sn_([a-zA-Z0-9][a-zA-Z0-9_-]{0,63})_part(\d+)_seg(\d+)$")
PART_ID = re.compile(r"^sn-([A-Za-z0-9][A-Za-z0-9_-]{0,63})$")


def local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


def children(parent: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in parent if local_name(c.tag) == name]


def child(parent: ET.Element, name: str, *, required: bool = False) -> ET.Element | None:
    found = children(parent, name)
    if len(found) > 1:
        raise MidiFormatError(f"Duplicate MusicXML {name} element")
    if not found:
        if required:
            raise MidiFormatError(f"Missing MusicXML {name} element")
        return None
    return found[0]


def required_text(parent: ET.Element, name: str) -> str:
    node = child(parent, name, required=True)
    if len(node) or node.text is None or not node.text.strip():
        raise MidiFormatError(f"Expected plain text in MusicXML {name}")
    return node.text.strip()


def bounded_integer(text: str, label: str, minimum: int, maximum: int) -> int:
    if type(text) is not str or not re.fullmatch(r"[0-9]{1,12}", text.strip()):
        raise MidiFormatError(f"Invalid MusicXML {label} integer")
    value = int(text.strip())
    if not minimum <= value <= maximum:
        raise MidiFormatError(f"MusicXML {label} integer outside supported range")
    return value


def integer_element(parent: ET.Element, name: str, minimum: int, maximum: int) -> int:
    return bounded_integer(required_text(parent, name), name, minimum, maximum)


def validate_children(node: ET.Element, supported: set[str], *, context: str) -> None:
    for item in node:
        if local_name(item.tag) not in supported:
            raise MidiFormatError(f"Unsupported {context}: {local_name(item.tag)}")


def parse_musicxml(raw: bytes) -> ET.Element:
    if type(raw) is not bytes or not 40 <= len(raw) <= MAX_MUSICXML_BYTES:
        raise MidiLimitError("MusicXML document size outside allowed bounds")
    # ElementTree never fetches external entities, but blocking any DTD and
    # entity declaration makes the input policy predictable across runtimes.
    if b"<!" in raw.upper() or b"<?" in raw[8:]:
        raise MidiFormatError("DTD, entity declarations and processing instructions disallowed")
    try:
        if raw.startswith(b"\xef\xbb\xbf"):
            raw = raw[3:]
        decoded = raw.decode("utf-8", "strict")
        root = ET.fromstring(decoded)
    except (UnicodeError, ET.ParseError, ValueError, RecursionError) as exc:
        raise MidiFormatError("Malformed UTF-8 MusicXML document") from exc
    if local_name(root.tag) != "score-partwise":
        raise MidiFormatError("Only MusicXML score-partwise documents are supported")
    if root.attrib.get("version", "4.0") not in ("3.1", "4.0"):
        raise MidiFormatError("Unsupported MusicXML version")
    count = 0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        count += 1
        if count > MAX_MUSICXML_ELEMENTS:
            raise MidiLimitError("MusicXML element budget exceeded")
        if depth > MAX_XML_DEPTH:
            raise MidiLimitError("MusicXML nesting exceeds depth limit")
        if len(node.attrib) > 16 or any(len(k) > MAX_XML_NAME or len(v) > MAX_XML_TEXT
                                        for k, v in node.attrib.items()):
            raise MidiLimitError("MusicXML attribute size budget exceeded")
        if node.text is not None and len(node.text) > MAX_XML_TEXT:
            raise MidiLimitError("MusicXML element text is oversized")
        if node.tail is not None and len(node.tail) > MAX_XML_TEXT:
            raise MidiLimitError("MusicXML tail text is oversized")
        stack.extend((c, depth + 1) for c in node)
    return root


def pitch_to_xml(pitch: int) -> tuple[str, int, int]:
    integer(pitch, "pitch", 0, 127)
    step, alter = PITCH_NAMES[pitch % 12]
    return step, alter, pitch // 12 - 1


def pitch_from_xml(elem: ET.Element) -> int:
    validate_children(elem, {"step", "alter", "octave"}, context="pitch")
    step = required_text(elem, "step")
    if step not in PITCH_STEPS:
        raise MidiFormatError("Invalid MusicXML pitch step")
    octave_text = required_text(elem, "octave")
    if not re.fullmatch(r"-?[0-9]{1,2}", octave_text):
        raise MidiFormatError("Invalid MusicXML octave")
    octave = int(octave_text)
    if not -1 <= octave <= 9:
        raise MidiFormatError("MusicXML octave outside MIDI-pitch bounds")
    alter = child(elem, "alter")
    semitone = 0
    if alter is not None:
        text = (alter.text or "").strip()
        if text not in ("-2", "-1", "0", "1", "2"):
            raise MidiFormatError("Microtonal or invalid pitch alteration is unsupported")
        semitone = int(text)
    midi = (octave + 1) * 12 + PITCH_BASE[PITCH_STEPS.index(step)] + semitone
    return integer(midi, "converted MIDI pitch", 0, 127)


def tempo_to_bpm(microseconds: int) -> str:
    integer(microseconds, "MusicXML tempo", 1, 0xFFFFFF)
    with localcontext() as ctx:
        ctx.prec = 64
        return format(Decimal(60_000_000) / Decimal(microseconds), ".28f").rstrip("0").rstrip(".")


def bpm_to_tempo(value: str) -> int:
    if type(value) is not str or len(value) > 96:
        raise MidiFormatError("Invalid tempo BPM text")
    try:
        with localcontext() as ctx:
            ctx.prec = 64
            bpm = Decimal(value)
            if not bpm.is_finite() or bpm <= 0:
                raise MidiFormatError("MusicXML tempo must be positive and finite")
            microseconds = int((Decimal(60_000_000) / bpm).to_integral_value())
            return integer(microseconds, "MusicXML tempo", 1, 0xFFFFFF)
    except (InvalidOperation, OverflowError, ZeroDivisionError) as exc:
        raise MidiFormatError("Invalid MusicXML BPM value") from exc


def velocity_to_dynamics(velocity: int) -> str:
    """MusicXML percentage of a 90-velocity reference, not 7-bit MIDI raw.

    Fixed decimal precision keeps exported values portable through text
    editors and MusicXML validators while ensuring integer round-trip for
    all 128 possible release velocities and 127 onset velocities.
    """
    integer(velocity, "MusicXML MIDI velocity", 0, 127)
    with localcontext() as ctx:
        ctx.prec = 48
        return format(Decimal(velocity) * Decimal(100) / Decimal(90), ".18f").rstrip("0").rstrip(".")


def dynamics_to_velocity(text: str, *, allow_zero: bool) -> int:
    if type(text) is not str or not 1 <= len(text) <= 40:
        raise MidiFormatError("Invalid MusicXML dynamics decimal")
    if not re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{1,24})?", text):
        raise MidiFormatError("Invalid MusicXML dynamics number syntax")
    try:
        value = Decimal(text)
        if value < 0 or value > Decimal(127) * 100 / 90:
            raise MidiFormatError("MusicXML dynamics outside MIDI 7-bit range")
        midi = int((value * Decimal(90) / Decimal(100)).to_integral_value(rounding=ROUND_HALF_UP))
        return integer(midi, "MusicXML dynamics MIDI velocity", 0 if allow_zero else 1, 127)
    except (InvalidOperation, ValueError, OverflowError) as exc:
        raise MidiFormatError("Invalid MusicXML dynamics value") from exc


def serialize_xml(root: ET.Element) -> bytes:
    if not isinstance(root, ET.Element):
        raise MidiWorkstationError("Expected XML element")
    raw = ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    if len(raw) > MAX_MUSICXML_BYTES:
        raise MidiLimitError("MusicXML result exceeds byte budget")
    return raw


@dataclass(frozen=True, slots=True)
class MusicXmlReceipt:
    source_sha256: str
    output_sha256: str
    parts: int
    measures: int
    notes: int
    tempo_points: int
    meter_points: int
    losses: tuple[str, ...]
    protocol: str = MUSICXML_PROTOCOL

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
            "output_sha256": self.output_sha256,
            "parts": self.parts,
            "measures": self.measures,
            "notes": self.notes,
            "tempo_points": self.tempo_points,
            "meter_points": self.meter_points,
            "semantic_losses": list(self.losses),
            "lossless_roundtrip_claimed": False,
            "model_inference": False,
        }


def fingerprint(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def xml_tag(parent: ET.Element, tag: str, value: object = None, **attrs) -> ET.Element:
    el = ET.SubElement(parent, tag, {key: str(value) for key, value in attrs.items()})
    if value is not None:
        el.text = str(value)
    return el
