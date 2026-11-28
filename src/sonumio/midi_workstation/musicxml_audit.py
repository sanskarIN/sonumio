"""Read-only preflight of MusicXML expressive/engraving compatibility.

Report unsupported notation *before* attempting conversion. This is an
inspection-oriented API, not a repair engine: it never infers missing musical
information, drops unsupported features, mutates XML or executes embedded
resources. It delegates definitive importability to the strict import codec.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import xml.etree.ElementTree as ET

from .contracts import MidiFormatError, MidiLimitError, MidiWorkstationError
from .musicxml_common import (
    MUSICXML_PROTOCOL, MAX_MUSICXML_BYTES, local_name, parse_musicxml,
)
from .musicxml_import import decode_musicxml

AUDIT_PROTOCOL = "sonumio-musicxml-compatibility-audit-v1"
MAX_FINDINGS = 128
MAX_AUDIT_PARTS = 48
MAX_AUDIT_MEASURES = 512
MAX_AUDIT_NOTES = 96_000

# These elements represent engraving or performance semantics that cannot
# safely be imported into the present fixed-pitch, fixed-duration score model.
NOTE_UNSUPPORTED = frozenset({
    "unpitched", "cue", "grace", "time-modification", "play", "lyric",
    "instrument", "listen", "dynamics", "time-only", "harmony", "glissando",
})
MEASURE_UNSUPPORTED = frozenset({
    "barline", "harmony", "figured-bass", "grouping", "print", "bookmark",
    "link", "direction", "sound", "notehead", "transpose",
})
KNOWN_NOTATION_DROPS = frozenset({
    "accidental", "type", "dot", "stem", "beam", "clef", "key",
    "default-x", "default-y", "color", "font-size", "placement",
})


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    severity: str
    part: int
    measure: int
    element: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AuditResult:
    source_sha256: str
    source_bytes: int
    compatible: bool
    parts: int
    measures: int
    note_elements: int
    findings: tuple[AuditFinding, ...]
    semantic_losses: tuple[str, ...]
    score_sha256: str | None
    protocol: str = AUDIT_PROTOCOL

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "compatible": self.compatible,
            "parts": self.parts,
            "measures": self.measures,
            "note_elements": self.note_elements,
            "findings": [finding.as_dict() for finding in self.findings],
            "semantic_losses": list(self.semantic_losses),
            "score_sha256": self.score_sha256,
            "source_modified": False,
            "trained_model_inference": False,
        }


def _append(findings: list[AuditFinding], code: str, severity: str,
            part: int, measure: int, element: str, message: str) -> None:
    if len(findings) >= MAX_FINDINGS:
        return
    findings.append(AuditFinding(code,severity,part,measure,element[:80],message[:256]))


def audit_musicxml(raw: bytes) -> AuditResult:
    """Scan input with the exact same bounded XML parser as the importer."""
    root = parse_musicxml(raw)
    findings: list[AuditFinding] = []
    parts = [node for node in root if local_name(node.tag) == "part"]
    note_count = 0
    measures = 0
    if len(parts) > MAX_AUDIT_PARTS:
        raise MidiLimitError("MusicXML audit part budget exceeded")
    for part_index, part in enumerate(parts):
        local_measures = [node for node in part if local_name(node.tag) == "measure"]
        if len(local_measures) > MAX_AUDIT_MEASURES:
            raise MidiLimitError("MusicXML audit measure budget exceeded")
        measures += len(local_measures)
        for measure_index, measure in enumerate(local_measures):
            for element in measure:
                tag = local_name(element.tag)
                if tag not in {"attributes", "direction", "note", "backup", "forward"}:
                    _append(findings,"unsupported-measure-element", "error",
                            part_index+1, measure_index+1, tag,
                            f"The measure feature {tag} has no supported timeline interpretation")
                if tag == "direction":
                    sounds = [c for c in element if local_name(c.tag) == "sound"]
                    if not sounds or any(set(c.attrib)!={"tempo"} for c in sounds):
                        _append(findings,"unsupported-direction", "error",
                                part_index+1, measure_index+1,tag,
                                "Only an explicit sound tempo direction is supported")
                if tag == "note":
                    note_count += 1
                    if note_count > MAX_AUDIT_NOTES:
                        raise MidiLimitError("MusicXML audit note count exceeds limit")
                    for item in element:
                        kind = local_name(item.tag)
                        if kind in NOTE_UNSUPPORTED:
                            _append(findings,"unsupported-note-feature", "error",
                                    part_index+1, measure_index+1,kind,
                                    f"The note feature {kind} would lose expressive meaning")
                        if kind == "notations":
                            for notation in item:
                                kind = local_name(notation.tag)
                                if kind != "tied":
                                    _append(findings,"unsupported-notation", "error",
                                            part_index+1, measure_index+1,kind,
                                            f"The notation feature {kind} is not interpreted")
                if tag == "attributes":
                    for item in element:
                        kind = local_name(item.tag)
                        if kind == "transpose":
                            _append(findings,"transposed-instrument", "error",
                                    part_index+1,measure_index+1,kind,
                                    "Instrument transposition cannot be inferred from display pitch")
    try:
        score, receipt = decode_musicxml(raw)
    except MidiWorkstationError as exc:
        _append(findings,"import-rejected", "error",0,0,"score-partwise",str(exc))
        return AuditResult(sha256(raw).hexdigest(),len(raw),False,len(parts),
                           measures,note_count,tuple(findings),(),None)
    return AuditResult(sha256(raw).hexdigest(),len(raw),True,len(parts),
                       measures,note_count,tuple(findings),receipt.losses,
                       score.sha256)
