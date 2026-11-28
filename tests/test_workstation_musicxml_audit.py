"""Actionable, non-mutating MusicXML compatibility audit tests."""
from hashlib import sha256
import json
import xml.etree.ElementTree as ET

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError, MidiLimitError
from sonumio.midi_workstation.musicxml_audit import audit_musicxml
from sonumio.midi_workstation.musicxml_cli import main
from sonumio.midi_workstation.musicxml_export import encode_musicxml
from sonumio.midi_workstation.mxl import encode_mxl
from sonumio.midi_workstation.score import Score, ScoreNote


def valid():
    return encode_musicxml(Score.empty())[0]


def with_measure_feature(tag):
    root = ET.fromstring(valid())
    ET.SubElement(root.find("part").find("measure"), tag)
    return ET.tostring(root, encoding="utf-8")


def with_note_feature(tag):
    root = ET.fromstring(valid())
    node = ET.SubElement(root.find("part").find("measure"), "note")
    ET.SubElement(node,tag)
    return ET.tostring(root, encoding="utf-8")


def test_valid_score_audits_cleanly_without_silent_semantic_omissions():
    data=valid()
    result=audit_musicxml(data)
    assert result.compatible is True
    assert result.source_sha256 == sha256(data).hexdigest()
    assert result.parts == 1
    assert result.measures == 1
    assert result.note_elements == 0
    assert not result.findings
    assert result.semantic_losses
    assert result.as_dict()["source_modified"] is False


@pytest.mark.parametrize("tag", ["barline", "harmony", "figured-bass", "grouping",
                                      "print", "bookmark", "unknown-performance"])
def test_unsupported_measure_features_get_actionable_findings(tag):
    result=audit_musicxml(with_measure_feature(tag))
    assert not result.compatible
    assert any(f.element==tag and f.severity=="error" for f in result.findings)
    assert any(f.code=="import-rejected" for f in result.findings)


@pytest.mark.parametrize("tag", ["unpitched", "cue", "grace", "time-modification",
                                      "play", "lyric", "instrument", "listen"])
def test_unsupported_note_features_are_reported(tag):
    result=audit_musicxml(with_note_feature(tag))
    assert not result.compatible
    assert any(f.element==tag and f.code=="unsupported-note-feature"
               for f in result.findings)


def test_notation_ornament_is_flagged_separately_from_import_failure():
    root=ET.fromstring(valid())
    note=ET.SubElement(root.find("part").find("measure"),"note")
    notations=ET.SubElement(note,"notations")
    ET.SubElement(notations,"ornaments")
    result=audit_musicxml(ET.tostring(root,encoding="utf-8"))
    assert not result.compatible
    assert any(f.code=="unsupported-notation" for f in result.findings)


def test_audit_findings_are_bounded_to_128():
    root=ET.fromstring(valid())
    measure=root.find("part").find("measure")
    for _ in range(500):
        ET.SubElement(measure,"barline")
    result=audit_musicxml(ET.tostring(root,encoding="utf-8"))
    assert len(result.findings)==128
    assert not result.compatible


def test_malformed_xml_has_no_success_report():
    with pytest.raises(MidiFormatError):
        audit_musicxml(b"<score-partwise>"*4)


def test_audit_does_not_change_original_xml_input():
    raw=valid()
    before=sha256(raw).digest()
    audit_musicxml(raw)
    assert sha256(raw).digest()==before


def test_cli_audit_output_for_compatible_xml(tmp_path,capsys):
    src=tmp_path/"score.musicxml"
    src.write_bytes(valid())
    assert main(["audit",str(src)])==0
    output=json.loads(capsys.readouterr().out)
    assert output["audit"]["compatible"] is True
    assert output["source_sha256"]==sha256(src.read_bytes()).hexdigest()
    assert output["source_unchanged"] is True


def test_cli_audit_output_for_unimportable_xml_still_reports_without_writing(tmp_path,capsys):
    src=tmp_path/"bad.musicxml"
    src.write_bytes(with_measure_feature("barline"))
    assert main(["audit",str(src)])==0
    payload=json.loads(capsys.readouterr().out)
    assert payload["audit"]["compatible"] is False
    assert payload["audit"]["findings"]
    assert set(tmp_path.iterdir())=={src}


def test_cli_audit_compressed_xml(tmp_path,capsys):
    src=tmp_path/"packed.mxl"
    src.write_bytes(encode_mxl(valid())[0])
    assert main(["audit",str(src)])==0
    result=json.loads(capsys.readouterr().out)
    assert result["audit"]["compatible"] is True
    assert result["package"]["compressed"] is True


def test_cli_audit_rejects_bad_source_integrity(tmp_path,capsys):
    src=tmp_path/"score.musicxml"
    src.write_bytes(valid())
    assert main(["audit",str(src),"--expected-sha256","0"*64])==2
    assert capsys.readouterr().out==""


def test_audit_does_not_claim_generated_model():
    report=audit_musicxml(valid()).as_dict()
    assert report["trained_model_inference"] is False
    assert report["source_modified"] is False
