"""Local-only MusicXML / MXL command-line acceptance and publication safety."""
from hashlib import sha256
import json
from pathlib import Path
import os
import sys

import pytest

from sonumio.midi_workstation.musicxml_cli import main
from sonumio.midi_workstation.musicxml_import import decode_musicxml
from sonumio.midi_workstation.musicxml_export import encode_musicxml
from sonumio.midi_workstation.score import Score, ScoreTrack, ScoreNote, note_sort_key
from sonumio.midi_workstation.score_io import encode_score, decode_score
from sonumio.midi_workstation.mxl import decode_mxl


def _source(tmp_path):
    result = tmp_path / "original.json"
    result.write_bytes(encode_score(Score.empty()))
    return result


def _receipt(capsys):
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.strip().startswith("{")
    return json.loads(captured.out)


def _convert(tmp_path, capsys, operation):
    source = _source(tmp_path)
    suffix = ".mxl" if operation == "to-mxl" else ".musicxml"
    target = tmp_path / ("projection" + suffix)
    assert main([operation,str(source),"--output",str(target)]) == 0
    return source,target,_receipt(capsys)


def test_help_returns_success(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "MusicXML" in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["to-xml", "to-mxl"])
def test_export_writes_checked_file_with_correct_extension(tmp_path,capsys,operation):
    source,target,receipt = _convert(tmp_path,capsys,operation)
    raw = target.read_bytes()
    assert raw
    assert receipt["publication"]["output_sha256"] == sha256(raw).hexdigest()
    assert receipt["source"]["source_sha256"] == sha256(source.read_bytes()).hexdigest()
    assert receipt["musicxml"]["notes"] == 0
    if operation == "to-mxl":
        assert decode_mxl(raw)[0].startswith(b"<?xml ")
    else:
        assert raw.startswith(b"<?xml ")


def test_plain_xml_export_deterministic_after_repeating(tmp_path,capsys):
    source,target,one = _convert(tmp_path,capsys,"to-xml")
    raw = target.read_bytes()
    destination = tmp_path / "other.musicxml"
    assert main(["to-xml",str(source),"--output",str(destination)]) == 0
    _receipt(capsys)
    assert destination.read_bytes() == raw


def test_mxl_export_deterministic_after_repeating(tmp_path,capsys):
    source,target,one = _convert(tmp_path,capsys,"to-mxl")
    raw = target.read_bytes()
    destination = tmp_path / "other.mxl"
    assert main(["to-mxl",str(source),"--output",str(destination)]) == 0
    _receipt(capsys)
    assert destination.read_bytes() == raw


def test_import_plain_musicxml_creates_canonical_score(tmp_path,capsys):
    source,target,_ = _convert(tmp_path,capsys,"to-xml")
    result = tmp_path / "imported.json"
    assert main(["from-xml",str(target),"--output",str(result)]) == 0
    receipt = _receipt(capsys)
    imported = decode_score(result.read_bytes())
    assert imported.title == "Original composition"
    assert imported.note_count == 0
    assert receipt["score_sha256"] == imported.sha256
    assert receipt["conversion"]["lossless_roundtrip_claimed"] is False


def test_import_mxl_creates_canonical_score(tmp_path,capsys):
    source,target,_ = _convert(tmp_path,capsys,"to-mxl")
    result = tmp_path / "imported.json"
    assert main(["from-mxl",str(target),"--output",str(result)]) == 0
    receipt = _receipt(capsys)
    imported = decode_score(result.read_bytes())
    assert imported.note_count == 0
    assert receipt["score_sha256"] == imported.sha256


@pytest.mark.parametrize("operation", ["to-xml", "to-mxl"])
def test_missing_expected_input_sha_does_not_publish(tmp_path,capsys,operation):
    source = _source(tmp_path)
    output = tmp_path / ("result" + (".mxl" if operation=="to-mxl" else ".musicxml"))
    args = [operation,str(source),"--output",str(output),"--expected-sha256","0"*64]
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SHA-256" in captured.err
    assert not output.exists()


@pytest.mark.parametrize("operation", ["to-xml", "to-mxl"])
def test_mismatched_output_digest_does_not_publish(tmp_path,capsys,operation):
    source = _source(tmp_path)
    output = tmp_path / ("result" + (".mxl" if operation=="to-mxl" else ".musicxml"))
    args = [operation,str(source),"--output",str(output),"--expected-output-sha256","0"*64]
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not output.exists()


@pytest.mark.parametrize("operation", ["to-xml", "to-mxl"])
def test_wrong_extension_is_rejected(tmp_path,capsys,operation):
    source = _source(tmp_path)
    output = tmp_path / "result.bin"
    assert main([operation,str(source),"--output",str(output)]) == 2
    assert not output.exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("operation", ["to-xml", "to-mxl"])
def test_no_clobber_default_preserves_existing_bytes(tmp_path,capsys,operation):
    source,target,_ = _convert(tmp_path,capsys,operation)
    previous = target.read_bytes()
    assert main([operation,str(source),"--output",str(target)]) == 2
    assert target.read_bytes() == previous
    assert capsys.readouterr().out == ""


def test_explicit_overwrite_supports_current_existing_file(tmp_path,capsys):
    source,target,_ = _convert(tmp_path,capsys,"to-xml")
    before = target.read_bytes()
    assert main(["to-xml",str(source),"--output",str(target),"--overwrite"]) == 0
    _receipt(capsys)
    assert target.read_bytes() == before


@pytest.mark.parametrize("operation", ["to-xml", "to-mxl"])
def test_symlink_source_refused(tmp_path,capsys,operation):
    source = _source(tmp_path)
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(source)
    except OSError:
        pytest.skip("Host disallows symlinks")
    output = tmp_path / ("new" + (".mxl" if operation=="to-mxl" else ".musicxml"))
    assert main([operation,str(alias),"--output",str(output)]) == 2
    assert not output.exists()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("operation", ["to-xml", "to-mxl"])
def test_symlink_destination_refused(tmp_path,capsys,operation):
    source = _source(tmp_path)
    victim = tmp_path / "victim"
    victim.write_text("valuable")
    output = tmp_path / ("evil" + (".mxl" if operation=="to-mxl" else ".musicxml"))
    try:
        output.symlink_to(victim)
    except OSError:
        pytest.skip("Host disallows symlinks")
    assert main([operation,str(source),"--output",str(output),"--overwrite"]) == 2
    assert victim.read_text() == "valuable"
    assert capsys.readouterr().out == ""


def test_inspect_plain_xml_is_read_only(tmp_path,capsys):
    _,source,_ = _convert(tmp_path,capsys,"to-xml")
    before = source.read_bytes()
    assert main(["inspect",str(source)]) == 0
    receipt = _receipt(capsys)
    assert receipt["source_unchanged"] is True
    assert receipt["source_sha256"] == sha256(before).hexdigest()
    assert source.read_bytes() == before


def test_inspect_compressed_package_reports_both_digests(tmp_path,capsys):
    _,source,_ = _convert(tmp_path,capsys,"to-mxl")
    assert main(["inspect",str(source)]) == 0
    receipt = _receipt(capsys)
    assert receipt["package"]["compressed"] is True
    assert receipt["package"]["archive_sha256"] == sha256(source.read_bytes()).hexdigest()


def test_truncated_xml_does_not_publish_score(tmp_path,capsys):
    bad = tmp_path / "bad.musicxml"
    bad.write_bytes(b"<score-partwise"*12)
    target = tmp_path / "not-created.json"
    assert main(["from-xml",str(bad),"--output",str(target)]) == 2
    assert not target.exists()
    assert capsys.readouterr().out == ""


def test_truncated_zip_does_not_publish_score(tmp_path,capsys):
    bad = tmp_path / "bad.mxl"
    bad.write_bytes(b"PK\x03\x04"*200)
    target = tmp_path / "not-created.json"
    assert main(["from-mxl",str(bad),"--output",str(target)]) == 2
    assert not target.exists()
    assert capsys.readouterr().out == ""


def test_missing_parent_refused_by_default(tmp_path,capsys):
    source = _source(tmp_path)
    output = tmp_path / "uncreated" / "result.musicxml"
    assert main(["to-xml",str(source),"--output",str(output)]) == 2
    assert not output.exists()
    assert capsys.readouterr().out == ""


def test_create_parent_requires_explicit_flag(tmp_path,capsys):
    source = _source(tmp_path)
    output = tmp_path / "created" / "result.musicxml"
    assert main(["to-xml",str(source),"--output",str(output),"--create-parent"]) == 0
    _receipt(capsys)
    assert output.is_file()


def test_pinned_output_matches_generated_digest(tmp_path,capsys):
    source = _source(tmp_path)
    raw = encode_musicxml(Score.empty())[0]
    output = tmp_path / "pinned.musicxml"
    assert main(["to-xml",str(source),"--output",str(output),
                 "--expected-output-sha256",sha256(raw).hexdigest()]) == 0
    _receipt(capsys)
    assert output.read_bytes() == raw


def test_rejects_attempted_xml_in_place_replace(tmp_path,capsys):
    _,source,_ = _convert(tmp_path,capsys,"to-xml")
    before = source.read_bytes()
    assert main(["from-xml",str(source),"--output",str(source),"--overwrite"]) == 2
    assert source.read_bytes() == before
    assert capsys.readouterr().out == ""


def test_same_file_hardlink_output_is_refused(tmp_path,capsys):
    source = _source(tmp_path)
    alias = tmp_path / "alias.json"
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("Hardlinks unavailable")
    target = tmp_path / "new.musicxml"
    assert main(["to-xml",str(alias),"--output",str(target)]) == 0
    _receipt(capsys)
    assert source.read_bytes() == alias.read_bytes()


@pytest.mark.parametrize("command", ["to-xml", "to-mxl", "from-xml", "from-mxl", "inspect"])
def test_nonexistent_source_is_not_success(command,tmp_path,capsys):
    missing=tmp_path/"missing.json"
    argv=[command,str(missing)]
    if command!="inspect":
        argv += ["--output",str(tmp_path/"output.json")]
    assert main(argv) == 2
    assert capsys.readouterr().out == ""
