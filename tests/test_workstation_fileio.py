"""Canonical score snapshots, tamper detection and local atomic publication tests."""
from dataclasses import replace
import hashlib
import json
import os

import pytest

from sonumio.midi_workstation.contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.publish import (
    PublicationOptions, publish_bytes, publish_midi, publish_score,
)
from sonumio.midi_workstation.score import Score, ScoreTrack, ScoreNote, to_midi
from sonumio.midi_workstation.score_io import (
    decode_score, encode_score, load_score, parse_score_snapshot,
    canonical_score_receipt,
)
from sonumio.midi_workstation.smf import read_smf


def example():
    original = Score.empty(ppqn=480, end=3840, title="Original composition")
    track = original.tracks[0]
    notes = (ScoreNote("note1", 120, 160, 60),
             ScoreNote("note2", 600, 240, 64),
             ScoreNote("note3", 1200, 360, 67))
    return original.with_tracks((track.replace_notes(notes),))


def test_canonical_roundtrip_preserves_full_typed_score():
    score = example()
    raw = encode_score(score)
    assert decode_score(raw) == score
    assert encode_score(decode_score(raw)) == raw
    assert hashlib.sha256(raw).hexdigest() == score.sha256
    assert canonical_score_receipt(score)["note_count"] == 3
    assert canonical_score_receipt(score)["external_assets_referenced"] is False


def test_whitespace_and_key_order_produce_identical_canonical_score_digest():
    original = example()
    payload = json.loads(encode_score(original))
    pretty = json.dumps(payload, indent=3).encode("utf-8")
    snapshot = parse_score_snapshot(pretty)
    assert snapshot.score == original
    assert snapshot.source_sha256 != snapshot.canonical_sha256
    assert snapshot.canonical_sha256 == original.sha256
    assert snapshot.source_bytes > snapshot.canonical_bytes
    assert snapshot.as_dict()["model_inference"] is False


def test_source_file_snapshot_is_bound_to_optional_digest(tmp_path):
    score = example()
    path = tmp_path / "project.json"
    source = encode_score(score)
    path.write_bytes(source)
    sha = hashlib.sha256(source).hexdigest()
    snapshot = load_score(path, expected_sha256=sha)
    assert snapshot.score == score
    assert snapshot.source_sha256 == sha
    with pytest.raises(MidiIntegrityError):
        load_score(path, expected_sha256="0" * 64)


def test_unknown_score_top_level_field_rejected():
    fields = json.loads(encode_score(example()))
    fields["external_script"] = "run-something"
    with pytest.raises(MidiFormatError):
        decode_score(json.dumps(fields).encode("utf-8"))


def test_unknown_track_or_note_field_rejected():
    original = json.loads(encode_score(example()))
    for target in ("tracks", "notes"):
        document = json.loads(json.dumps(original))
        if target == "tracks":
            document["tracks"][0]["filesystem_path"] = "forbidden"
        else:
            document["tracks"][0]["notes"][0]["eval"] = "forbidden"
        with pytest.raises(MidiFormatError):
            decode_score(json.dumps(document).encode())


def test_invalid_score_tempo_or_meter_arrays_rejected():
    original = json.loads(encode_score(example()))
    for key in ("tempo", "meter", "tracks"):
        document = dict(original)
        document[key] = {"not": "a list"}
        with pytest.raises(MidiFormatError):
            decode_score(json.dumps(document).encode())


def test_duplicate_key_and_nonfinite_json_rejected():
    source = encode_score(example())
    with pytest.raises(MidiFormatError):
        decode_score(b'{"title":"first",' + source[1:])
    with pytest.raises(MidiFormatError):
        decode_score(source.replace(b'"velocity":100', b'"velocity":NaN'))


def test_invalid_unicode_and_truncated_json_rejected():
    for bad in (b"", b"\xff", b"{", b"[]", b"null"):
        with pytest.raises(MidiWorkstationError):
            decode_score(bad)


def test_missing_or_symlinked_source_rejected(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(MidiWorkstationError):
        load_score(missing)
    actual = tmp_path / "source.json"
    actual.write_bytes(encode_score(example()))
    symlink = tmp_path / "symbolic.json"
    try:
        symlink.symlink_to(actual)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")
    with pytest.raises(MidiWorkstationError):
        load_score(symlink)


@pytest.mark.parametrize("payload", [b"payload1", b"\x00\xff\x10\x00", b"new bytes"])
def test_atomic_publication_receipt_matches_actual_bytes(tmp_path, payload):
    destination = tmp_path / "published.bin"
    receipt = publish_bytes(destination, payload)
    assert receipt.output_sha256 == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload
    assert receipt.bytes_written == len(payload)
    assert not receipt.overwrote
    assert receipt.as_dict()["remote_publication"] is False
    assert not list(tmp_path.glob("*.tmp"))


def test_no_clobber_is_default(tmp_path):
    destination = tmp_path / "score.bin"
    destination.write_bytes(b"old")
    with pytest.raises(MidiWorkstationError):
        publish_bytes(destination, b"new")
    assert destination.read_bytes() == b"old"


def test_overwrite_requires_exact_expected_previous_digest_when_pinned(tmp_path):
    target = tmp_path / "score.bin"
    target.write_bytes(b"original")
    previous = hashlib.sha256(b"original").hexdigest()
    options = PublicationOptions(overwrite=True,
                                 expected_previous_sha256=previous)
    publication = publish_bytes(target, b"updated", options)
    assert publication.previous_sha256 == previous
    assert publication.overwrote
    assert target.read_bytes() == b"updated"
    with pytest.raises(MidiIntegrityError):
        publish_bytes(target, b"changed", options)
    assert target.read_bytes() == b"updated"


def test_wrong_generated_output_hash_never_writes(tmp_path):
    destination = tmp_path / "output.bin"
    with pytest.raises(MidiIntegrityError):
        publish_bytes(destination, b"new", PublicationOptions(
            expected_sha256="0" * 64,
        ))
    assert not destination.exists()


def test_path_parent_requires_explicit_creation(tmp_path):
    output = tmp_path / "folder" / "out.bin"
    with pytest.raises(MidiWorkstationError):
        publish_bytes(output, b"data")
    assert not output.parent.exists()
    publish_bytes(output, b"data", PublicationOptions(create_parent=True))
    assert output.read_bytes() == b"data"


def test_output_symlink_never_followed(tmp_path):
    original = tmp_path / "original.bin"
    original.write_bytes(b"protected")
    alias = tmp_path / "alias.bin"
    try:
        alias.symlink_to(original)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks not supported")
    with pytest.raises(MidiWorkstationError):
        publish_bytes(alias, b"bad", PublicationOptions(overwrite=True))
    assert original.read_bytes() == b"protected"


def test_hardlinked_alias_rejected_before_overwrite(tmp_path):
    original = tmp_path / "input.bin"
    original.write_bytes(b"keep this")
    destination = tmp_path / "alias.bin"
    try:
        os.link(original, destination)
    except OSError:
        pytest.skip("Hardlink unavailable")
    with pytest.raises(MidiFormatError):
        publish_bytes(destination, b"new", PublicationOptions(overwrite=True),
                      input_path=original)
    assert original.read_bytes() == b"keep this"


def test_direct_source_output_identity_rejected(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(b"old")
    with pytest.raises(MidiFormatError):
        publish_bytes(source, b"new", PublicationOptions(overwrite=True),
                      input_path=source)
    assert source.read_bytes() == b"old"


def test_empty_publication_or_invalid_extension_fails_closed(tmp_path):
    destination = tmp_path / "file.json"
    with pytest.raises(MidiWorkstationError):
        publish_bytes(destination, b"")
    with pytest.raises(MidiFormatError):
        publish_bytes(destination, b"bytes", expected_extension=".mid")
    assert not destination.exists()


def test_validation_for_overwrite_flags_and_old_digest():
    for kwargs in (
        {"overwrite": 1},
        {"create_parent": 0},
        {"expected_sha256": "not-a-digest"},
        {"expected_previous_sha256": "0" * 64},
    ):
        with pytest.raises(MidiWorkstationError):
            PublicationOptions(**kwargs)


def test_publish_midi_roundtrip_verified_reader(tmp_path):
    score = example()
    destination = tmp_path / "export.mid"
    receipt = publish_midi(destination, to_midi(score))
    snapshot = read_smf(destination, expected_sha256=receipt.output_sha256)
    assert snapshot.sha256 == receipt.output_sha256
    assert snapshot.midi.end == score.end
    assert len(snapshot.midi.tracks) >= 2


def test_published_score_can_be_loaded_and_reexported(tmp_path):
    score = example()
    destination = tmp_path / "project.json"
    publication = publish_score(destination, score)
    loaded = load_score(destination, expected_sha256=publication.output_sha256)
    assert loaded.score == score
    assert loaded.canonical_sha256 == publication.output_sha256
    next_path = tmp_path / "project-replayed.json"
    second = publish_score(next_path, loaded.score)
    assert second.output_sha256 == publication.output_sha256
    assert next_path.read_bytes() == destination.read_bytes()
