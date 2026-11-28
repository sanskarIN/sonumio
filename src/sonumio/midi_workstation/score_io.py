"""Strict, canonical and digest-verifiable JSON interchange for symbolic scores.

Score JSON contains only numerical and textual music data. Unknown keys,
duplicate keys, NaNs, path fields and arbitrary script payloads are rejected.
Local files are read through bounded regular descriptors from contracts.py.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256 as _sha256
from pathlib import Path

from .contracts import (
    MAX_JSON_BYTES, MidiFormatError, MidiIntegrityError,
    MidiWorkstationError, assert_keys, canonical_json, parse_json,
    read_regular_file, sha256,
)
from .score import (
    SCORE_PROTOCOL, Score, ScoreNote, ScoreTrack,
)
from .tempo import MeterPoint, TempoPoint

SCORE_JSON_MAX_BYTES = MAX_JSON_BYTES


def encode_score(score: Score) -> bytes:
    if not isinstance(score, Score):
        raise MidiWorkstationError("Expected typed Score")
    return canonical_json(score.to_json_obj(), limit=SCORE_JSON_MAX_BYTES)


def decode_score(data: bytes) -> Score:
    raw = parse_json(data, limit=SCORE_JSON_MAX_BYTES)
    fields = assert_keys(raw, {
        "protocol", "title", "ppqn", "end", "revision",
        "tempo", "meter", "tracks",
    }, context="workstation score")
    if fields["protocol"] != SCORE_PROTOCOL:
        raise MidiFormatError("Unsupported symbolic score version")
    if (type(fields["tempo"]) is not list or type(fields["meter"]) is not list
            or type(fields["tracks"]) is not list):
        raise MidiFormatError("Score timeline collections must be JSON arrays")
    tempo = []
    for obj in fields["tempo"]:
        fields_tempo = assert_keys(obj, {"tick", "microseconds_per_quarter"},
                                    context="score tempo point")
        tempo.append(TempoPoint(**fields_tempo))
    meter = []
    for obj in fields["meter"]:
        fields_meter = assert_keys(obj, {
            "tick", "numerator", "denominator", "clocks_per_metronome",
            "thirtyseconds_per_quarter",
        }, context="score meter point")
        meter.append(MeterPoint(**fields_meter))
    tracks = []
    for obj in fields["tracks"]:
        body = assert_keys(obj, {
            "id", "name", "channel", "program", "bank_msb", "bank_lsb",
            "notes", "muted", "solo",
        }, context="score track")
        if type(body["notes"]) is not list:
            raise MidiFormatError("Score track notes must be JSON array")
        notes = []
        for value in body["notes"]:
            note = assert_keys(value, {
                "id", "start", "duration", "pitch", "velocity",
                "release_velocity", "pan",
            }, context="score note")
            notes.append(ScoreNote(**note))
        track = ScoreTrack(
            body["id"], body["name"], body["channel"], body["program"],
            body["bank_msb"], body["bank_lsb"], tuple(notes),
            body["muted"], body["solo"],
        )
        tracks.append(track)
    return Score(
        fields["title"], fields["ppqn"], tuple(tracks),
        tuple(tempo), tuple(meter), fields["end"], fields["revision"],
    )


@dataclass(frozen=True, slots=True)
class ScoreSnapshot:
    score: Score
    source_sha256: str
    canonical_sha256: str
    source_bytes: int
    canonical_bytes: int

    def __post_init__(self):
        if not isinstance(self.score, Score):
            raise MidiWorkstationError("Score snapshot requires typed Score")
        sha256(self.source_sha256)
        sha256(self.canonical_sha256)
        if self.canonical_sha256 != self.score.sha256:
            raise MidiIntegrityError("Score snapshot differs from canonical score digest")

    def as_dict(self):
        return {
            "protocol": "sonumio-workstation-score-snapshot-v1",
            "source_sha256": self.source_sha256,
            "canonical_sha256": self.canonical_sha256,
            "source_bytes": self.source_bytes,
            "canonical_bytes": self.canonical_bytes,
            "score_revision": self.score.revision,
            "score_note_count": self.score.note_count,
            "source_unchanged": True,
            "model_inference": False,
        }


def parse_score_snapshot(raw: bytes) -> ScoreSnapshot:
    score = decode_score(raw)
    canonical = encode_score(score)
    return ScoreSnapshot(score, _sha256(raw).hexdigest(),
                         score.sha256, len(raw), len(canonical))


def load_score(path: str | Path, *, expected_sha256: str | None = None) -> ScoreSnapshot:
    data = read_regular_file(path, limit=SCORE_JSON_MAX_BYTES,
                             expected_sha256=expected_sha256)
    snapshot = parse_score_snapshot(data.content)
    if snapshot.source_sha256 != data.sha256:
        raise MidiIntegrityError("Score changed during read")
    return snapshot


def canonical_score_receipt(score: Score) -> dict:
    raw = encode_score(score)
    return {
        "protocol": "sonumio-workstation-canonical-score-v1",
        "score_sha256": _sha256(raw).hexdigest(),
        "canonical_bytes": len(raw),
        "source_revision": score.revision,
        "note_count": score.note_count,
        "external_assets_referenced": False,
        "models_loaded": False,
    }
