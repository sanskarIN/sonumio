"""Immutable, bounded undo/redo score history with digest-verified snapshots.

History preserves canonical Score snapshots in memory for deterministic
inspection and reversible editing. A hash chain detects accidental or
malicious serialization changes, but is not an authentication signature.
A newly committed edit invalidates the future redo branch explicitly.
There is no filesystem, GitHub or network side effect in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256 as _sha256

from .contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
    assert_keys, canonical_json, integer, parse_json, sha256,
)
from .score import Score
from .score_io import encode_score, decode_score

JOURNAL_PROTOCOL = "sonumio-workstation-revision-journal-v1"
MAX_HISTORY_STEPS = 128
MAX_HISTORY_BYTES = 4 * 1024 * 1024
MAX_JOURNAL_LABEL = 120
GENESIS_DIGEST = "0" * 64


@dataclass(frozen=True, slots=True)
class JournalSnapshot:
    score: Score
    label: str
    previous_digest: str
    entry_digest: str

    def __post_init__(self):
        if not isinstance(self.score, Score):
            raise MidiWorkstationError("Journal entry must contain typed Score")
        if (type(self.label) is not str or not 1 <= len(self.label) <= MAX_JOURNAL_LABEL
                or any(ord(x) < 32 for x in self.label)):
            raise MidiWorkstationError("Journal entry needs short printable description")
        sha256(self.previous_digest, "previous journal entry digest")
        sha256(self.entry_digest, "journal entry digest")
        expected = _entry_digest(self.score, self.label, self.previous_digest)
        if expected != self.entry_digest:
            raise MidiIntegrityError("Journal snapshot hash chain mismatch")

    @property
    def score_digest(self):
        return self.score.sha256

    @property
    def canonical_bytes(self):
        return len(encode_score(self.score))

    def as_dict(self):
        return {
            "label": self.label,
            "previous_digest": self.previous_digest,
            "entry_digest": self.entry_digest,
            "score_sha256": self.score_digest,
            "score_revision": self.score.revision,
            "note_count": self.score.note_count,
            "canonical_bytes": self.canonical_bytes,
        }


def _entry_digest(score, label, previous):
    payload = canonical_json({
        "score_sha256": score.sha256,
        "label": label,
        "previous": previous,
    })
    return _sha256(payload).hexdigest()


def _entry(score, label, previous):
    return JournalSnapshot(score, label, previous,
                           _entry_digest(score, label, previous))


@dataclass(frozen=True, slots=True)
class RevisionJournal:
    """Linear snapshot journal with optimistic digest leases and redo cursor."""

    entries: tuple[JournalSnapshot, ...]
    cursor: int
    protocol: str = JOURNAL_PROTOCOL

    def __post_init__(self):
        if self.protocol != JOURNAL_PROTOCOL:
            raise MidiFormatError("Unknown journal protocol")
        if (type(self.entries) is not tuple
                or not 1 <= len(self.entries) <= MAX_HISTORY_STEPS
                or any(not isinstance(e, JournalSnapshot) for e in self.entries)):
            raise MidiWorkstationError("Journal needs 1..128 valid snapshots")
        integer(self.cursor, "journal cursor", 0, len(self.entries)-1)
        if self.entries[0].previous_digest != GENESIS_DIGEST:
            raise MidiIntegrityError("Journal chain does not start at genesis")
        previous = GENESIS_DIGEST
        for index, item in enumerate(self.entries):
            if item.previous_digest != previous:
                raise MidiIntegrityError("Journal predecessor digest differs")
            if index and item.score.revision <= self.entries[index-1].score.revision:
                raise MidiIntegrityError("Journal score revisions must increase")
            previous = item.entry_digest
        if sum(item.canonical_bytes for item in self.entries) > MAX_HISTORY_BYTES:
            raise MidiLimitError("Journal exceeds snapshot memory limit")

    @classmethod
    def begin(cls, score: Score, *, label: str = "Initial score"):
        if not isinstance(score, Score):
            raise MidiWorkstationError("Cannot begin history without typed Score")
        return cls((_entry(score, label, GENESIS_DIGEST),), 0)

    @property
    def current(self):
        return self.entries[self.cursor].score

    @property
    def current_digest(self):
        return self.current.sha256

    @property
    def can_undo(self):
        return self.cursor > 0

    @property
    def can_redo(self):
        return self.cursor + 1 < len(self.entries)

    @property
    def total_canonical_bytes(self):
        return sum(item.canonical_bytes for item in self.entries)

    def commit(self, next_score: Score, label: str, *,
               expected_current_sha256: str):
        if not isinstance(next_score, Score):
            raise MidiWorkstationError("Journal commit requires typed Score")
        sha256(expected_current_sha256)
        if expected_current_sha256 != self.current_digest:
            raise MidiIntegrityError("Journal score changed since edit started")
        if next_score.revision <= self.current.revision:
            raise MidiWorkstationError("Journal commit must advance score revision")
        if next_score.sha256 == self.current_digest:
            raise MidiWorkstationError("Cannot append unchanged score snapshot")
        prefix = self.entries[:self.cursor+1]
        if len(prefix) >= MAX_HISTORY_STEPS:
            raise MidiLimitError("Journal has reached maximum undo depth")
        appended = _entry(next_score, label, prefix[-1].entry_digest)
        return RevisionJournal(prefix+(appended,), len(prefix))

    def undo(self, *, expected_current_sha256: str | None = None):
        if not self.can_undo:
            raise MidiWorkstationError("No earlier journal revision")
        if expected_current_sha256 is not None:
            sha256(expected_current_sha256)
            if expected_current_sha256 != self.current_digest:
                raise MidiIntegrityError("Undo cursor is stale")
        return replace(self, cursor=self.cursor-1)

    def redo(self, *, expected_current_sha256: str | None = None):
        if not self.can_redo:
            raise MidiWorkstationError("No redo revision in journal")
        if expected_current_sha256 is not None:
            sha256(expected_current_sha256)
            if expected_current_sha256 != self.current_digest:
                raise MidiIntegrityError("Redo cursor is stale")
        return replace(self, cursor=self.cursor+1)

    def seek(self, index: int):
        integer(index, "journal revision index", 0, len(self.entries)-1)
        return replace(self, cursor=index)

    def summary(self):
        return {
            "protocol": self.protocol,
            "cursor": self.cursor,
            "history_entries": len(self.entries),
            "can_undo": self.can_undo,
            "can_redo": self.can_redo,
            "current_score_sha256": self.current_digest,
            "current_entry_sha256": self.entries[self.cursor].entry_digest,
            "total_canonical_bytes": self.total_canonical_bytes,
            "entries": [item.as_dict() for item in self.entries],
            "remote_writes": False,
            "model_inference": False,
        }

    def to_bytes(self):
        data = {
            "protocol": self.protocol,
            "cursor": self.cursor,
            "entries": [{
                "label": item.label,
                "previous_digest": item.previous_digest,
                "entry_digest": item.entry_digest,
                "score_sha256": item.score.sha256,
                "score": item.score.to_json_obj(),
            } for item in self.entries],
        }
        return canonical_json(data, limit=MAX_HISTORY_BYTES)

    @classmethod
    def from_bytes(cls, raw: bytes):
        document = assert_keys(parse_json(raw, limit=MAX_HISTORY_BYTES),
                               {"protocol", "cursor", "entries"},
                               context="revision journal")
        if document["protocol"] != JOURNAL_PROTOCOL:
            raise MidiFormatError("Unknown revision journal schema")
        items = document["entries"]
        if type(items) is not list or not 1 <= len(items) <= MAX_HISTORY_STEPS:
            raise MidiFormatError("Journal entries must be a bounded JSON list")
        entries = []
        for item in items:
            fields = assert_keys(item, {
                "label", "previous_digest", "entry_digest", "score_sha256", "score",
            }, context="journal entry")
            score = decode_score(canonical_json(fields["score"]))
            if score.sha256 != fields["score_sha256"]:
                raise MidiIntegrityError("Journal score digest mismatch")
            entries.append(_entry(score, fields["label"], fields["previous_digest"]))
            if entries[-1].entry_digest != fields["entry_digest"]:
                raise MidiIntegrityError("Journal entry digest mismatch")
        return cls(tuple(entries), document["cursor"])
