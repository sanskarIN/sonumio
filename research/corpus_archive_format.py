"""Strict binary and canonical JSON format for private reviewed-corpus archives.

The format contains actual local audio and private provenance. Never publish
these files merely because the Sonumio source code is Apache-2.0 licensed.
This module validates structure, not the underlying legal rights.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct

import numpy as np

from research.audio_dataset import ClipProvenance, caption_category
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, SAMPLE_RATE

PROTOCOL = "sonumio-private-corpus-archive-v1"
MAGIC = b"SCORPUS1"
FOOTER_MAGIC = b"SCIDX001"
FOOTER = struct.Struct("<Q8s")
METADATA_LENGTH = struct.Struct("<I")
SPLITS = ("train", "validation", "test")
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_INDEX_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_SHARD_CLIPS = 256
MAX_ROWS = 32_768
MAX_BATCH_CLIPS = 1024
PCM_BYTES_PER_CLIP = CLIP_FRAMES * 4
_HEX = frozenset("0123456789abcdef")


class ArchiveError(ValueError):
    """Invalid or unverifiable private corpus archive."""


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def require_sha256(value: object) -> str:
    if not is_sha256(value):
        raise ArchiveError("Expected lowercase SHA256")
    return value


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ArchiveError("Invalid canonical archive JSON") from exc


def parse_canonical(data: bytes) -> object:
    """Re-encoding rejects duplicate keys, noncanonical layout and nonfinite JSON."""
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ArchiveError("Invalid archive JSON") from exc
    if canonical_json(value) != data:
        raise ArchiveError("Archive JSON is not canonical")
    return value


def _integer(value: object, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ArchiveError(f"Invalid archive {name}")
    return value


def _keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ArchiveError(f"Invalid archive {name} fields")
    return value


def _text(value: object, limit: int, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ArchiveError(f"Invalid archive {name}")
    return value


@dataclass(frozen=True, slots=True)
class ArchiveRow:
    index: int
    label: int
    caption: str
    asset_id: str
    group_id: str
    language: str
    source_line: int
    crop_index: int

    @property
    def provenance(self) -> ClipProvenance:
        return ClipProvenance(self.asset_id, self.group_id, self.language,
                              self.source_line, self.crop_index)

    def as_dict(self) -> dict:
        return {
            "index": self.index, "label": self.label, "caption": self.caption,
            "asset_id": self.asset_id, "group_id": self.group_id,
            "language": self.language, "source_line": self.source_line,
            "crop_index": self.crop_index,
        }

    @classmethod
    def parse(cls, value: object, expected_index: int) -> "ArchiveRow":
        row = _keys(value, {"index", "label", "caption", "asset_id", "group_id",
                            "language", "source_line", "crop_index"}, "row")
        index = _integer(row["index"], 0, MAX_ROWS - 1, "row index")
        if index != expected_index:
            raise ArchiveError("Nonconsecutive archive row index")
        label = _integer(row["label"], 0, len(CATEGORIES) - 1, "label")
        caption = _text(row["caption"], 800, "caption")
        if caption_category(caption) != label:
            raise ArchiveError("Archive caption does not match label")
        asset = _text(row["asset_id"], 16_384, "asset ID")
        group = _text(row["group_id"], 16_384, "group ID")
        language = _text(row["language"], 2, "language")
        if language not in ("en", "hi"):
            raise ArchiveError("Invalid archive language")
        source_line = _integer(row["source_line"], 1, 32 * 1024 * 1024, "source line")
        crop = _integer(row["crop_index"], 1, 8, "crop index")
        return cls(index, label, caption, asset, group, language, source_line, crop)


@dataclass(frozen=True, slots=True)
class ShardDescriptor:
    split: str
    start: int
    count: int
    offset: int
    length: int
    sha256: str

    def as_dict(self) -> dict:
        return {
            "split": self.split, "start": self.start, "count": self.count,
            "offset": self.offset, "length": self.length, "sha256": self.sha256,
        }

    @classmethod
    def parse(cls, value: object) -> "ShardDescriptor":
        row = _keys(value, {"split", "start", "count", "offset", "length", "sha256"},
                    "shard descriptor")
        split = row["split"]
        if split not in SPLITS:
            raise ArchiveError("Invalid archive split")
        count = _integer(row["count"], 1, MAX_SHARD_CLIPS, "shard count")
        length = _integer(
            row["length"], METADATA_LENGTH.size + count * PCM_BYTES_PER_CLIP + 2,
            METADATA_LENGTH.size + count * PCM_BYTES_PER_CLIP + MAX_METADATA_BYTES,
            "shard length",
        )
        return cls(
            split, _integer(row["start"], 0, MAX_ROWS - 1, "shard start"),
            count, _integer(row["offset"], len(MAGIC), MAX_ARCHIVE_BYTES, "shard offset"),
            length, require_sha256(row["sha256"]),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ArchiveShard:
    descriptor: ShardDescriptor
    rows: tuple[ArchiveRow, ...]
    clips: np.ndarray


def encode_shard(rows: tuple[ArchiveRow, ...], clips: np.ndarray) -> bytes:
    if not 1 <= len(rows) <= MAX_SHARD_CLIPS:
        raise ArchiveError("Invalid shard row count")
    if (not isinstance(clips, np.ndarray) or clips.dtype != np.float32
            or clips.shape != (len(rows), 1, CLIP_FRAMES)
            or not np.isfinite(clips).all()):
        raise ArchiveError("Invalid shard PCM geometry")
    metadata = canonical_json([row.as_dict() for row in rows])
    if len(metadata) > MAX_METADATA_BYTES:
        raise ArchiveError("Shard metadata exceeds byte budget")
    pcm = np.asarray(clips, dtype="<f4", order="C").tobytes(order="C")
    return METADATA_LENGTH.pack(len(metadata)) + metadata + pcm


def decode_shard(descriptor: ShardDescriptor, payload: bytes) -> ArchiveShard:
    if len(payload) != descriptor.length or digest(payload) != descriptor.sha256:
        raise ArchiveError("Archive shard SHA256 or length mismatch")
    metadata_size, = METADATA_LENGTH.unpack_from(payload)
    if not 2 <= metadata_size <= MAX_METADATA_BYTES:
        raise ArchiveError("Invalid shard metadata length")
    start = METADATA_LENGTH.size
    stop = start + metadata_size
    if stop + descriptor.count * PCM_BYTES_PER_CLIP != len(payload):
        raise ArchiveError("Shard PCM byte geometry mismatch")
    value = parse_canonical(payload[start:stop])
    if not isinstance(value, list) or len(value) != descriptor.count:
        raise ArchiveError("Shard row count mismatch")
    rows = tuple(ArchiveRow.parse(row, descriptor.start + i)
                 for i, row in enumerate(value))
    clips = np.frombuffer(payload, dtype="<f4", offset=stop).reshape(
        descriptor.count, 1, CLIP_FRAMES,
    )
    # Native float32 on big-endian hosts requires an immutable converted
    # backing. On ordinary little-endian hosts the original bytes suffice.
    if clips.dtype != np.float32:
        clips = np.frombuffer(clips.astype(np.float32).tobytes(),
                              dtype=np.float32).reshape(clips.shape)
    if not np.isfinite(clips).all():
        raise ArchiveError("Nonfinite archived PCM")
    return ArchiveShard(descriptor, rows, clips)


@dataclass(frozen=True, slots=True)
class ArchiveIndex:
    manifest_sha256: str
    counts: tuple[int, int, int]
    fingerprints: tuple[str, str, str]
    shards: tuple[ShardDescriptor, ...]

    def as_dict(self) -> dict:
        return {
            "protocol": PROTOCOL, "sample_rate": SAMPLE_RATE, "clip_frames": CLIP_FRAMES,
            "manifest_sha256": self.manifest_sha256,
            "split_counts": dict(zip(SPLITS, self.counts)),
            "split_fingerprints": dict(zip(SPLITS, self.fingerprints)),
            "shards": [item.as_dict() for item in self.shards],
        }

    @classmethod
    def parse(cls, value: object, index_offset: int) -> "ArchiveIndex":
        row = _keys(value, {"protocol", "sample_rate", "clip_frames", "manifest_sha256",
                            "split_counts", "split_fingerprints", "shards"}, "index")
        if (row["protocol"] != PROTOCOL or type(row["sample_rate"]) is not int
                or row["sample_rate"] != SAMPLE_RATE
                or type(row["clip_frames"]) is not int or row["clip_frames"] != CLIP_FRAMES):
            raise ArchiveError("Unsupported archive protocol or sample geometry")
        counts_obj = _keys(row["split_counts"], set(SPLITS), "split counts")
        counts = tuple(_integer(counts_obj[s], 0, MAX_ROWS, "split count") for s in SPLITS)
        if not counts[0] or not counts[1] or sum(counts) > MAX_ROWS:
            raise ArchiveError("Invalid total archive crop count")
        fp = _keys(row["split_fingerprints"], set(SPLITS), "split fingerprints")
        fingerprints = tuple(require_sha256(fp[s]) for s in SPLITS)
        if not isinstance(row["shards"], list) or not 1 <= len(row["shards"]) <= MAX_ROWS:
            raise ArchiveError("Invalid archive shard count")
        shards = tuple(ShardDescriptor.parse(item) for item in row["shards"])
        position = len(MAGIC)
        seen = {s: 0 for s in SPLITS}
        previous_split = 0
        for item in shards:
            rank = SPLITS.index(item.split)
            if (rank < previous_split or item.offset != position
                    or item.start != seen[item.split]):
                raise ArchiveError("Archive shards overlap, contain gaps or are reordered")
            previous_split = rank
            seen[item.split] += item.count
            if seen[item.split] > counts[rank]:
                raise ArchiveError("Archive shard exceeds declared split count")
            position += item.length
        if position != index_offset or tuple(seen[s] for s in SPLITS) != counts:
            raise ArchiveError("Archive index does not exactly cover payload and split counts")
        return cls(require_sha256(row["manifest_sha256"]), counts, fingerprints, shards)
