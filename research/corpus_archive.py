"""Build and read a private, checksummed local reviewed-corpus archive.

The archive stores actual PCM and private provenance. It is not a public
artifact and does not convey recording rights. It is a reproducible local
cache of the existing reviewed ingestion recipe, not a new training model.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile

import numpy as np

from research.audio_dataset import LoadedSplit, load_reviewed_splits, _cropped_clip_digest
from research.corpus_archive_format import (
    ArchiveError, ArchiveIndex, ArchiveRow, ArchiveShard, ShardDescriptor,
    PROTOCOL, MAGIC, FOOTER_MAGIC, FOOTER, SPLITS, MAX_ARCHIVE_BYTES,
    MAX_INDEX_BYTES, MAX_SHARD_CLIPS, MAX_ROWS, MAX_BATCH_CLIPS,
    canonical_json, parse_canonical, require_sha256, digest, encode_shard, decode_shard,
)
from research.corpus_readiness import _validate_split
from research.wave_diffusion import CLIP_FRAMES, SAMPLE_RATE
from sonumio._file_identity import same_snapshot, same_path_descriptor


def _write_exact(fp, data: bytes, hasher) -> None:
    if fp.write(data) != len(data):
        raise ArchiveError("Short write while building private archive")
    hasher.update(data)


def _validate_ingestion(splits: dict[str, LoadedSplit], report: dict) -> str:
    if not isinstance(splits, dict) or set(splits) != set(SPLITS):
        raise ArchiveError("Expected all three reviewed splits")
    if not isinstance(report, dict) or report.get("wav_bytes_verified") is not True:
        raise ArchiveError("Archive requires verified ingestion report")
    try:
        fingerprint = require_sha256(report["declaration"]["manifest_fingerprint"])
        recorded = report["split_fingerprints"]
        counts = report["clip_counts"]
    except (KeyError, TypeError) as exc:
        raise ArchiveError("Incomplete verified ingestion report") from exc
    for name in SPLITS:
        _validate_split(name, splits[name])
        if (require_sha256(splits[name].fingerprint) != recorded.get(name)
                or len(splits[name].labels) != counts.get(name)):
            raise ArchiveError("Ingestion report differs from supplied split")
    total = sum(len(splits[s].labels) for s in SPLITS)
    if not len(splits["train"].labels) or not len(splits["validation"].labels) or total > MAX_ROWS:
        raise ArchiveError("Archive crop count exceeds format budget")
    return fingerprint


class _ProvenanceGuard:
    """Check internal identities and exact crop leakage without retaining PCM."""
    def __init__(self) -> None:
        self.assets = {}
        self.groups = {}
        self.crops = set()
        self.hashes = {}

    def add(self, shard: ArchiveShard) -> None:
        split = shard.descriptor.split
        for row, clip in zip(shard.rows, shard.clips, strict=True):
            owner = (split, row.group_id, row.language, row.source_line,
                     row.label, row.caption)
            if self.assets.setdefault(row.asset_id, owner) != owner:
                raise ArchiveError("Inconsistent archived asset provenance")
            if self.groups.setdefault(row.group_id, split) != split:
                raise ArchiveError("Archived recording group crosses splits")
            identity = (row.asset_id, row.crop_index)
            if identity in self.crops:
                raise ArchiveError("Duplicate archived crop identity")
            self.crops.add(identity)
            fingerprint = _cropped_clip_digest(clip[0])
            if self.hashes.setdefault(fingerprint, split) != split:
                raise ArchiveError("Exact archived model input crosses splits")


@dataclass(frozen=True, slots=True)
class ArchiveBuildResult:
    archive_sha256: str
    archive_bytes: int
    manifest_sha256: str
    split_counts: dict[str, int]
    shard_count: int

    def as_dict(self) -> dict:
        return {
            "protocol": PROTOCOL, "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes, "manifest_sha256": self.manifest_sha256,
            "split_counts": self.split_counts, "shard_count": self.shard_count,
            "contains_private_audio_and_provenance": True,
            "rights_independently_verified": False, "model_trained": False,
        }


def _write_archive(
    splits: dict[str, LoadedSplit], ingestion: dict, destination: str | Path, *,
    shard_clips: int, expected_sha256: str | None = None,
) -> ArchiveBuildResult:
    """Internal export of a just-verified ingestion result.

    Do not call this with hand-constructed or edited LoadedSplit values as
    a substitute for the public rights-gated ingestion path.
    """
    if type(shard_clips) is not int or not 1 <= shard_clips <= MAX_SHARD_CLIPS:
        raise ArchiveError("Invalid archive shard clip count")
    if expected_sha256 is not None:
        require_sha256(expected_sha256)
    manifest_hash = _validate_ingestion(splits, ingestion)
    destination = Path(destination)
    if destination.suffix != ".scorpus" or not destination.parent.is_dir():
        raise ArchiveError("Archive requires .scorpus path in an existing directory")
    if destination.exists() or destination.is_symlink():
        raise ArchiveError("Archive destination already exists")
    scratch = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=destination.parent, prefix=".sonumio-corpus-",
            suffix=".partial", delete=False,
        ) as fp:
            scratch = Path(fp.name)
            hasher = hashlib.sha256()
            _write_exact(fp, MAGIC, hasher)
            descriptors = []
            guard = _ProvenanceGuard()
            for name in SPLITS:
                split = splits[name]
                for start in range(0, len(split.labels), shard_clips):
                    stop = min(len(split.labels), start + shard_clips)
                    rows = []
                    for index in range(start, stop):
                        p = split.provenance[index]
                        rows.append(ArchiveRow(
                            index, int(split.labels[index]), split.captions[index],
                            p.asset_id, p.group_id, p.language, p.source_line, p.crop_index,
                        ))
                    payload = encode_shard(tuple(rows), split.clips[start:stop])
                    desc = ShardDescriptor(
                        name, start, stop - start, fp.tell(), len(payload), digest(payload),
                    )
                    guard.add(decode_shard(desc, payload))
                    if fp.tell() + len(payload) + FOOTER.size > MAX_ARCHIVE_BYTES:
                        raise ArchiveError("Private archive exceeds byte budget")
                    _write_exact(fp, payload, hasher)
                    descriptors.append(desc)
            index = ArchiveIndex(
                manifest_hash, tuple(len(splits[name].labels) for name in SPLITS),
                tuple(splits[name].fingerprint for name in SPLITS), tuple(descriptors),
            )
            # Validate complete geometry before any publication.
            ArchiveIndex.parse(index.as_dict(), fp.tell())
            raw_index = canonical_json(index.as_dict())
            if len(raw_index) > MAX_INDEX_BYTES:
                raise ArchiveError("Private archive index exceeds byte budget")
            footer = FOOTER.pack(len(raw_index), FOOTER_MAGIC)
            if fp.tell() + len(raw_index) + len(footer) > MAX_ARCHIVE_BYTES:
                raise ArchiveError("Private archive exceeds byte budget")
            _write_exact(fp, raw_index, hasher)
            _write_exact(fp, footer, hasher)
            size = fp.tell()
            archive_hash = hasher.hexdigest()
            if expected_sha256 is not None and archive_hash != expected_sha256:
                raise ArchiveError("Private archive output SHA256 mismatch")
            fp.flush()
            os.fsync(fp.fileno())
        # Verify the actual scratch bytes before the atomic create-if-absent.
        with CorpusArchiveReader(scratch, expected_sha256=archive_hash) as reader:
            if reader.archive_bytes != size:
                raise ArchiveError("Private archive scratch size mismatch")
        try:
            os.link(scratch, destination)
        except OSError as exc:
            raise ArchiveError("Atomic no-clobber archive publication failed") from exc
        return ArchiveBuildResult(
            archive_hash, size, manifest_hash, dict(zip(SPLITS, index.counts)), len(descriptors),
        )
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)


def build_archive(
    manifest: str | Path, audio_root: str | Path, destination: str | Path, *,
    clips_per_asset: int = 2, shard_clips: int = 64,
    expected_sha256: str | None = None,
) -> ArchiveBuildResult:
    """Run existing reviewed ingestion, then export a private local cache.

    The build phase remains limited by the in-memory ingestion implementation.
    The reader can subsequently fetch bounded batches without loading the
    complete PCM corpus. No network, weight export or public upload occurs.
    """
    if type(shard_clips) is not int or not 1 <= shard_clips <= MAX_SHARD_CLIPS:
        raise ArchiveError("Invalid archive shard clip count")
    if expected_sha256 is not None:
        require_sha256(expected_sha256)
    destination = Path(destination)
    if destination.suffix != ".scorpus" or not destination.parent.is_dir():
        raise ArchiveError("Archive requires .scorpus path in an existing directory")
    if destination.exists() or destination.is_symlink():
        raise ArchiveError("Archive destination already exists")
    splits, ingestion = load_reviewed_splits(
        manifest, audio_root, clips_per_asset=clips_per_asset,
    )
    return _write_archive(splits, ingestion, destination, shard_clips=shard_clips,
                          expected_sha256=expected_sha256)


@dataclass(frozen=True, slots=True, eq=False)
class ArchiveBatch:
    clips: np.ndarray
    labels: np.ndarray
    rows: tuple[ArchiveRow, ...]


class CorpusArchiveReader:
    """Verify a complete archive, then read bounded shards or training batches.

    All original bytes are hashed on construction. Supplying a trusted hash
    binds the cache to a previously approved build. Each shard is rechecked
    on access, and descriptor/path metadata is checked again. This class is
    synchronous and not safe for concurrent use of one reader instance.
    """

    def __init__(self, path: str | Path, *, expected_sha256: str | None = None):
        if expected_sha256 is not None:
            require_sha256(expected_sha256)
        self.path = Path(path)
        self._fp = None
        self._verified = False
        try:
            self._path_snapshot = self.path.lstat()
            if not stat.S_ISREG(self._path_snapshot.st_mode):
                raise ArchiveError("Expected ordinary private archive file")
            self.archive_bytes = self._path_snapshot.st_size
            if not len(MAGIC) + FOOTER.size < self.archive_bytes <= MAX_ARCHIVE_BYTES:
                raise ArchiveError("Archive size outside supported range")
            flags = (os.O_RDONLY | getattr(os, "O_BINARY", 0)
                     | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
            fd = os.open(self.path, flags)
            try:
                self._fp = os.fdopen(fd, "rb", buffering=0)
            except BaseException:
                os.close(fd)
                raise
            self._descriptor_snapshot = os.fstat(self._fp.fileno())
            self._check_snapshot()
            self._verify(expected_sha256)
        except BaseException as exc:
            self.close()
            if isinstance(exc, OSError):
                raise ArchiveError("Cannot open or verify private archive") from exc
            raise

    def _check_snapshot(self) -> None:
        if self._fp is None:
            raise ArchiveError("Private archive reader is closed")
        try:
            descriptor = os.fstat(self._fp.fileno())
            path = self.path.lstat()
        except OSError as exc:
            raise ArchiveError("Private archive disappeared or changed") from exc
        if (not stat.S_ISREG(descriptor.st_mode) or not stat.S_ISREG(path.st_mode)
                or not same_snapshot(self._path_snapshot, path)
                or not same_snapshot(self._descriptor_snapshot, descriptor)
                or not same_path_descriptor(path, descriptor)):
            raise ArchiveError("Private archive changed during read")

    def _read_at(self, offset: int, length: int) -> bytes:
        if self._fp is None:
            raise ArchiveError("Private archive reader is closed")
        if offset < 0 or length < 0 or offset + length > self.archive_bytes:
            raise ArchiveError("Private archive read exceeds declared size")
        self._fp.seek(offset)
        data = bytearray()
        while len(data) < length:
            part = self._fp.read(min(1 << 20, length - len(data)))
            if not part:
                raise ArchiveError("Truncated private archive")
            data.extend(part)
        return bytes(data)

    def _verify(self, expected_sha256: str | None) -> None:
        if self._read_at(0, len(MAGIC)) != MAGIC:
            raise ArchiveError("Invalid private archive magic")
        footer = self._read_at(self.archive_bytes - FOOTER.size, FOOTER.size)
        index_size, footer_magic = FOOTER.unpack(footer)
        if footer_magic != FOOTER_MAGIC or not 1 <= index_size <= MAX_INDEX_BYTES:
            raise ArchiveError("Invalid private archive footer")
        index_offset = self.archive_bytes - FOOTER.size - index_size
        if index_offset <= len(MAGIC):
            raise ArchiveError("Invalid private archive index offset")
        index_bytes = self._read_at(index_offset, index_size)
        index = ArchiveIndex.parse(parse_canonical(index_bytes), index_offset)
        hasher = hashlib.sha256(MAGIC)
        guard = _ProvenanceGuard()
        for item in index.shards:
            payload = self._read_at(item.offset, item.length)
            guard.add(decode_shard(item, payload))
            hasher.update(payload)
        hasher.update(index_bytes)
        hasher.update(footer)
        actual = hasher.hexdigest()
        self._check_snapshot()
        if expected_sha256 is not None and actual != expected_sha256:
            raise ArchiveError("Private archive SHA256 mismatch")
        self.archive_sha256 = actual
        self.index = index
        self._by_split = {
            split: tuple(item for item in index.shards if item.split == split)
            for split in SPLITS
        }
        self._starts = {
            split: tuple(item.start for item in self._by_split[split]) for split in SPLITS
        }
        self._verified = True

    def __enter__(self) -> "CorpusArchiveReader":
        if not self._verified or self._fp is None:
            raise ArchiveError("Private archive reader is closed")
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._verified = False
        if self._fp is not None:
            fp, self._fp = self._fp, None
            fp.close()

    def _require_open(self) -> None:
        if not self._verified or self._fp is None:
            raise ArchiveError("Private archive reader is closed")

    def report(self) -> dict:
        self._require_open()
        self._check_snapshot()
        return {
            "protocol": PROTOCOL, "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes, "manifest_sha256": self.index.manifest_sha256,
            "split_counts": dict(zip(SPLITS, self.index.counts)),
            "split_fingerprints": dict(zip(SPLITS, self.index.fingerprints)),
            "shard_count": len(self.index.shards), "sample_rate": SAMPLE_RATE,
            "clip_frames": CLIP_FRAMES, "all_shards_verified": True,
            "exact_cross_split_crop_leakage_detected": False,
            "contains_private_audio_and_provenance": True,
            "rights_independently_verified": False, "model_trained": False,
        }

    def _descriptor(self, split: str, index: int) -> ShardDescriptor:
        self._require_open()
        if split not in SPLITS:
            raise ArchiveError("Unknown archive split")
        if type(index) is not int or not 0 <= index < self.index.counts[SPLITS.index(split)]:
            raise ArchiveError("Archive crop index outside split")
        shard = bisect_right(self._starts[split], index) - 1
        return self._by_split[split][shard]

    def _read_shard(self, descriptor: ShardDescriptor) -> ArchiveShard:
        self._require_open()
        try:
            self._check_snapshot()
            result = decode_shard(
                descriptor, self._read_at(descriptor.offset, descriptor.length),
            )
            self._check_snapshot()
            return result
        except (ArchiveError, OSError):
            self.close()
            raise

    def iter_shards(self, split: str):
        self._require_open()
        if split not in SPLITS:
            raise ArchiveError("Unknown archive split")
        for item in self._by_split[split]:
            yield self._read_shard(item)

    def read_clip(self, split: str, index: int) -> tuple[ArchiveRow, np.ndarray]:
        descriptor = self._descriptor(split, index)
        shard = self._read_shard(descriptor)
        local = index - descriptor.start
        return shard.rows[local], shard.clips[local]

    def read_train_batch(self, indices: list[int] | tuple[int, ...]) -> ArchiveBatch:
        """Fetch ordered training rows only; duplicate indices are intentional.

        Group requests by shard so each shard is decoded once per batch.
        Returned PCM is an independent writable batch, not archive storage.
        No validation/test index can select a non-training recording.
        """
        self._require_open()
        if not isinstance(indices, (list, tuple)) or not 1 <= len(indices) <= MAX_BATCH_CLIPS:
            raise ArchiveError("Invalid archive training batch size")
        groups = {}
        for position, index in enumerate(indices):
            descriptor = self._descriptor("train", index)
            groups.setdefault(descriptor, []).append((position, index - descriptor.start))
        clips = np.empty((len(indices), 1, CLIP_FRAMES), dtype=np.float32)
        labels = np.empty(len(indices), dtype=np.int64)
        rows = [None] * len(indices)
        for descriptor, positions in groups.items():
            shard = self._read_shard(descriptor)
            for position, local in positions:
                row = shard.rows[local]
                clips[position] = shard.clips[local]
                labels[position] = row.label
                rows[position] = row
        return ArchiveBatch(clips, labels, tuple(rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build a private local reviewed-corpus cache")
    build.add_argument("manifest", type=Path)
    build.add_argument("audio_root", type=Path)
    build.add_argument("destination", type=Path)
    build.add_argument("--clips-per-asset", type=int, default=2)
    build.add_argument("--shard-clips", type=int, default=64)
    build.add_argument("--expected-sha256")
    inspect = commands.add_parser("inspect", help="Verify private cache; emit aggregate report only")
    inspect.add_argument("archive", type=Path)
    inspect.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_archive(
                args.manifest, args.audio_root, args.destination,
                clips_per_asset=args.clips_per_asset, shard_clips=args.shard_clips,
                expected_sha256=args.expected_sha256,
            ).as_dict()
        else:
            with CorpusArchiveReader(args.archive, expected_sha256=args.expected_sha256) as reader:
                result = reader.report()
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Private corpus archive: {exc}\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
