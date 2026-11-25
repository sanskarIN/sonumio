"""Streaming numerical readiness audit over a verified private corpus archive.

This mirrors the existing in-memory readiness policy without retaining the
whole corpus. Aggregate reports never expose captions, paths or identifiers.
Exact crop checks and manifest rights declarations are not independent
verification of recording rights, voices or statistical independence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from research.audio_dataset import _cropped_clip_digest
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError, ArchiveRow, SPLITS, require_sha256
from research.corpus_readiness import ReadinessPolicy, clip_quality, LANGUAGES
from research.wave_diffusion import CATEGORIES

PROTOCOL = "sonumio-streamed-archive-readiness-v1"
ELIGIBLE_DOMAIN = b"sonumio-eligible-crop-provenance-v1\0"
METRICS = ("centered_rms_dbfs", "active_fraction", "near_clip_fraction",
           "absolute_dc", "peak")


@dataclass(frozen=True, slots=True)
class EligibleArchiveClip:
    row: ArchiveRow
    pcm_sha256: str


@dataclass(frozen=True, slots=True)
class ArchiveReadinessResult:
    archive_sha256: str
    report: dict
    eligible: dict[str, tuple[EligibleArchiveClip, ...]]
    eligible_fingerprints: dict[str, str]

    def indices(self, split: str) -> tuple[int, ...]:
        if split not in SPLITS:
            raise ArchiveError("Unknown readiness split")
        return tuple(clip.row.index for clip in self.eligible[split])


def _hash_eligible(hasher, row: ArchiveRow, pcm_sha256: str) -> None:
    # Intentionally match the in-memory readiness fingerprint protocol.
    payload = json.dumps(
        [row.index, row.label, row.asset_id, row.group_id, row.language,
         row.source_line, row.crop_index, row.caption],
        ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(len(payload).to_bytes(4, "little"))
    hasher.update(payload)
    hasher.update(bytes.fromhex(require_sha256(pcm_sha256)))


def eligible_fingerprint(clips: tuple[EligibleArchiveClip, ...]) -> str:
    hasher = hashlib.sha256(ELIGIBLE_DOMAIN)
    last = -1
    for clip in clips:
        if not isinstance(clip, EligibleArchiveClip) or clip.row.index <= last:
            raise ArchiveError("Eligible archive indices must be strictly ordered")
        last = clip.row.index
        _hash_eligible(hasher, clip.row, clip.pcm_sha256)
    return hasher.hexdigest()


class _CategorySummary:
    def __init__(self) -> None:
        self.clips = self.eligible_clips = 0
        self.groups = set()
        self.eligible_groups = set()
        self.assets = set()
        self.eligible_assets = set()
        self.languages = Counter()
        self.eligible_languages = Counter()
        self.metrics = {key: [] for key in METRICS}

    def add(self, row: ArchiveRow, quality: dict) -> None:
        self.clips += 1
        self.groups.add(row.group_id)
        self.assets.add(row.asset_id)
        self.languages[row.language] += 1
        for key in METRICS:
            if quality[key] is not None:
                self.metrics[key].append(quality[key])
        if not quality["reasons"]:
            self.eligible_clips += 1
            self.eligible_groups.add(row.group_id)
            self.eligible_assets.add(row.asset_id)
            self.eligible_languages[row.language] += 1

    def report(self) -> dict:
        metrics = {}
        for key, values in self.metrics.items():
            metrics[key] = {
                "defined_clips": len(values),
                "minimum": min(values) if values else None,
                "median": float(np.median(values)) if values else None,
                "maximum": max(values) if values else None,
            }
        return {
            "clips": self.clips, "eligible_clips": self.eligible_clips,
            "recording_groups": len(self.groups),
            "eligible_recording_groups": len(self.eligible_groups),
            "assets": len(self.assets), "eligible_assets": len(self.eligible_assets),
            "clips_by_language": {lang: self.languages[lang] for lang in LANGUAGES},
            "eligible_clips_by_language": {
                lang: self.eligible_languages[lang] for lang in LANGUAGES
            },
            "signal_metrics": metrics,
        }


def audit_archive(
    reader: CorpusArchiveReader, policy: ReadinessPolicy | None = None,
) -> ArchiveReadinessResult:
    """Apply the existing numerical crop policy to every verified archive shard.

    Persistent state is metadata, fingerprints and scalar summaries; PCM is
    processed one shard at a time. Retaining eligible identifiers is private
    in-process state, never part of the aggregate report.
    """
    if not isinstance(reader, CorpusArchiveReader):
        raise ArchiveError("Expected verified private corpus reader")
    if policy is None:
        policy = ReadinessPolicy()
    if not isinstance(policy, ReadinessPolicy):
        raise ArchiveError("Expected numerical corpus readiness policy")
    metadata = reader.report()
    summaries = {}
    eligible = {}
    fingerprints = {}
    for split in SPLITS:
        by_category = [_CategorySummary() for _ in CATEGORIES]
        seen_hashes = set()
        groups = set()
        eligible_groups = set()
        rejected = Counter()
        rows = []
        expected = 0
        for shard in reader.iter_shards(split):
            for row, clip in zip(shard.rows, shard.clips, strict=True):
                if row.index != expected:
                    raise ArchiveError("Archive readiness row order changed")
                expected += 1
                waveform = clip[0]
                quality = clip_quality(waveform, policy)
                pcm_hash = _cropped_clip_digest(waveform)
                if pcm_hash in seen_hashes:
                    quality["reasons"].append("repeated_exact_crop")
                seen_hashes.add(pcm_hash)
                groups.add(row.group_id)
                rejected.update(quality["reasons"])
                by_category[row.label].add(row, quality)
                if not quality["reasons"]:
                    rows.append(EligibleArchiveClip(row, pcm_hash))
                    eligible_groups.add(row.group_id)
        if expected != metadata["split_counts"][split]:
            raise ArchiveError("Archive readiness did not consume declared split")
        eligible[split] = tuple(rows)
        fingerprints[split] = eligible_fingerprint(eligible[split])
        summaries[split] = {
            "split_fingerprint": metadata["split_fingerprints"][split],
            "clips": expected, "eligible_clips": len(rows),
            "recording_groups": len(groups),
            "eligible_recording_groups": len(eligible_groups),
            "rejection_reason_counts": dict(sorted(rejected.items())),
            "by_category": {
                name: summary.report()
                for name, summary in zip(CATEGORIES, by_category, strict=True)
            },
        }
    failures = []
    for split, required in (
        ("train", policy.min_train_groups_per_category),
        ("validation", policy.min_validation_groups_per_category),
    ):
        for category in CATEGORIES:
            observed = summaries[split]["by_category"][category]["eligible_recording_groups"]
            if observed < required:
                failures.append({"split": split, "category": category,
                                 "groups": observed, "required": required})
    # Recheck descriptor metadata after the entire streamed audit.
    if reader.report()["archive_sha256"] != metadata["archive_sha256"]:
        raise ArchiveError("Private archive identity changed during audit")
    report = {
        "protocol": PROTOCOL, "policy": policy.as_dict(),
        "archive_sha256": metadata["archive_sha256"],
        "manifest_sha256": metadata["manifest_sha256"],
        "splits": summaries, "coverage_failures": failures,
        "readiness_gate_passed": not failures,
        "eligible_split_fingerprints": fingerprints.copy(),
        "exact_cross_split_crop_overlap": False,
        "rights_and_voice_claims_independently_verified": False,
        "model_quality_established": False,
        "audio_or_weights_saved": False,
        "pcm_processed_one_shard_at_a_time": True,
    }
    return ArchiveReadinessResult(metadata["archive_sha256"], report, eligible, fingerprints)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        with CorpusArchiveReader(args.archive, expected_sha256=args.expected_sha256) as reader:
            result = audit_archive(reader)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Archive readiness: {exc}\n")
    print(json.dumps(result.report, sort_keys=True, allow_nan=False))
    return 2 if args.require_ready and not result.report["readiness_gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
