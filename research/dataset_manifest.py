"""Validate declarative training-audio provenance metadata before ingestion.

Validation establishes a bounded, single-read manifest snapshot. It checks
declared metadata consistency, NOT the truth of licenses, provenance or consent;
a human review remains mandatory before research training.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from io import BytesIO
from pathlib import Path

from research.local_snapshot import SnapshotError, read_local_snapshot

REQUIRED = {
    "asset_id", "source_url", "sha256", "caption", "language", "group_id",
    "split", "license", "source_kind", "rights_reviewed", "creator_permission",
    "voice_present", "sample_rate", "duration_seconds",
}
VALID_SPLITS = {"train", "validation", "test"}
VALID_LANGUAGES = {"en", "hi"}
VALID_SOURCE_KINDS = {"owned", "cc0", "direct-permission"}
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024
MAX_RECORDS = 10_000


class ManifestError(ValueError):
    """One dataset record failed the enforceable declaration gates."""


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    row = {}
    for key, value in pairs:
        if key in row:
            raise ValueError("duplicate JSON object key")
        row[key] = value
    return row


def _reject_constant(value: str):
    raise ValueError("Nonfinite JSON number")


def _finite_json_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite JSON number")
    return result


def load_manifest_snapshot(path: str | Path) -> tuple[tuple[tuple[int, dict], ...], dict]:
    """Return validated (source line, record) pairs from one bounded byte snapshot.

    This avoids validating a file and silently loading changed captions from
    a second manifest read. File-level SHA256 is in the report; audio WAV
    hashes are separately verified by the ingestion loader.
    """
    try:
        raw = read_local_snapshot(path, max_bytes=MAX_MANIFEST_BYTES)
    except SnapshotError as exc:
        raise ManifestError(f"manifest snapshot: {exc}") from exc
    identities: set[str] = set()
    hashes: set[str] = set()
    group_splits: dict[str, str] = {}
    counts: Counter[str] = Counter()
    records: list[tuple[int, dict]] = []

    for line_no, line in enumerate(BytesIO(raw), 1):
        def error(reason: str) -> None:
            raise ManifestError(f"line {line_no}: {reason}")
        if len(line.rstrip(b"\r\n")) > MAX_LINE_BYTES:
            error("manifest record exceeds 16 KiB limit")
        if not line.strip():
            continue
        if len(records) >= MAX_RECORDS:
            error("manifest exceeds 10000 record limit")
        try:
            row = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_keys,
                             parse_constant=_reject_constant, parse_float=_finite_json_float)
            # Reject lone escaped UTF-16 surrogates even in optional metadata.
            json.dumps(row, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ManifestError(f"line {line_no}: invalid or duplicate-key JSON") from exc
        if not isinstance(row, dict):
            error("manifest entry must be a JSON object")
        if missing := REQUIRED - row.keys():
            error(f"missing fields {sorted(missing)}")
        if not all(isinstance(row[k], str) and row[k].strip() for k in
                   ("asset_id", "source_url", "caption", "group_id", "sha256", "license")):
            error("metadata identifiers and captions must be non-empty strings")
        if len(row["caption"]) > 800:
            error("caption must have at most 800 characters")
        if len(row["sha256"]) != 64 or any(ch not in "0123456789abcdef" for ch in row["sha256"]):
            error("sha256 must be a lowercase 64-character hex checksum")
        if row["asset_id"] in identities or row["sha256"] in hashes:
            error("duplicate asset_id or waveform checksum")
        if not isinstance(row["split"], str) or row["split"] not in VALID_SPLITS:
            error("unsupported split")
        if not isinstance(row["language"], str) or row["language"] not in VALID_LANGUAGES:
            error("unsupported language")
        if not isinstance(row["source_kind"], str) or row["source_kind"] not in VALID_SOURCE_KINDS:
            error("source_kind must be owned, cc0, or direct-permission")
        if row["rights_reviewed"] is not True:
            error("record requires manual license/rights review")
        if row["source_kind"] == "cc0":
            if row["license"] != "CC0-1.0":
                error("CC0 source_kind requires explicit CC0-1.0 declaration")
        elif row["creator_permission"] is not True:
            error("owned and direct-permission recordings require explicit permission")
        if row["voice_present"] is not False:
            error("voice recordings are excluded from the initial Sonumio training dataset")
        if type(row["sample_rate"]) is not int or not 8_000 <= row["sample_rate"] <= 192_000:
            error("sample_rate out of range")
        duration = row["duration_seconds"]
        if (type(duration) not in (float, int) or not 0.1 <= duration <= 600.0
                or not math.isfinite(duration)):
            error("duration_seconds must be finite and within range")
        group_id = row["group_id"]
        if group_id in group_splits and group_splits[group_id] != row["split"]:
            error("recording group leaks across train/evaluation splits")
        identities.add(row["asset_id"])
        hashes.add(row["sha256"])
        group_splits[group_id] = row["split"]
        counts[row["split"]] += 1
        records.append((line_no, row))

    if not records:
        raise ManifestError("manifest contains no records")
    report = {
        "status": "declared metadata checks passed; human verification still required",
        "records": len(records), "split_counts": dict(sorted(counts.items())),
        "unique_recording_groups": len(group_splits),
        "manifest_fingerprint": hashlib.sha256(raw).hexdigest(),
    }
    return tuple(records), report


def validate_manifest(path: str | Path) -> dict:
    """Validate metadata; file bytes are hashed as consumed by the loader."""
    _, report = load_manifest_snapshot(path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_manifest(args.manifest), indent=2))


if __name__ == "__main__":
    main()
