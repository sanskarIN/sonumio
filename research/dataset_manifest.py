"""Validate declarative training-audio provenance metadata before ingestion.

This checks consistency but DOES NOT verify truth of submitted license or
consent claims. Human rights review remains mandatory before training.
"""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = {
    "asset_id", "source_url", "sha256", "caption", "language", "group_id",
    "split", "license", "source_kind", "rights_reviewed", "creator_permission",
    "voice_present", "sample_rate", "duration_seconds",
}
VALID_SPLITS = {"train", "validation", "test"}
VALID_LANGUAGES = {"en", "hi"}
VALID_SOURCE_KINDS = {"owned", "cc0", "direct-permission"}

class ManifestError(ValueError):
    """One dataset record failed the enforceable declaration gates."""

def validate_manifest(path: str | Path) -> dict:
    path = Path(path)
    identities: set[str] = set()
    hashes: set[str] = set()
    group_splits: dict[str, str] = {}
    counts: Counter[str] = Counter()
    hasher = hashlib.sha256()
    with path.open("r", encoding="utf-8") as infile:
        for line_no, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"line {line_no}: invalid JSON") from exc
            if not isinstance(row, dict) or (missing := REQUIRED - row.keys()):
                raise ManifestError(f"line {line_no}: missing fields {sorted(missing) if isinstance(row, dict) else 'record'}")
            def error(reason: str):
                raise ManifestError(f"line {line_no}: {reason}")
            if not all(isinstance(row[k], str) and row[k].strip() for k in
                       ("asset_id", "source_url", "caption", "group_id", "sha256", "license")):
                error("metadata identifiers and captions must be non-empty strings")
            if len(row["sha256"]) != 64 or any(ch not in "0123456789abcdef" for ch in row["sha256"]):
                error("sha256 must be a lowercase 64-character hex checksum")
            if row["asset_id"] in identities or row["sha256"] in hashes:
                error("duplicate asset_id or waveform checksum")
            if row["split"] not in VALID_SPLITS or row["language"] not in VALID_LANGUAGES:
                error("unsupported split or language")
            if row["source_kind"] not in VALID_SOURCE_KINDS:
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
            if not isinstance(row["sample_rate"], int) or isinstance(row["sample_rate"], bool) or not 8_000 <= row["sample_rate"] <= 192_000:
                error("sample_rate out of range")
            if not isinstance(row["duration_seconds"], (float, int)) or isinstance(row["duration_seconds"], bool) or not 0.1 <= row["duration_seconds"] <= 600.0:
                error("duration_seconds out of range")
            group_id = row["group_id"]
            if group_id in group_splits and group_splits[group_id] != row["split"]:
                error("recording group leaks across train/evaluation splits")
            identities.add(row["asset_id"])
            hashes.add(row["sha256"])
            group_splits[group_id] = row["split"]
            counts[row["split"]] += 1
            hasher.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    if not counts:
        raise ManifestError("manifest contains no records")
    return {
        "status": "declared metadata checks passed; human verification still required",
        "records": sum(counts.values()), "split_counts": dict(sorted(counts.items())),
        "unique_recording_groups": len(group_splits), "manifest_fingerprint": hasher.hexdigest(),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_manifest(args.manifest), indent=2))

if __name__ == "__main__":
    main()
