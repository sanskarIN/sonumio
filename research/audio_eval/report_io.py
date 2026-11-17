"""Strict, local-only, deterministic numerical acoustic-report snapshots.

A report snapshot is not an audio archive: NumPy arrays, PCM payloads, WAV
bytes, model checkpoint bytes and arbitrary objects are not serializable.
Only small plain JSON metric structures are accepted. File output is an
explicit atomic LOCAL action, never an automatic research side effect.
Checksums detect accidental/tampered changes; they are not signatures,
identity authentication or evidence of copyrighted recording rights.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from sonumio.audio_core.errors import (
    AudioContractError, AudioFormatError, AudioLimitError,
    AudioIntegrityError,
)
from sonumio.audio_core.hashing import (
    canonical_json_bytes, parse_canonical_json, strict_sha256,
)

REPORT_PROTOCOL = "sonumio-numeric-audio-report-archive-v1"
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_COUNT = 128
_ALLOWED_DOMAINS = frozenset({
    "audio-core-health", "numerical-acoustic-evaluation",
    "first-party-dsp-reproducibility", "research-benchmark",
    "human-review-notes",
})
_FORBIDDEN_KEYS = frozenset({
    "audio_bytes", "wav_bytes", "pcm_samples", "raw_audio",
    "raw_waveform", "waveform_bytes", "base64_audio",
    "private_audio", "model_checkpoint", "checkpoint_bytes",
    "weight_bytes", "model_weights_bytes", "authentication_token",
    "authorization", "api_key", "secret", "password",
})
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")


def _numeric_metadata_only(value: Any, depth: int = 0) -> None:
    """A conservative validator guarding accidental embedded audio/secrets."""
    if depth > 16:
        raise AudioLimitError("Numerical report nesting exceeds maximum")
    if type(value) is dict:
        if len(value) > 1000:
            raise AudioLimitError("Numerical report object too large")
        for key, item in value.items():
            if not isinstance(key, str):
                raise AudioContractError("Report requires string metadata keys")
            if key.lower() in _FORBIDDEN_KEYS:
                raise AudioContractError("PCM, checkpoint or secret data cannot be archived")
            _numeric_metadata_only(item, depth + 1)
        return
    if type(value) in (tuple, list):
        if len(value) > 10000:
            raise AudioLimitError("Numerical report array too large")
        for item in value:
            _numeric_metadata_only(item, depth + 1)
        return
    if value is None or type(value) in (str, bool, int, float):
        return
    raise AudioContractError("Report cannot contain binary, PCM or Python objects")


@dataclass(frozen=True, slots=True)
class NumericReport:
    """Canonical small JSON metric report with content identity."""

    report_id: str
    domain: str
    payload: dict
    source_note: str = ""

    def __post_init__(self):
        if not isinstance(self.report_id, str) or _ID.fullmatch(self.report_id) is None:
            raise AudioContractError("Invalid numerical report identifier")
        if self.domain not in _ALLOWED_DOMAINS:
            raise AudioContractError("Unknown declared numerical report domain")
        if type(self.payload) is not dict or not self.payload:
            raise AudioContractError("Report payload must be a nonempty JSON object")
        if not isinstance(self.source_note, str) or len(self.source_note) > 1000:
            raise AudioContractError("Invalid report source note")
        _numeric_metadata_only(self.payload)
        canonical_json_bytes(self.payload, max_bytes=MAX_REPORT_BYTES)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(
            self.payload, max_bytes=MAX_REPORT_BYTES
        )).hexdigest()

    def header(self) -> dict:
        return {
            "report_id": self.report_id,
            "domain": self.domain,
            "source_note": self.source_note,
            "payload_sha256": self.content_sha256,
        }

    def as_dict(self) -> dict:
        data = {
            "protocol": REPORT_PROTOCOL,
            "header": self.header(),
            "payload": self.payload,
            "audio_files_embedded": False,
            "checkpoint_or_model_weights_embedded": False,
            "human_rights_authenticated": False,
            "release_authorized": False,
        }
        # Record SHA also binds metadata flags and report identity.
        data["record_sha256"] = hashlib.sha256(canonical_json_bytes(
            data, max_bytes=MAX_REPORT_BYTES
        )).hexdigest()
        return data

    @property
    def sha256(self) -> str:
        return self.as_dict()["record_sha256"]

    @classmethod
    def from_dict(cls, raw: dict) -> "NumericReport":
        if type(raw) is not dict or set(raw) != {
            "protocol", "header", "payload", "record_sha256",
            "audio_files_embedded", "checkpoint_or_model_weights_embedded",
            "human_rights_authenticated", "release_authorized",
        }:
            raise AudioIntegrityError("Invalid versioned numeric archive schema")
        if (raw["protocol"] != REPORT_PROTOCOL
                or raw["audio_files_embedded"] is not False
                or raw["checkpoint_or_model_weights_embedded"] is not False
                or raw["human_rights_authenticated"] is not False
                or raw["release_authorized"] is not False):
            raise AudioIntegrityError("Invalid or forged acoustic report claims")
        header = raw["header"]
        if type(header) is not dict or set(header) != {
            "report_id", "domain", "source_note", "payload_sha256"
        }:
            raise AudioIntegrityError("Report header was changed")
        actual_sha = hashlib.sha256(canonical_json_bytes(
            {k: v for k, v in raw.items() if k != "record_sha256"},
            max_bytes=MAX_REPORT_BYTES
        )).hexdigest()
        if actual_sha != strict_sha256(raw["record_sha256"]):
            raise AudioIntegrityError("Numerical report canonical digest mismatch")
        payload_digest = hashlib.sha256(canonical_json_bytes(
            raw["payload"], max_bytes=MAX_REPORT_BYTES
        )).hexdigest()
        if payload_digest != strict_sha256(header["payload_sha256"]):
            raise AudioIntegrityError("Numerical report payload digest mismatch")
        result = cls(header["report_id"], header["domain"], raw["payload"],
                     header["source_note"])
        if result.sha256 != raw["record_sha256"]:
            raise AudioIntegrityError("Numerical report serialization drift")
        return result


@dataclass(frozen=True, slots=True)
class NumericReportBundle:
    """Ordered, deduplicated collection of validated report fingerprints."""

    reports: tuple[NumericReport, ...]
    bundle_id: str

    def __post_init__(self):
        if not isinstance(self.reports, tuple) or not 1 <= len(
            self.reports
        ) <= MAX_BUNDLE_COUNT:
            raise AudioContractError("Expected 1..128 numerical reports")
        if any(not isinstance(x, NumericReport) for x in self.reports):
            raise AudioContractError("Bundle contains non-report entries")
        if (not isinstance(self.bundle_id, str)
                or _ID.fullmatch(self.bundle_id) is None):
            raise AudioContractError("Invalid numerical report bundle id")
        keys = [x.report_id for x in self.reports]
        if len(set(keys)) != len(keys):
            raise AudioContractError("Duplicate report ID in one snapshot bundle")

    def as_dict(self) -> dict:
        return {
            "protocol": "sonumio-numeric-audio-report-bundle-v1",
            "bundle_id": self.bundle_id,
            "reports": [
                {"report_id": r.report_id,
                 "domain": r.domain, "sha256": r.sha256}
                for r in self.reports
            ],
            "numeric_payloads_embedded": False,
            "audio_files_embedded": False,
            "model_weights_embedded": False,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


def serialize_numeric_report(report: NumericReport) -> bytes:
    if not isinstance(report, NumericReport):
        raise AudioContractError("Expected NumericReport")
    encoded = canonical_json_bytes(
        report.as_dict(), max_bytes=MAX_REPORT_BYTES
    ) + b"\n"
    if len(encoded) > MAX_REPORT_BYTES:
        raise AudioLimitError("Numerical report serialization exceeds cap")
    return encoded


def deserialize_numeric_report(blob: bytes) -> NumericReport:
    if not isinstance(blob, bytes) or not 0 < len(blob) <= MAX_REPORT_BYTES:
        raise AudioFormatError("Expected bounded numerical JSON report bytes")
    obj = parse_canonical_json(blob.strip(), max_bytes=MAX_REPORT_BYTES)
    return NumericReport.from_dict(obj)


def read_numeric_report(path: str | Path) -> NumericReport:
    file = Path(path)
    if not file.is_file() or file.is_symlink() or file.suffix.lower() != ".json":
        raise AudioFormatError("Expected ordinary local .json report")
    if file.stat().st_size > MAX_REPORT_BYTES:
        raise AudioLimitError("Numerical report file exceeds byte budget")
    value = file.read_bytes()
    if len(value) > MAX_REPORT_BYTES:
        raise AudioLimitError("Numerical report grew during snapshot read")
    return deserialize_numeric_report(value)


def write_numeric_report(report: NumericReport, path: str | Path, *,
                         overwrite: bool = False) -> Path:
    """Explicit, atomic, local-only numeric JSON output; no implicit directories."""
    if type(overwrite) is not bool:
        raise AudioContractError("overwrite must be Boolean")
    file = Path(path)
    if file.suffix.lower() != ".json" or not file.parent.is_dir() or file.is_symlink():
        raise AudioFormatError("Invalid local .json report destination")
    if file.exists() and not overwrite:
        raise AudioFormatError("Existing numeric report requires overwrite opt-in")
    content = serialize_numeric_report(report)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=file.parent, prefix=".sonumio-metrics-", suffix=".partial",
            delete=False
        ) as fp:
            temp = Path(fp.name)
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        if file.exists() and not overwrite:
            raise AudioFormatError("Report destination became occupied")
        os.replace(temp, file)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return file


def compare_numeric_report_versions(previous: NumericReport,
                                    current: NumericReport) -> dict:
    if not isinstance(previous, NumericReport) or not isinstance(
        current, NumericReport
    ):
        raise AudioContractError("Expected two validated numerical reports")
    if (previous.report_id != current.report_id
            or previous.domain != current.domain):
        raise AudioContractError("Cannot compare unrelated numerical report identities")
    before, after = previous.payload, current.payload
    shared = set(before) & set(after)
    return {
        "report_id": previous.report_id,
        "domain": previous.domain,
        "old_sha256": previous.sha256,
        "new_sha256": current.sha256,
        "changed": previous.sha256 != current.sha256,
        "added_payload_keys": sorted(set(after) - set(before)),
        "deleted_payload_keys": sorted(set(before) - set(after)),
        "changed_payload_keys": sorted(
            key for key in shared
            if canonical_json_bytes(before[key]) != canonical_json_bytes(after[key])
        ),
        "rights_or_release_status_inferred": False,
        "binary_audio_or_weights_compared": False,
    }
