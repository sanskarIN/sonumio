"""Select a single verified asset from a canonical sound library.

No ZIP extraction or archive-derived filesystem destination is performed.
The caller chooses the output path; both source library and selected WAV
are verified before atomic publication.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import os

from .asset_pack import identifier, MAX_VARIANTS
from .asset_archive import _snapshot, _read_archive, MAX_ARCHIVE_BYTES, verify_pack_bytes
from .hashing import strict_sha256
from .errors import AudioContractError, AudioFormatError, AudioIntegrityError


@dataclass(frozen=True, slots=True)
class SelectedAsset:
    archive_sha256: str
    manifest_sha256: str
    record: dict
    wav: bytes

    def as_dict(self):
        return {
            "archive_sha256": self.archive_sha256,
            "manifest_sha256": self.manifest_sha256,
            "asset": self.record,
            "selected_wav_bytes": len(self.wav),
            "selected_wav_sha256": hashlib.sha256(self.wav).hexdigest(),
            "archive_extracted": False,
        }


def select_asset(raw, asset_id, variant=1, *, expected_archive_sha256=None,
                 expected_wav_sha256=None):
    identifier(asset_id)
    if type(variant) is not int or not 1 <= variant <= MAX_VARIANTS:
        raise AudioContractError("Invalid sound pack variant index")
    if expected_wav_sha256 is not None:
        strict_sha256(expected_wav_sha256)
    receipt = verify_pack_bytes(raw, expected_sha256=expected_archive_sha256)
    member = f"audio/{asset_id}-{variant:03d}.wav"
    matches = [r for r in receipt.manifest["assets"] if r["member"] == member]
    if len(matches) != 1:
        raise AudioContractError("Requested asset variant is not present")
    record = matches[0]
    _, _, members = _read_archive(raw)
    wav = members[member]
    digest = hashlib.sha256(wav).hexdigest()
    if digest != record["wav_sha256"]:
        raise AudioIntegrityError("Selected asset differs from verified manifest")
    if expected_wav_sha256 is not None and digest != expected_wav_sha256:
        raise AudioIntegrityError("Selected WAV SHA256 mismatch")
    return SelectedAsset(receipt.archive_sha256, receipt.manifest_sha256, record, wav)


def list_pack_assets(raw, *, tags=(), source=None, expected_archive_sha256=None):
    """Filter verified records by an AND match of explicit recipe tags."""
    if type(tags) is not tuple or any(type(tag) is not str for tag in tags):
        raise AudioContractError("Selection tags must be a tuple of strings")
    if len(tags)>16 or len(set(tags))!=len(tags):
        raise AudioContractError("Selection tags must be bounded and unique")
    if source not in (None,"nature","procedural"):
        raise AudioContractError("Unknown asset source filter")
    receipt=verify_pack_bytes(raw,expected_sha256=expected_archive_sha256)
    recipes={r["asset_id"]:r for r in receipt.manifest["plan"]["recipes"]}
    records=[]
    for record in receipt.manifest["assets"]:
        recipe=recipes[record["asset_id"]]
        if source is not None and record["source"]!=source:
            continue
        if not set(tags)<=set(recipe["tags"]):
            continue
        records.append({**record,"tags":list(recipe["tags"])})
    return {
        "archive_sha256":receipt.archive_sha256,
        "manifest_sha256":receipt.manifest_sha256,
        "matched_assets":len(records),
        "assets":records,
    }


def list_pack_file(path, *, tags=(), source=None, expected_archive_sha256=None):
    return list_pack_assets(_snapshot(path,MAX_ARCHIVE_BYTES),tags=tags,source=source,
                            expected_archive_sha256=expected_archive_sha256)


def export_pack_asset(source,destination,asset_id,variant=1, *,
                      expected_archive_sha256=None,expected_wav_sha256=None,
                      overwrite=False,create_parent=False):
    from ..atomic_cli_wav import publish_generated_wav
    source,target=Path(source),Path(destination)
    if target.suffix.lower()!=".wav":
        raise AudioFormatError("Selected asset destination must end in .wav")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Asset publication flags must be boolean")
    def distinct():
        if source.resolve()==target.resolve():
            raise AudioFormatError("Sound pack and destination must differ")
        if target.exists():
            try:
                if os.path.samefile(source,target):
                    raise AudioFormatError("Sound pack and destination share file identity")
            except FileNotFoundError:
                pass
    distinct()
    selected=select_asset(_snapshot(source,MAX_ARCHIVE_BYTES),asset_id,variant,
        expected_archive_sha256=expected_archive_sha256,
        expected_wav_sha256=expected_wav_sha256)
    distinct()
    result=publish_generated_wav(target,selected.wav,overwrite=overwrite,
        create_parent=create_parent,expected_sha256=selected.record["wav_sha256"])
    return {**selected.as_dict(),"destination":str(result.path),
            "output_sha256":result.wav_sha256}
