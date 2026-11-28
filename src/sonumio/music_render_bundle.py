"""Deterministic local render bundles; fixed members, no extraction to disk."""
from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile

from .errors import GenerationError
from .music_render_recipe import (
    MAX_RECIPE_BYTES, MAX_RECEIPT_BYTES, encode_recipe, decode_recipe,
    encode_receipt, decode_receipt, render_recipe, verify_render,
)
from .music_render_wav import MAX_FRAMES

FORMAT = "sonumio-render-bundle-v1"
MAX_BUNDLE_BYTES = 20 * 1024 * 1024
MEMBERS = {
    "manifest.json": 8192,
    "recipe.json": MAX_RECIPE_BYTES,
    "receipt.json": MAX_RECEIPT_BYTES,
    "audio.wav": 44 + MAX_FRAMES * 4,
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _manifest(recipe, report, contents):
    return {
        "format": FORMAT, "recipe_sha256": recipe.sha256,
        "wav_sha256": report["wav_sha256"], "receipt_sha256": report["receipt_sha256"],
        "files": {name: {"bytes": len(data), "sha256": _sha(data)}
                  for name, data in contents.items()},
        "authenticated": False,
    }


def encode_bundle(recipe, wav, report):
    verify_render(recipe, wav, report)
    contents = {"recipe.json": encode_recipe(recipe),
                "receipt.json": encode_receipt(report), "audio.wav": wav}
    contents["manifest.json"] = _canonical(_manifest(recipe, report, contents))
    return _pack_members(contents, MEMBERS, MAX_BUNDLE_BYTES)


def _pack_members(contents, members, max_bytes):
    if set(contents) != set(members):
        raise GenerationError("Unexpected render bundle contents")
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name in members:
            data = contents[name]
            if not 1 <= len(data) <= members[name]:
                raise GenerationError("Render bundle member exceeds byte budget")
            info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, data)
    data = target.getvalue()
    if len(data) > max_bytes:
        raise GenerationError("Render bundle exceeds byte budget")
    return data


def create_bundle(recipe):
    wav, report = render_recipe(recipe)
    return encode_bundle(recipe, wav, report)


def _read_members(data, members=MEMBERS, max_bytes=MAX_BUNDLE_BYTES):
    if type(data) is not bytes or not 1 <= len(data) <= max_bytes:
        raise GenerationError("Invalid render bundle byte count")
    try:
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            infos = archive.infolist()
            if (len(infos) != len(members) or {info.filename for info in infos} != set(members)
                    or archive.comment):
                raise GenerationError("Unexpected render bundle members")
            result = {}
            for info in infos:
                if (info.is_dir() or info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED
                        or info.compress_size != info.file_size
                        or not 1 <= info.file_size <= members[info.filename]
                        or info.extra or info.comment
                        or stat.S_IFMT(info.external_attr >> 16) != stat.S_IFREG):
                    raise GenerationError("Invalid bounded render bundle member")
                with archive.open(info, "r") as stream:
                    payload = stream.read(members[info.filename]+1)
                    if len(payload) != info.file_size or stream.read(1):
                        raise GenerationError("Render bundle member size mismatch")
                    result[info.filename] = payload
            return result
    except (zipfile.BadZipFile, OSError, ValueError, RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed render bundle") from exc


def verify_bundle(data, *, replay=False, expected_recipe_sha256=None):
    contents = _read_members(data)
    recipe = decode_recipe(contents["recipe.json"], expected_sha256=expected_recipe_sha256)
    report = decode_receipt(contents["receipt.json"])
    wav = contents["audio.wav"]
    manifest = contents.pop("manifest.json")
    if manifest != _canonical(_manifest(recipe, report, contents)):
        raise GenerationError("Render bundle manifest differs from verified content")
    verification = verify_render(recipe, wav, report, replay=replay)
    # Re-encoding rejects trailing garbage, arbitrary ordering, duplicate
    # local records and noncanonical ZIP metadata without extracting files.
    if encode_bundle(recipe, wav, report) != data:
        raise GenerationError("Render bundle archive is not canonical")
    return recipe, wav, report, {
        **verification, "format": FORMAT, "bundle_sha256": _sha(data),
        "bundle_bytes": len(data), "members_verified": list(MEMBERS),
    }


def read_bundle(path):
    with open(path, "rb") as stream:
        data = stream.read(MAX_BUNDLE_BYTES+1)
    if not 1 <= len(data) <= MAX_BUNDLE_BYTES:
        raise GenerationError("Invalid render bundle byte count")
    return data
