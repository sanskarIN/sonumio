"""Canonical complete-sound archive with shared bounded ZIP verification."""
from __future__ import annotations

from .errors import GenerationError
from .music_render_bundle import _pack_members, _read_members, MAX_BUNDLE_BYTES
from .music_render_recipe import MAX_RECEIPT_BYTES, encode_receipt, decode_receipt, _sha, _canonical
from .music_render_wav import MAX_FRAMES
from .music_sound_recipe import (
    MAX_BYTES, encode_sound_recipe, decode_sound_recipe, render_sound, verify_sound,
)

FORMAT = "sonumio-complete-sound-bundle-v1"
MEMBERS = {
    "manifest.json": 8192,
    "sound.json": MAX_BYTES,
    "receipt.json": MAX_RECEIPT_BYTES,
    "audio.wav": 44 + MAX_FRAMES * 4,
}


def _manifest(recipe, report, contents):
    return {
        "format":FORMAT,"sound_recipe_sha256":recipe.sha256,
        "wav_sha256":report["wav_sha256"],"receipt_sha256":report["receipt_sha256"],
        "files":{name:{"bytes":len(value),"sha256":_sha(value)} for name,value in contents.items()},
        "authenticated":False,
    }


def encode_sound_bundle(recipe, wav, report):
    verify_sound(recipe,wav,report)
    contents={"sound.json":encode_sound_recipe(recipe),"receipt.json":encode_receipt(report),
              "audio.wav":wav}
    contents["manifest.json"]=_canonical(_manifest(recipe,report,contents))
    return _pack_members(contents,MEMBERS,MAX_BUNDLE_BYTES)


def create_sound_bundle(recipe):
    wav,report=render_sound(recipe)
    return encode_sound_bundle(recipe,wav,report)


def verify_sound_bundle(data, *, expected_sha256=None, replay=False):
    contents=_read_members(data,MEMBERS,MAX_BUNDLE_BYTES)
    recipe=decode_sound_recipe(contents["sound.json"],expected_sha256=expected_sha256)
    report=decode_receipt(contents["receipt.json"])
    wav=contents["audio.wav"]
    manifest=contents.pop("manifest.json")
    if manifest!=_canonical(_manifest(recipe,report,contents)):
        raise GenerationError("Complete sound bundle manifest mismatch")
    checked=verify_sound(recipe,wav,report,replay=replay)
    if encode_sound_bundle(recipe,wav,report)!=data:
        raise GenerationError("Complete sound bundle archive is not canonical")
    return recipe,wav,report,{
        **checked,"format":FORMAT,"bundle_sha256":_sha(data),"bundle_bytes":len(data),
        "members_verified":list(MEMBERS),
    }


def read_sound_bundle(path):
    with open(path,"rb") as stream:
        data=stream.read(MAX_BUNDLE_BYTES+1)
    if not 1<=len(data)<=MAX_BUNDLE_BYTES:
        raise GenerationError("Invalid complete sound bundle size")
    return data
