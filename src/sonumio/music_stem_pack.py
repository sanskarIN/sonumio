"""Deterministic, verified stereo PCM stem archives for original arrangements.

The ZIP contains a canonical score plus eight dry PCM16 stem members. A
single attenuation factor preserves relative stem levels. Verification checks
all bytes and integer PCM receipts without extracting files. Optional replay
tests the procedural source claim in a compatible local numerical runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import math
import zipfile

import numpy as np

from .arrangement import Arrangement, TRACKS, arrangement_digest
from .audio import SAMPLE_RATE, encode_wav_channels
from .audio_core.asset_archive import _snapshot
from .audio_core.hashing import canonical_json_bytes, parse_canonical_json, strict_sha256
from .audio_core.wave_reader import _decode_canonical_header
from .errors import GenerationError
from .music_mix import render_stems
from .music_publish import MAX_STEM_PACK_BYTES, publish_music_artifact
from .music_score_io import MAX_SCORE_BYTES, decode_score, encode_score

PROTOCOL = "sonumio-original-music-stems-v1"
MANIFEST_NAME = "manifest.json"
SCORE_NAME = "score.json"
MAX_FRAMES = 60 * SAMPLE_RATE
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MEMBER_BYTES = 44 + 4 * MAX_FRAMES
PEAK_CEILING = .9
FIXED_DATE = (1980, 1, 1, 0, 0, 0)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _member(track):
    if type(track) is not str or track not in TRACKS:
        raise GenerationError("Unknown original music stem")
    return "stems/" + track + ".wav"


def _expected_names():
    return (MANIFEST_NAME, SCORE_NAME, *sorted(_member(track) for track in TRACKS))


def _validate_score(score):
    if not isinstance(score, Arrangement):
        raise GenerationError("Expected Arrangement for stem export")
    frames = score.plan.frame(score.plan.ticks)
    if not 1 <= frames <= MAX_FRAMES:
        raise GenerationError("Stem package exceeds 60-second frame budget")
    return frames


def _attenuation(source_peak):
    if (type(source_peak) not in (float, int)
            or not math.isfinite(source_peak) or not 0 <= source_peak <= 10000):
        raise GenerationError("Invalid declared source stem peak")
    return min(1., PEAK_CEILING / max(float(source_peak), 1e-15))


def _info(name):
    info = zipfile.ZipInfo(name, FIXED_DATE)
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_STORED
    return info


def _pack(manifest_raw, score_raw, members):
    if type(manifest_raw) is not bytes or not 1 <= len(manifest_raw) <= MAX_MANIFEST_BYTES:
        raise GenerationError("Invalid music stem manifest byte count")
    if type(score_raw) is not bytes or not 1 <= len(score_raw) <= MAX_SCORE_BYTES:
        raise GenerationError("Invalid embedded score byte count")
    if type(members) is not dict or set(members) != set(TRACKS):
        raise GenerationError("Expected exactly eight music stems")
    if any(type(wav) is not bytes or not 44 < len(wav) <= MAX_MEMBER_BYTES
           for wav in members.values()):
        raise GenerationError("Invalid stem WAV byte count")
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        archive.writestr(_info(MANIFEST_NAME), manifest_raw)
        archive.writestr(_info(SCORE_NAME), score_raw)
        for track in sorted(TRACKS):
            archive.writestr(_info(_member(track)), members[track])
    raw = sink.getvalue()
    if len(raw) > MAX_STEM_PACK_BYTES:
        raise GenerationError("Stem ZIP exceeds bounded byte budget")
    return raw


def _inspect_stem(track, raw, frames):
    if type(raw) is not bytes or len(raw) != 44 + frames * 4:
        raise GenerationError("Stem WAV byte count differs from score clock")
    geometry = _decode_canonical_header(raw[:44], len(raw))
    if (geometry.sample_rate, geometry.channels, geometry.bits_per_sample,
            geometry.frames) != (SAMPLE_RATE, 2, 16, frames):
        raise GenerationError("Unexpected stem WAV geometry")
    # Compute exact integer receipts with bounded temporary allocation.
    pcm = np.frombuffer(raw, dtype="<i2", offset=44)
    peak = [0, 0]
    sums = [0, 0]
    squares = [0, 0]
    nonzero = [0, 0]
    for start in range(0, frames, 16384):
        block = pcm[2 * start:2 * min(frames, start + 16384)].reshape(-1, 2).astype(np.int64)
        for channel in (0, 1):
            values = block[:, channel]
            peak[channel] = max(peak[channel], int(np.max(np.abs(values))))
            sums[channel] += int(np.sum(values, dtype=np.int64))
            squares[channel] += int(np.sum(values * values, dtype=np.int64))
            nonzero[channel] += int(np.count_nonzero(values))
    if max(peak) > round(PEAK_CEILING * 32767):
        raise GenerationError("Stem PCM exceeds common peak ceiling")
    return {
        "track": track, "member": _member(track), "wav_sha256": _digest(raw),
        "bytes": len(raw), "peak_pcm16": peak, "sample_sum": sums,
        "sample_square_sum": squares, "nonzero_samples": nonzero,
    }


def _manifest(score, score_raw, members, source_peak):
    frames = _validate_score(score)
    scale = _attenuation(source_peak)  # Validate before any numeric conversion.
    return {
        "format": PROTOCOL,
        "score_sha256": arrangement_digest(score),
        "score_bytes_sha256": _digest(score_raw),
        "score_bytes": len(score_raw),
        "sample_rate": SAMPLE_RATE, "channels": 2, "bits_per_sample": 16,
        "frames": frames,
        "declared_dry_source_peak": float(source_peak),
        "common_attenuation": scale,
        "peak_ceiling": PEAK_CEILING,
        "stems": [_inspect_stem(track, members[track], frames) for track in TRACKS],
        "dry_stems": True, "room_effects_applied": False,
        "static_mix_gains_applied": False, "learned_model": False,
    }


def _build(score):
    frames = _validate_score(score)  # Refuse oversize scores before synthesis.
    stems = render_stems(score)
    if type(stems) is not dict or set(stems) != set(TRACKS):
        raise GenerationError("Unexpected original stem collection")
    for stem in stems.values():
        if (not isinstance(stem, np.ndarray) or stem.dtype != np.float32
                or stem.shape != (2, frames) or not np.isfinite(stem).all()):
            raise GenerationError("Invalid dry stem samples")
    source_peak = max(float(np.max(np.abs(stem))) for stem in stems.values())
    scale = _attenuation(source_peak)
    members = {}
    for track in TRACKS:
        # Do not mutate dry stem samples or normalize each track separately.
        scaled = (stems[track].astype(np.float64) * scale).astype(np.float32)
        members[track] = encode_wav_channels(scaled, SAMPLE_RATE)
    score_raw = encode_score(score)
    manifest = _manifest(score, score_raw, members, source_peak)
    return _pack(canonical_json_bytes(manifest, max_bytes=MAX_MANIFEST_BYTES),
                 score_raw, members)


def _read_member(archive, info, limit):
    if (info.compress_type != zipfile.ZIP_STORED or info.flag_bits != 0
            or info.extra or info.comment or info.is_dir()
            or info.orig_filename != info.filename
            or info.create_system != 3 or info.external_attr != (0o100644 << 16)
            or info.date_time != FIXED_DATE):
        raise GenerationError("Noncanonical music stem ZIP member")
    if info.compress_size != info.file_size or not 1 <= info.file_size <= limit:
        raise GenerationError("Music stem ZIP member exceeds byte budget")
    with archive.open(info, "r") as stream:
        data = stream.read(limit + 1)
        if len(data) != info.file_size or stream.read(1):
            raise GenerationError("Music stem ZIP member length mismatch")
    return data


def _unpack(raw):
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_STEM_PACK_BYTES:
        raise GenerationError("Invalid bounded music stem ZIP bytes")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            if archive.comment or tuple(info.filename for info in infos) != _expected_names():
                raise GenerationError("Unexpected music stem ZIP members or order")
            manifest_raw = _read_member(archive, infos[0], MAX_MANIFEST_BYTES)
            manifest = parse_canonical_json(manifest_raw, max_bytes=MAX_MANIFEST_BYTES)
            if type(manifest) is not dict or manifest.get("format") != PROTOCOL:
                raise GenerationError("Unknown music stem manifest format")
            score_raw = _read_member(archive, infos[1], MAX_SCORE_BYTES)
            score = decode_score(score_raw)
            frames = _validate_score(score)
            if score_raw != encode_score(score):
                raise GenerationError("Embedded score bytes are not canonical")
            size = 44 + 4 * frames
            if any(info.file_size != size for info in infos[2:]):
                raise GenerationError("Stem ZIP sizes differ from embedded score")
            members = {
                track: _read_member(archive, info, size)
                for track, info in zip(sorted(TRACKS), infos[2:], strict=True)
            }
    except (zipfile.BadZipFile, zipfile.LargeZipFile, EOFError, NotImplementedError,
            RecursionError, RuntimeError, UnicodeError, OverflowError) as exc:
        raise GenerationError("Invalid music stem ZIP structure") from exc
    return manifest_raw, manifest, score_raw, score, members


@dataclass(frozen=True, slots=True)
class VerifiedStemPack:
    archive_sha256: str
    archive_bytes: int
    manifest_sha256: str
    manifest: dict
    replay_verified: bool

    def as_dict(self):
        return {
            "protocol": PROTOCOL,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "manifest_sha256": self.manifest_sha256,
            "manifest": self.manifest,
            "recipe_replay_verified": self.replay_verified,
            "source_peak_replayed": self.replay_verified,
            "signature_verified": False,
        }


def _verify(raw, expected_sha256=None, replay=False):
    if type(replay) is not bool:
        raise GenerationError("Stem replay flag must be boolean")
    if type(raw) is not bytes or not 1 <= len(raw) <= MAX_STEM_PACK_BYTES:
        raise GenerationError("Invalid bounded music stem ZIP bytes")
    digest = _digest(raw)
    if expected_sha256 is not None and digest != strict_sha256(expected_sha256):
        raise GenerationError("Music stem archive SHA256 mismatch")
    manifest_raw, declared, score_raw, score, members = _unpack(raw)
    source_peak = declared.get("declared_dry_source_peak")
    expected = _manifest(score, score_raw, members, source_peak)
    canonical = canonical_json_bytes(expected, max_bytes=MAX_MANIFEST_BYTES)
    if canonical != manifest_raw:
        raise GenerationError("Music stem manifest differs from actual PCM receipts")
    if _pack(canonical, score_raw, members) != raw:
        raise GenerationError("Music stem archive differs from canonical ZIP")
    if replay and _build(score) != raw:
        raise GenerationError("Original music stem recipe replay mismatch")
    receipt = VerifiedStemPack(digest, len(raw), _digest(manifest_raw), expected, replay)
    return receipt, members


def verify_stem_pack(raw, *, expected_sha256=None, replay=False):
    """Validate all archive members; optionally repeat the original renderer."""
    return _verify(raw, expected_sha256, replay)[0]


def create_stem_pack(score):
    raw = _build(score)
    return raw, verify_stem_pack(raw)


def read_stem_pack(path, *, expected_sha256=None, replay=False):
    raw = _snapshot(path, MAX_STEM_PACK_BYTES)
    return verify_stem_pack(raw, expected_sha256=expected_sha256, replay=replay)


def select_stem(raw, track, *, expected_sha256=None):
    """Return one exact WAV only after the entire package has been checked."""
    _member(track)
    receipt, members = _verify(raw, expected_sha256)
    return members[track], receipt


def select_stem_file(path, track, *, expected_sha256=None):
    return select_stem(_snapshot(path, MAX_STEM_PACK_BYTES), track,
                       expected_sha256=expected_sha256)


def publish_stem_pack(path, raw, *, overwrite=False, expected_sha256=None):
    verified = verify_stem_pack(raw, expected_sha256=expected_sha256)
    receipt = publish_music_artifact(path, raw, kind="stems", overwrite=overwrite,
                                     expected_sha256=verified.archive_sha256)
    return {**verified.as_dict(), "output_sha256": receipt.sha256,
            "bytes_written": receipt.bytes_written}
