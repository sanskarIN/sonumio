"""Deterministic multi-stem original-music packages, with explicit cost ceilings.

A stem pack is a SINGLE atomically published ZIP, not a directory of
partially-written audio files. Each track is rendered independently through
Sonumio's existing finite-Fourier wavetable engine and shares the original
score's exact global sample clock. All member WAV files are integer PCM.

The package is source-only work; it does not infer MIDI sound-bank fidelity,
load external samples, train a model, or claim perceptual audio quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256 as _sha256
from io import BytesIO
import json
import math
import numpy as np
from pathlib import Path
import re
import struct
import wave
import zipfile

from .contracts import (MidiFormatError, MidiIntegrityError, MidiLimitError,
                        MidiWorkstationError, integer, sha256)
from .score import NOTE_ID_RE, Score
from .transport import TransportSettings, build_schedule
from .wavetable_bridge import WavetableBridgeSettings
from .publish import PublicationOptions, publish_bytes

STEM_PROTOCOL = "sonumio-original-finite-fourier-stems-v1"
MAX_STEMS = 8
MAX_STEM_NOTES = 2048
MAX_TOTAL_PCM_SAMPLES = 900_000
MAX_STEM_PACKAGE_BYTES = 4 * 1024 * 1024
MAX_STEM_MANIFEST_BYTES = 32768
MAX_STEM_WAV_BYTES = 3 * 1024 * 1024
MAX_ZIP_ENTRY_BYTES = MAX_STEM_WAV_BYTES
_STEM_NAME = re.compile(r"^stems/([A-Za-z0-9][A-Za-z0-9_-]{0,63})\.wav$")


@dataclass(frozen=True, slots=True)
class StemPatch:
    track_id: str
    waveform: str = "triangle"
    gain: float = .08
    table_size: int = 2048
    max_harmonics: int = 64
    attack_frames: int = 64
    decay_frames: int = 256
    sustain_level: float = .7
    release_frames: int = 512

    def __post_init__(self):
        if type(self.track_id) is not str or not NOTE_ID_RE.fullmatch(self.track_id):
            raise MidiWorkstationError("Stem requires a valid stable track identifier")
        WavetableBridgeSettings(
            waveform=self.waveform, master_gain=self.gain,
            table_size=self.table_size, max_harmonics=self.max_harmonics,
            attack_frames=self.attack_frames, decay_frames=self.decay_frames,
            sustain_level=self.sustain_level, release_frames=self.release_frames,
        )
        if self.max_harmonics >= self.table_size // 2:
            raise MidiWorkstationError("Stem harmonic count exceeds Fourier table Nyquist")

    def bridge(self, *, sample_rate: int, channels: int,
               include_muted: bool, output_tail: int):
        return WavetableBridgeSettings(
            sample_rate=sample_rate, channels=channels,
            waveform=self.waveform, table_size=self.table_size,
            max_harmonics=self.max_harmonics, master_gain=self.gain,
            attack_frames=self.attack_frames, decay_frames=self.decay_frames,
            sustain_level=self.sustain_level, release_frames=output_tail,
            include_muted=include_muted, maximum_notes=MAX_STEM_NOTES,
        )


@dataclass(frozen=True, slots=True)
class StemPackagePlan:
    patches: tuple[StemPatch, ...]
    sample_rate: int = 24000
    channels: int = 2
    bits: int = 16
    dither_seed: int | None = None
    include_muted: bool = False
    skip_percussion: bool = False
    maximum_samples: int = MAX_TOTAL_PCM_SAMPLES

    def __post_init__(self):
        if (type(self.patches) is not tuple or not 1 <= len(self.patches) <= MAX_STEMS
                or any(not isinstance(x, StemPatch) for x in self.patches)):
            raise MidiWorkstationError("Stem package expects 1..8 typed patch routes")
        ids = tuple(x.track_id for x in self.patches)
        if ids != tuple(sorted(set(ids))):
            raise MidiWorkstationError("Stem patches must use unique sorted track IDs")
        integer(self.sample_rate, "stem sample rate", 8000, 48000)
        integer(self.channels, "stem output channel count", 1, 2)
        if type(self.bits) is not int or self.bits not in (8,16,24,32):
            raise MidiWorkstationError("Stem package uses 8/16/24/32-bit integer PCM")
        if self.dither_seed is not None:
            integer(self.dither_seed, "stem dither seed", 0, 2**32 - 1)
        if type(self.include_muted) is not bool or type(self.skip_percussion) is not bool:
            raise MidiWorkstationError("Stem selection policy must be boolean")
        integer(self.maximum_samples, "stem sample budget", 1, MAX_TOTAL_PCM_SAMPLES)


@dataclass(frozen=True, slots=True)
class StemPreflight:
    source_score_sha256: str
    sample_rate: int
    channels: int
    frames: int
    tail_frames: int
    stems: int
    selected_note_count: int
    skipped_percussion_notes: int
    estimated_pcm_bytes: int
    maximum_pcm_bytes: int
    protocol: str = "sonumio-stem-package-preflight-v1"

    def as_dict(self):
        return {**asdict(self), "audio_rendered": False,
                "trained_model_inference": False, "source_mutated": False}


def preflight_stems(score: Score, plan: StemPackagePlan) -> StemPreflight:
    if not isinstance(score, Score) or not isinstance(plan, StemPackagePlan):
        raise MidiWorkstationError("Expected validated score and stem package plan")
    patches = {p.track_id: p for p in plan.patches}
    all_tracks = {t.id: t for t in score.tracks}
    missing = set(patches) - set(all_tracks)
    if missing:
        raise MidiWorkstationError("Stem patches reference unknown score track IDs")
    active = set(all_tracks) if plan.include_muted else set(score.active_track_ids)
    pitched = {t.id for t in score.tracks if t.pitched and t.id in active}
    if set(patches) != pitched:
        raise MidiWorkstationError(
            "Stem patch list must cover all active pitched tracks exactly"
        )
    percussion = sum(len(t.notes) for t in score.tracks if t.channel == 9 and t.id in active)
    if percussion and not plan.skip_percussion:
        raise MidiWorkstationError("Percussion requires explicit skip_percussion permission")
    count = sum(len(all_tracks[id].notes) for id in pitched)
    if count > MAX_STEM_NOTES:
        raise MidiLimitError("Too many pitched notes for bounded stem export")
    if plan.channels == 1 and any(n.pan != 64 for id in pitched
                                    for n in all_tracks[id].notes):
        raise MidiWorkstationError("Mono stem export requires centered note pan")
    tail = max(p.release_frames for p in plan.patches)
    schedule = build_schedule(score, TransportSettings(
        sample_rate=plan.sample_rate, tail_frames=tail,
        include_muted=plan.include_muted,
    ))
    sample_count = schedule.output_frames * plan.channels * len(patches)
    if sample_count > plan.maximum_samples:
        raise MidiLimitError("Stem package exceeds declared sample budget")
    payload_bytes = sample_count * (plan.bits // 8)
    # Include one 44-byte PCM RIFF header and a conservative ZIP metadata budget.
    predicted = payload_bytes + 1024 * len(patches) + MAX_STEM_MANIFEST_BYTES
    if predicted > MAX_STEM_PACKAGE_BYTES:
        raise MidiLimitError("Stem package exceeds atomic publication byte budget")
    return StemPreflight(score.sha256, plan.sample_rate, plan.channels,
                         schedule.output_frames, tail, len(patches), count,
                         percussion, predicted, MAX_STEM_PACKAGE_BYTES)


def _canon(document: dict) -> bytes:
    try:
        data = json.dumps(document, sort_keys=True, ensure_ascii=True,
                          separators=(",",":"), allow_nan=False).encode("ascii")
    except (TypeError, OverflowError, ValueError) as exc:
        raise MidiFormatError("Stem package manifest is not canonical JSON") from exc
    if len(data) > MAX_STEM_MANIFEST_BYTES:
        raise MidiLimitError("Stem package manifest exceeds budget")
    return data


def _zipmember(name: str, data: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    if type(data) is not bytes or len(data) > MAX_ZIP_ENTRY_BYTES:
        raise MidiLimitError("Stem member exceeds byte budget")
    info = zipfile.ZipInfo(name, date_time=(1980,1,1,0,0,0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info, data


def _verify_pcm(data: bytes, *, rate: int, channels: int,
                frames: int, bits: int) -> None:
    if type(data) is not bytes or len(data) > MAX_STEM_WAV_BYTES:
        raise MidiLimitError("Stem WAV exceeds byte budget")
    try:
        with wave.open(BytesIO(data), "rb") as wav:
            if (wav.getcomptype() != "NONE" or wav.getframerate() != rate
                    or wav.getnchannels() != channels or wav.getnframes() != frames
                    or wav.getsampwidth() != bits//8):
                raise MidiFormatError("Stem WAV format does not match shared clock")
            # Detect maliciously truncated RIFF payload, not just header count.
            samples = wav.readframes(frames + 1)
            if len(samples) != frames * channels * (bits//8):
                raise MidiIntegrityError("Stem PCM data truncated or altered")
    except (wave.Error, EOFError, struct.error) as exc:
        raise MidiFormatError("Malformed integer PCM stem WAV") from exc


def _read_zip(raw: bytes):
    if type(raw) is not bytes or not 100 <= len(raw) <= MAX_STEM_PACKAGE_BYTES:
        raise MidiLimitError("Stem archive outside bounded byte count")
    try:
        with zipfile.ZipFile(BytesIO(raw), "r") as package:
            names = package.namelist()
            if (len(names) < 2 or len(names) > MAX_STEMS+1
                    or len(set(names)) != len(names)
                    or names[0] != "manifest.json"):
                raise MidiFormatError("Invalid canonical stem archive members")
            if any(not _STEM_NAME.fullmatch(x) for x in names[1:]):
                raise MidiFormatError("Unsafe or unexpected stem path in archive")
            if names[1:] != sorted(names[1:]):
                raise MidiFormatError("Stem archive member names are not sorted")
            for info in package.infolist():
                if (info.flag_bits & 1 or info.compress_type != zipfile.ZIP_STORED
                        or info.file_size > MAX_ZIP_ENTRY_BYTES or info.is_dir()
                        or (info.external_attr >> 16) & 0o170000 not in (0,0o100000)):
                    raise MidiFormatError("Unsafe stem archive member attributes")
            manifest = package.read("manifest.json")
            if len(manifest) > MAX_STEM_MANIFEST_BYTES:
                raise MidiLimitError("Oversized stem manifest")
            entries = tuple((name, package.read(name)) for name in names[1:])
    except (OSError, zipfile.BadZipFile, RuntimeError, EOFError, ValueError,
            struct.error, zlib_error_type()) as exc:
        raise MidiFormatError("Malformed or corrupt stem ZIP archive") from exc
    return manifest, entries


def zlib_error_type():
    import zlib
    return zlib.error


def inspect_stem_pack(raw: bytes, *, expected_sha256: str | None = None) -> dict:
    if expected_sha256 is not None:
        sha256(expected_sha256, "expected stem package digest")
        if _sha256(raw).hexdigest() != expected_sha256:
            raise MidiIntegrityError("Stem package digest does not match pinned value")
    manifest_bytes, members = _read_zip(raw)
    try:
        data = json.loads(manifest_bytes.decode("ascii"),
                          object_pairs_hook=_reject_duplicate)
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise MidiFormatError("Invalid stem archive manifest JSON") from exc
    if (type(data) is not dict or set(data) != {"protocol","score_sha256",
         "sample_rate","channels","frames","bits","stems",
         "skipped_percussion_notes","dither_seed","original_synthesis"}
         or data["protocol"] != STEM_PROTOCOL or data["original_synthesis"] is not True):
        raise MidiFormatError("Unknown or incomplete stem package manifest")
    if _canon(data) != manifest_bytes:
        raise MidiFormatError("Stem manifest is not canonical JSON")
    sha256(data["score_sha256"], "source score digest")
    for field, low, high in (("sample_rate",8000,48000),
                             ("channels",1,2),("frames",1,24000*120),
                             ("bits",8,32),("skipped_percussion_notes",0,MAX_STEM_NOTES)):
        integer(data[field],field,low,high)
    if data["bits"] not in (8,16,24,32):
        raise MidiFormatError("Unsupported stem PCM bit depth")
    if data["dither_seed"] is not None:
        integer(data["dither_seed"], "stem dither seed",0,2**32-1)
    rows = data["stems"]
    if type(rows) is not list or len(rows) != len(members) or not 1 <= len(rows) <= MAX_STEMS:
        raise MidiFormatError("Stem manifest entry count does not match archive")
    if data["channels"]*data["frames"]*len(rows) > MAX_TOTAL_PCM_SAMPLES:
        raise MidiLimitError("Stem archive exceeds decoded sample budget")
    for (name, content), row in zip(members,rows):
        if (type(row) is not dict or set(row) != {"track_id", "member", "sha256",
            "notes", "patch", "source_audio_peak", "source_audio_rms", "pcm_bytes"}):
            raise MidiFormatError("Stem manifest entry fields are not recognized")
        if (row["member"] != name or row["track_id"] != _STEM_NAME.fullmatch(name).group(1)
                or type(row["notes"]) is not int or row["notes"] < 0
                or type(row["pcm_bytes"]) is not int or row["pcm_bytes"] != len(content)):
            raise MidiFormatError("Stem archive manifest/member mismatch")
        sha256(row["sha256"], "stem member digest")
        if _sha256(content).hexdigest() != row["sha256"]:
            raise MidiIntegrityError("Stem WAV failed member SHA-256 validation")
        _verify_pcm(content, rate=data["sample_rate"],channels=data["channels"],
                    frames=data["frames"],bits=data["bits"])
        patch = row["patch"]
        if type(patch) is not dict:
            raise MidiFormatError("Stem patch must be a bounded object")
        StemPatch(**patch)
        if row["notes"] > MAX_STEM_NOTES or type(row["source_audio_peak"]) not in (int,float) or type(row["source_audio_rms"]) not in (int,float):
            raise MidiFormatError("Invalid stem report statistics")
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for value in (row["source_audio_peak"],row["source_audio_rms"])):
            raise MidiFormatError("Stem audio statistics must be bounded and finite")
    return {"protocol":"sonumio-stem-package-inspection-v1",
            "package_sha256":_sha256(raw).hexdigest(),
            "source_score_sha256":data["score_sha256"],
            "stems":len(rows), "members":[row["member"] for row in rows],
            "sample_rate":data["sample_rate"],"channels":data["channels"],
            "frames":data["frames"],"bits":data["bits"],
            "original_mathematical_synthesis":True,
            "trained_model_inference":False,"external_samples_used":False}


def _reject_duplicate(pairs):
    out = {}
    for name,value in pairs:
        if name in out:
            raise MidiFormatError("Duplicate stem manifest JSON field")
        out[name]=value
    return out


def _seed(base: int | None, name: str) -> int | None:
    if base is None:
        return None
    return int.from_bytes(_sha256(f"{base}:{name}".encode("ascii")).digest()[:4],"big")


def render_stem_pack(score: Score, plan: StemPackagePlan):
    """Render all selected stems, then validate the completed single ZIP.

    The package is returned in memory. No paths or filesystem side effects
    occur before explicit separate publication.
    """
    budget = preflight_stems(score,plan)
    try:
        from sonumio.audio_core.dither import QuantizationSettings
        from sonumio.audio_core.waveio import encode_wav_bytes
        from sonumio.audio_core.errors import (
            AudioContractError, AudioProcessingError, AudioLimitError, AudioIntegrityError
        )
        from .wavetable_bridge import render_score_wavetable
    except ImportError as exc:
        raise MidiWorkstationError("Sonumio audio_core is required to render stem WAV") from exc
    rows = []
    members = []
    for patch in plan.patches:
        track=score.track_by_id(patch.track_id)
        # Keep original global end, timebase and exact note positions intact.
        subset=replace(score,tracks=(track,))
        bridge=patch.bridge(sample_rate=plan.sample_rate,channels=plan.channels,
                            include_muted=plan.include_muted,
                            output_tail=patch.release_frames)
        try:
            rendered=render_score_wavetable(subset,bridge)
            audio=rendered.audio
            if (audio.frames>budget.frames or audio.channels!=budget.channels
                    or audio.sample_rate!=budget.sample_rate):
                raise MidiIntegrityError("Stem renderer disagrees with shared score clock")
            # Preserve the *individual* patch's release envelope. Zero padding
            # extends the output timeline without falsely lengthening the note.
            if audio.frames < budget.frames:
                from sonumio.audio_core.contracts import make_buffer
                expanded=np.zeros((plan.channels,budget.frames),dtype=np.float32)
                expanded[:,:audio.frames]=audio.samples
                audio=make_buffer(expanded,plan.sample_rate)
            if not math.isfinite(rendered.report.output_peak) or rendered.report.output_peak>1.:
                raise MidiWorkstationError("Stem would clip integer PCM; reduce its patch gain")
            settings=QuantizationSettings(bits=plan.bits,
                      dither=plan.dither_seed is not None,
                      seed=_seed(plan.dither_seed,patch.track_id))
            wav=encode_wav_bytes(audio, settings=settings)
        except (AudioContractError,AudioProcessingError,AudioLimitError,
                AudioIntegrityError) as exc:
            raise MidiWorkstationError(f"Stem synthesis failed: {exc}") from exc
        if len(wav)>MAX_STEM_WAV_BYTES:
            raise MidiLimitError("Rendered stem exceeds per-file WAV limit")
        _verify_pcm(wav,rate=plan.sample_rate,channels=plan.channels,
                    frames=budget.frames,bits=plan.bits)
        path=f"stems/{patch.track_id}.wav"
        rows.append({"track_id":patch.track_id,"member":path,
                     "sha256":_sha256(wav).hexdigest(),
                     "notes":rendered.report.rendered_notes,
                     "patch":asdict(patch),
                     "source_audio_peak":rendered.report.output_peak,
                     "source_audio_rms":rendered.report.output_rms,
                     "pcm_bytes":len(wav)})
        members.append((path,wav))
    manifest = _canon({"protocol":STEM_PROTOCOL,"score_sha256":score.sha256,
                       "sample_rate":plan.sample_rate,"channels":plan.channels,
                       "frames":budget.frames,"bits":plan.bits,
                       "stems":rows,"skipped_percussion_notes":budget.skipped_percussion_notes,
                       "dither_seed":plan.dither_seed,"original_synthesis":True})
    stream=BytesIO()
    with zipfile.ZipFile(stream,"w") as package:
        info,payload=_zipmember("manifest.json",manifest)
        package.writestr(info,payload)
        for path,wav in members:
            info,payload=_zipmember(path,wav)
            package.writestr(info,payload)
    packed=stream.getvalue()
    if len(packed)>MAX_STEM_PACKAGE_BYTES:
        raise MidiLimitError("Finished stem archive exceeds atomic byte cap")
    receipt=inspect_stem_pack(packed)
    if receipt["source_score_sha256"]!=score.sha256:
        raise MidiIntegrityError("Produced archive misidentifies source score")
    return packed,receipt


def publish_stem_pack(score: Score, plan: StemPackagePlan, destination,
                      options: PublicationOptions = PublicationOptions(), *,
                      source_path: str | Path | None = None,
                      protected_paths: tuple[str | Path,...] = ()) -> dict:
    from .publish import _distinct
    if type(protected_paths) is not tuple:
        raise MidiWorkstationError("Protected input paths must be a tuple")
    for source in protected_paths:
        _distinct(source,destination)
    packed,report=render_stem_pack(score,plan)
    for source in protected_paths:
        _distinct(source,destination)
    published=publish_bytes(destination,packed,options,input_path=source_path,
                            expected_extension=".zip")
    return {**report,"publication":published.as_dict(),
            "source_unchanged":True,"remote_publication":False}
