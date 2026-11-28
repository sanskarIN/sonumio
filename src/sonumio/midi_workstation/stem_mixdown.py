"""Offline, phase-aligned, explicit-gain mixdown of validated Sonumio stems.

The pack's exact clock, dimensions and SHA-256 ledger are authenticated
*before* any PCM is interpreted. Decoding uses standard-library PCM WAV and
bounded NumPy arrays, never a dynamic plugin or imported script. Track gains
are an explicit complete set; no track can be silently omitted from a mix.

A mixdown is a new original mathematical audio artifact. It is not evidence
of trained model fidelity, studio mastering or perceptual quality.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256 as _sha256
from io import BytesIO
import json
import math
from pathlib import Path
import wave

import numpy as np

from .contracts import MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError, integer, sha256
from .score import NOTE_ID_RE
from .stem_package import _read_zip, inspect_stem_pack, MAX_TOTAL_PCM_SAMPLES
from .publish import _distinct

MIXDOWN_PROTOCOL="sonumio-stem-package-offline-mixdown-v1"
MAX_OUTPUT_PEAK=1.0


@dataclass(frozen=True, slots=True)
class StemMixGain:
    track_id: str
    gain_milli: int = 1000

    def __post_init__(self):
        if type(self.track_id) is not str or not NOTE_ID_RE.fullmatch(self.track_id):
            raise MidiWorkstationError("Mixdown gain requires stable score track identifier")
        integer(self.gain_milli,"mixdown track gain milli",0,2000)


@dataclass(frozen=True, slots=True)
class StemMixSettings:
    gains: tuple[StemMixGain, ...]
    master_milli: int = 1000
    peak_policy: str = "reject"
    target_peak_milli: int = 980
    output_bits: int = 16
    dither_seed: int | None = None
    maximum_samples: int = MAX_TOTAL_PCM_SAMPLES

    def __post_init__(self):
        if (type(self.gains) is not tuple or not 1 <= len(self.gains) <= 8
                or any(not isinstance(g,StemMixGain) for g in self.gains)):
            raise MidiWorkstationError("Mixdown requires 1..8 typed gain rows")
        ids=tuple(g.track_id for g in self.gains)
        if ids != tuple(sorted(set(ids))):
            raise MidiWorkstationError("Mixdown track gains must be unique and sorted")
        integer(self.master_milli,"master gain milli",0,2000)
        if self.peak_policy not in ("reject","normalize"):
            raise MidiWorkstationError("Unknown explicit mixdown peak handling policy")
        integer(self.target_peak_milli,"mix target peak milli",1,1000)
        if type(self.output_bits) is not int or self.output_bits not in (8,16,24,32):
            raise MidiWorkstationError("Mixdown output must use 8/16/24/32-bit integer PCM")
        if self.dither_seed is not None:
            integer(self.dither_seed,"mixdown dither seed",0,2**32-1)
        integer(self.maximum_samples,"mixdown output sample budget",1,MAX_TOTAL_PCM_SAMPLES)


@dataclass(frozen=True, slots=True)
class StemMixReceipt:
    source_stem_package_sha256: str
    source_score_sha256: str
    frames: int
    sample_rate: int
    channels: int
    input_bits: int
    output_bits: int
    track_count: int
    selected_track_count: int
    pre_peak: float
    post_peak: float
    applied_normalization: float
    resulting_float_sha256: str
    peak_policy: str
    protocol: str = MIXDOWN_PROTOCOL

    def as_dict(self):
        return {**asdict(self),"original_mathematical_synthesis":True,
                "trained_model_inference":False,"external_samples_used":False,
                "perceptual_quality_established":False,
                "source_archive_modified":False}


@dataclass(frozen=True,slots=True)
class StemMixResult:
    samples: np.ndarray
    report: StemMixReceipt

    def __post_init__(self):
        if not isinstance(self.samples,np.ndarray) or self.samples.ndim!=2:
            raise MidiWorkstationError("Mix result requires channel-first PCM array")
        if self.samples.dtype!=np.float32 or self.samples.flags.writeable:
            raise MidiWorkstationError("Mix result must be read-only float32")
        if not isinstance(self.report,StemMixReceipt):
            raise MidiWorkstationError("Mix result requires verified provenance report")


def _decode_pcm_integer(data: bytes, *, bits: int, channels: int, frames: int) -> np.ndarray:
    """Return channel-first float64 samples from signed integer WAV PCM.

    8-bit PCM is unsigned by the RIFF specification; 24-bit PCM is manually
    sign-extended rather than relying on a non-portable dtype.
    """
    if type(data) is not bytes or type(bits) is not int or bits not in (8,16,24,32):
        raise MidiFormatError("Expected bounded integer-PCM WAV content")
    integer(channels,"mixdown PCM channel count",1,2)
    integer(frames,"mixdown PCM frames",1,MAX_TOTAL_PCM_SAMPLES)
    frame_bytes=channels * (bits//8)
    try:
        with wave.open(BytesIO(data),"rb") as wav:
            if (wav.getcomptype()!="NONE" or wav.getnchannels()!=channels
                    or wav.getnframes()!=frames or wav.getsampwidth()!=bits//8):
                raise MidiFormatError("Stem WAV geometry disagrees with manifest")
            raw=wav.readframes(frames+1)
    except (wave.Error,EOFError,ValueError) as exc:
        raise MidiFormatError("Invalid integer PCM source in stem mixdown") from exc
    if len(raw)!=frames*frame_bytes:
        raise MidiIntegrityError("Truncated stem PCM payload")
    if bits==8:
        normalized=(np.frombuffer(raw,dtype=np.uint8).astype(np.float64)-128.)/128.
    elif bits==16:
        normalized=np.frombuffer(raw,dtype="<i2").astype(np.float64)/32768.
    elif bits==32:
        normalized=np.frombuffer(raw,dtype="<i4").astype(np.float64)/2147483648.
    else:
        values=np.frombuffer(raw,dtype=np.uint8).reshape((-1,3))
        word=(values[:,0].astype(np.int32)
              | (values[:,1].astype(np.int32)<<8)
              | (values[:,2].astype(np.int32)<<16))
        sign=np.where((word & 0x800000)!=0,word - 0x1000000,word)
        normalized=sign.astype(np.float64)/8388608.
    if normalized.size!=frames*channels or not np.isfinite(normalized).all():
        raise MidiIntegrityError("Decoded integer PCM violates finite shape contract")
    return normalized.reshape((frames,channels)).T


def mix_stem_package(raw: bytes,settings: StemMixSettings) -> StemMixResult:
    if not isinstance(settings,StemMixSettings):
        raise MidiWorkstationError("Expected bounded stem mix settings")
    inspection=inspect_stem_pack(raw)
    archive_manifest,members=_read_zip(raw)
    metadata=json.loads(archive_manifest.decode("ascii"))
    names=tuple(row["track_id"] for row in metadata["stems"])
    if names!=tuple(g.track_id for g in settings.gains):
        raise MidiWorkstationError("Mixdown gains must address every stem in canonical order")
    frames=inspection["frames"]
    channels=inspection["channels"]
    rate=inspection["sample_rate"]
    bits=inspection["bits"]
    samples=frames*channels
    if samples>settings.maximum_samples:
        raise MidiLimitError("Mixdown output exceeds requested sample budget")
    # Explicit float64 accumulator minimizes track-summation error; only one
    # decoded stem is live in memory at a time.
    master=np.zeros((channels,frames),dtype=np.float64)
    selected=0
    for (_,wav),route in zip(members,settings.gains):
        if route.gain_milli:
            selected+=1
            contribution=_decode_pcm_integer(wav,bits=bits,channels=channels,frames=frames)
            master+=(route.gain_milli/1000.)*contribution
    master *= settings.master_milli/1000.
    if not np.isfinite(master).all():
        raise MidiIntegrityError("Mixdown accumulated nonfinite PCM")
    peak=float(np.max(np.abs(master)))
    target=settings.target_peak_milli/1000.
    multiplier=1.
    if peak>target:
        if settings.peak_policy=="reject":
            raise MidiWorkstationError("Mix exceeds configured peak; reduce gain or select normalize")
        multiplier=target/peak
        master*=multiplier
    post=float(np.max(np.abs(master)))
    if post>1.:
        raise MidiIntegrityError("Mix unexpectedly exceeds float-to-PCM range")
    float_pcm=master.astype(np.float32)
    # Copy into immutable byte-backed data; do not trust writable owner flags.
    frozen=np.frombuffer(float_pcm.tobytes(order="C"),dtype=np.float32)
    frozen=frozen.reshape((channels,frames))
    report=StemMixReceipt(
        inspection["package_sha256"],inspection["source_score_sha256"],
        frames,rate,channels,bits,settings.output_bits,len(names),selected,
        peak,post,multiplier,_sha256(frozen.tobytes(order="C")).hexdigest(),
        settings.peak_policy,
    )
    return StemMixResult(frozen,report)


def publish_stem_mixdown(raw: bytes,settings: StemMixSettings,destination: str | Path, *,
                         source_path: str | Path | None = None,
                         overwrite: bool = False,create_parent: bool = False,
                         expected_output_sha256: str | None = None) -> dict:
    """Publish one SHA-verified PCM mix via the existing WAV atomic writer."""
    if not isinstance(settings,StemMixSettings):
        raise MidiWorkstationError("Expected StemMixSettings")
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise MidiWorkstationError("Mix publication flags must be boolean")
    if expected_output_sha256 is not None:
        sha256(expected_output_sha256,"expected mixdown output digest")
    output=Path(destination)
    if output.suffix.lower()!=".wav":
        raise MidiFormatError("Mixdown output requires a .wav extension")
    _distinct(source_path,output)
    result=mix_stem_package(raw,settings)
    try:
        from sonumio.audio_core.contracts import make_buffer
        from sonumio.audio_core.dither import QuantizationSettings
        from sonumio.audio_core.waveio import encode_wav_bytes
        from sonumio.atomic_cli_wav import publish_generated_wav
        from sonumio.audio_core.errors import (AudioContractError,AudioIntegrityError,
                                                AudioLimitError,AudioProcessingError)
    except ImportError as exc:
        raise MidiWorkstationError("Sonumio audio_core is required to publish PCM mix") from exc
    try:
        audio=make_buffer(result.samples,result.report.sample_rate)
        q=QuantizationSettings(bits=settings.output_bits,
                 dither=settings.dither_seed is not None,seed=settings.dither_seed)
        content=encode_wav_bytes(audio,settings=q)
        _distinct(source_path,output)
        publication=publish_generated_wav(output,content,overwrite=overwrite,
                   create_parent=create_parent,expected_sha256=expected_output_sha256)
    except (AudioContractError,AudioIntegrityError,AudioLimitError,AudioProcessingError) as exc:
        raise MidiWorkstationError(f"Mixdown publication failed: {exc}") from exc
    return {"protocol":"sonumio-stem-package-mix-publication-v1",
            "mix":result.report.as_dict(),
            "wav_sha256":publication.wav_sha256,
            "wav_bytes":publication.bytes_written,
            "source_archive_modified":False,
            "remote_publication":False,
            "model_inference":False}
