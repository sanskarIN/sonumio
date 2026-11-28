"""Deterministic bounded multi-instrument preview using Sonumio's audio core.

Each selected pitched track receives its own explicit Fourier patch. This
module composes existing wavetable renders; it never reimplements oscillators,
trains models, reads commercial samples or silently claims GM program fidelity.
Percussion needs a separate explicit renderer and is rejected by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (MidiIntegrityError, MidiLimitError, MidiWorkstationError,
                        integer, finite, sha256)
from .score import Score, NOTE_ID_RE
from .multitrack_preflight import build_multitrack_preflight, _peak_polyphony
from .publish import _distinct

PROTOCOL = "sonumio-multitrack-wavetable-preview-v1"
MAX_TRACK_PATCHES = 16
MAX_MULTI_NOTES = 2048
MAX_MIX_FLOAT_BYTES = 48 * 1024 * 1024
MAX_WORK_SAMPLES = 14_000_000
MAX_MIX_POLYPHONY = 64


@dataclass(frozen=True, slots=True)
class TrackPatch:
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
            raise MidiWorkstationError("Patch requires a stable workstation track ID")
        if self.waveform not in ("sine","saw","square","triangle"):
            raise MidiWorkstationError("Unsupported original instrument waveform")
        finite(self.gain,"instrument gain",0.,1.)
        integer(self.table_size,"instrument table size",256,8192)
        if self.table_size & (self.table_size-1):
            raise MidiWorkstationError("Instrument wavetable size must be a power of two")
        integer(self.max_harmonics,"instrument max harmonics",1,256)
        if self.max_harmonics > self.table_size//2 - 1:
            raise MidiWorkstationError("Instrument harmonics cannot exceed Nyquist table")
        for key in ("attack_frames","decay_frames","release_frames"):
            integer(getattr(self,key),key,0,1_000_000)
        finite(self.sustain_level,"instrument sustain",0.,1.)


@dataclass(frozen=True, slots=True)
class MultiTrackPreviewSettings:
    patches: tuple[TrackPatch,...]
    sample_rate: int = 24000
    channels: int = 2
    include_muted: bool = False
    allow_skip_percussion: bool = False
    normalization: str = "reject"
    maximum_peak: float = .98
    maximum_notes: int = MAX_MULTI_NOTES
    maximum_work_samples: int = MAX_WORK_SAMPLES

    def __post_init__(self):
        if type(self.patches) is not tuple or not 1 <= len(self.patches) <= MAX_TRACK_PATCHES:
            raise MidiWorkstationError("Expected 1..16 per-track instrument patches")
        if any(not isinstance(p,TrackPatch) for p in self.patches):
            raise MidiWorkstationError("Every instrument must use TrackPatch")
        ids=[p.track_id for p in self.patches]
        if ids != sorted(set(ids)):
            raise MidiWorkstationError("Instrument patches must be unique and sorted by track ID")
        integer(self.sample_rate,"mix sample rate",8000,192000)
        integer(self.channels,"mix channel count",1,2)
        if type(self.include_muted) is not bool or type(self.allow_skip_percussion) is not bool:
            raise MidiWorkstationError("Multitrack preview policy flags must be bool")
        if self.normalization not in ("reject","peak"):
            raise MidiWorkstationError("Unknown explicit mix normalization policy")
        finite(self.maximum_peak,"maximum preview peak",.01,1.)
        integer(self.maximum_notes,"maximum preview notes",1,MAX_MULTI_NOTES)
        integer(self.maximum_work_samples,"preview work sample budget",1,MAX_WORK_SAMPLES)


@dataclass(frozen=True, slots=True)
class MultiTrackPreviewResult:
    audio: Any
    report: dict


def render_multitrack_preview(
    score: Score, settings: MultiTrackPreviewSettings,
) -> MultiTrackPreviewResult:
    """Render independently patched tracks, combine in deterministic order."""
    preflight = build_multitrack_preflight(score, settings)
    # From this point forward, every expensive allocation corresponds to
    # a bounded resource estimate from the same canonical schedule.
    from sonumio.audio_core.wavetable import (
        WavetablePlan, WavetableNote, VoiceEnvelope, render_wavetable,
    )
    from sonumio.audio_core.contracts import make_buffer
    from sonumio.audio_core.hashing import digest_audio
    selected = preflight.selected_notes
    frames = preflight.output_frames
    channels = settings.channels
    polyphony = preflight.maximum_polyphony
    summed=np.zeros((channels,frames),dtype=np.float64)
    stem_ledger=[]
    for patch in settings.patches:  # canonical sorted order is part of reproducibility contract
        events=[n for n in selected if n.track_id==patch.track_id]
        notes=tuple(WavetableNote(
            start_frame=n.start_frame,gate_frames=n.frames,
            midi_note=n.pitch,velocity=float(n.velocity_fraction),
            pan=float(n.pan_signed),initial_phase=0.,
        ) for n in events)
        plan=WavetablePlan(
            sample_rate=settings.sample_rate,frames=frames,channels=channels,
            waveform=patch.waveform,table_size=patch.table_size,
            max_harmonics=patch.max_harmonics,master_gain=patch.gain,
            envelope=VoiceEnvelope(
                attack_frames=patch.attack_frames,decay_frames=patch.decay_frames,
                sustain_level=patch.sustain_level,release_frames=patch.release_frames,
            ),
        )
        output=render_wavetable(plan,notes)
        if (output.audio.sample_rate!=settings.sample_rate or
                output.audio.frames!=frames or output.audio.channels!=channels):
            raise MidiWorkstationError("Inconsistent per-instrument audio geometry")
        summed+=output.audio.samples.astype(np.float64)
        stem_ledger.append({
            "track_id":patch.track_id,
            "waveform":patch.waveform,
            "patch":asdict(patch),
            "rendered_notes":len(notes),
            "maximum_polyphony":output.report.maximum_polyphony,
            "peak":output.report.peak,
            "rms":output.report.rms,
            "pcm_sha256":digest_audio(output.audio),
        })
    raw_peak=float(np.max(np.abs(summed)))
    applied_scale=1.
    if raw_peak>settings.maximum_peak:
        if settings.normalization=="reject":
            raise MidiWorkstationError("Mix exceeds declared peak; reduce instrument gain")
        applied_scale=settings.maximum_peak/raw_peak
        summed*=applied_scale
    # Normalize and cast exactly once after deterministic float64 mixing.
    audio=make_buffer(summed.astype(np.float32),settings.sample_rate,
                      source_note="Sonumio original per-track finite-Fourier preview")
    final_peak=float(np.max(np.abs(audio.samples)))
    if final_peak>1.:
        raise MidiWorkstationError("Rendered mix unexpectedly exceeds PCM range")
    report={
        "protocol":PROTOCOL,
        "source_score_sha256":score.sha256,
        "sample_rate":settings.sample_rate,
        "channels":channels,"frames":frames,
        "rendered_notes":len(selected),
        "skipped_percussion_notes":preflight.skipped_percussion_notes,
        "tracks_rendered":len(settings.patches),
        "maximum_polyphony":polyphony,
        "mix_peak_before_normalization":raw_peak,
        "mix_peak":final_peak,
        "mix_rms":float(np.sqrt(np.mean(np.square(audio.samples.astype(np.float64))))),
        "normalization":settings.normalization,
        "applied_scale":applied_scale,
        "stems":stem_ledger,
        "pcm_sha256":digest_audio(audio),
        "external_recordings":False,
        "trained_model_inference":False,
        "perceptual_quality_established":False,
        "source_modified":False,
    }
    return MultiTrackPreviewResult(audio,report)


def export_multitrack_preview(
    score: Score,destination: str | Path,settings: MultiTrackPreviewSettings,*,
    source_path: str | Path | None=None,
    protected_paths: tuple[str | Path, ...]=(),
    output_bits: int=16,dither_seed: int | None=None,
    overwrite: bool=False,create_parent: bool=False,
    expected_output_sha256: str | None=None,
) -> dict:
    """Render, quantize and publish complete WAV only after integrity checks."""
    if not isinstance(score, Score) or not isinstance(settings, MultiTrackPreviewSettings):
        raise MidiWorkstationError("Expected immutable Score and multitrack settings")
    integer(output_bits, "PCM preview output bits", 1, 32)
    if output_bits not in (8, 16, 24, 32):
        raise MidiWorkstationError("PCM preview requires 8, 16, 24 or 32 bits")
    if dither_seed is not None:
        integer(dither_seed, "multitrack preview dither seed", 0, 2**32-1)
    if expected_output_sha256 is not None:
        sha256(expected_output_sha256, "expected output WAV digest")
    try:
        from sonumio.audio_core.dither import QuantizationSettings
        from sonumio.audio_core.waveio import encode_wav_bytes
        from sonumio.atomic_cli_wav import publish_generated_wav
        from sonumio.audio_core.errors import (
            AudioIntegrityError, AudioLimitError, AudioContractError,
            AudioProcessingError,
        )
    except ImportError as exc:
        raise MidiWorkstationError("Sonumio audio_core is required for multitrack WAV") from exc

    destination=Path(destination)
    if destination.suffix.lower()!=".wav":
        raise MidiWorkstationError("Multitrack output must end in .wav")
    _distinct(source_path,destination)
    if type(protected_paths) is not tuple:
        raise MidiWorkstationError("Protected source paths must be a tuple")
    for protected in protected_paths:
        _distinct(protected,destination)
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise MidiWorkstationError("Output publication flags must be boolean")
    try:
        q=QuantizationSettings(bits=output_bits,dither=dither_seed is not None,
                               seed=dither_seed)
        result=render_multitrack_preview(score,settings)
        wav=encode_wav_bytes(result.audio,settings=q)
        if len(wav)>64*1024*1024:
            raise MidiLimitError("Multitrack PCM exceeds WAV publication byte budget")
        _distinct(source_path,destination)
        for protected in protected_paths:
            _distinct(protected,destination)
        publication=publish_generated_wav(
            destination,wav,overwrite=overwrite,create_parent=create_parent,
            expected_sha256=expected_output_sha256,
        )
    except AudioIntegrityError as exc:
        raise MidiIntegrityError(f"Generated WAV integrity check: {exc}") from exc
    except AudioLimitError as exc:
        raise MidiLimitError(f"Multitrack audio resource budget: {exc}") from exc
    except (AudioContractError, AudioProcessingError) as exc:
        raise MidiWorkstationError(f"Multitrack audio publication: {exc}") from exc
    return {
        "protocol":"sonumio-multitrack-preview-publication-v1",
        "render":result.report,
        "wav_sha256":publication.wav_sha256,
        "wav_bytes":publication.bytes_written,
        "output_bits":output_bits,
        "dither_seed":dither_seed,
    }
