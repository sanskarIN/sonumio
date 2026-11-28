"""Local API contracts for the original procedural music studio.

The API accepts bounded composition and mix parameters, never an arbitrary
file path, uploaded score, external audio, model identifier or code.
The score is re-composed deterministically and optionally pinned by digest.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .arrangement import (
    Arrangement, ArrangementPlan, TRACKS, arrangement_digest, compose_arrangement,
)
from .audio import SAMPLE_RATE
from .errors import GenerationError
from .music_analysis import analyze_score
from .music_midi import encode_midi
from .music_mix import DEFAULT_LEVELS, MixPlan, generate_arrangement_audio
from .music_score_io import encode_score
from .music_note_edits import NoteEdits, apply_note_edits

MAX_PREVIEW_SECONDS = 30
MAX_PREVIEW_FRAMES = SAMPLE_RATE * MAX_PREVIEW_SECONDS
ENGINE_ID = "original-procedural-arrangement-v1"
DISCLAIMER = "Original procedural instrumental synthesis; not a trained AI model"

Track = Literal["keys", "pad", "bass", "lead", "pluck", "kick", "snare", "hat"]
Gain = Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)]
Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]


class StudioModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class MusicCompositionRequest(StudioModel):
    style: Literal["calm", "cinematic", "rhythmic"] = "cinematic"
    root: int = Field(default=60, ge=48, le=71)
    mode: Literal["major", "minor"] = "minor"
    bpm: int = Field(default=120, ge=60, le=160)
    bars: int = Field(default=4, ge=4, le=16)
    seed: int = Field(default=0, ge=0, lt=2**32)
    swing: float = Field(default=0.0, ge=0, le=0.6, allow_inf_nan=False)

    def to_plan(self) -> ArrangementPlan:
        plan = ArrangementPlan(**self.model_dump())
        if plan.frame(plan.ticks) > MAX_PREVIEW_FRAMES:
            raise GenerationError("Music preview must not exceed 30 seconds")
        return plan


class MusicMixRequest(StudioModel):
    gains: list[Gain] = Field(default_factory=lambda: list(DEFAULT_LEVELS),
                              min_length=len(TRACKS), max_length=len(TRACKS))
    mute: list[Track] = Field(default_factory=list, max_length=len(TRACKS))
    solo: list[Track] = Field(default_factory=list, max_length=len(TRACKS))
    room: float = Field(default=0.15, ge=0, le=1, allow_inf_nan=False)
    peak_ceiling: float = Field(default=0.9, ge=0.05, le=0.99, allow_inf_nan=False)

    def to_plan(self) -> MixPlan:
        return MixPlan(gains=tuple(self.gains), mute=tuple(self.mute),
                       solo=tuple(self.solo), room=self.room,
                       peak_ceiling=self.peak_ceiling)


class MusicScoreRequest(StudioModel):
    composition: MusicCompositionRequest
    expected_score_sha256: Digest | None = None
    note_edits: NoteEdits | None = None


class MusicRenderRequest(MusicScoreRequest):
    mix: MusicMixRequest = Field(default_factory=MusicMixRequest)


def prepare_score(request: MusicScoreRequest) -> Arrangement:
    score = compose_arrangement(request.composition.to_plan())
    if request.note_edits is not None:
        score, _ = apply_note_edits(score, request.note_edits)
    digest = arrangement_digest(score)
    if request.expected_score_sha256 is not None and request.expected_score_sha256 != digest:
        raise GenerationError("Score digest mismatch; regenerate the preview")
    return score


def describe_score(score: Arrangement) -> dict:
    analysis = analyze_score(score)
    return {
        "engine_id": ENGINE_ID,
        "learned_model": False,
        "external_recordings_used": False,
        "disclaimer": DISCLAIMER,
        "composition": asdict(score.plan),
        "score_sha256": arrangement_digest(score),
        "sample_rate": SAMPLE_RATE,
        "frames": score.plan.frame(score.plan.ticks),
        "duration_seconds": analysis["duration_seconds"],
        "bars": score.plan.bars,
        "events": analysis["events"],
        "note_events": [asdict(event) for event in score.events],
        "sections": [asdict(section) for section in score.sections],
        "harmony": list(score.harmony),
        "tracks": analysis["tracks"],
        "analysis": analysis,
    }


def export_score(score: Arrangement, format_name: Literal["json", "midi"]):
    if format_name == "json":
        content = encode_score(score)
        media_type = "application/json"
        filename = "sonumio-score.json"
    elif format_name == "midi":
        content = encode_midi(score)
        media_type = "audio/midi"
        filename = "sonumio-score.mid"
    else:
        raise GenerationError("Unsupported score export")
    return content, media_type, {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Sonumio-Score-SHA256": arrangement_digest(score),
        "X-Sonumio-Content-SHA256": sha256(content).hexdigest(),
        "Cache-Control": "no-store",
    }


def render_preview(score: Arrangement, mix: MixPlan):
    """Render only a score that meets the preview budget, including direct callers."""
    if score.plan.frame(score.plan.ticks) > MAX_PREVIEW_FRAMES:
        raise GenerationError("Music preview must not exceed 30 seconds")
    wav, report = generate_arrangement_audio(score, mix)
    if report["score_sha256"] != arrangement_digest(score):
        raise GenerationError("Music renderer returned an inconsistent score")
    if report["wav_sha256"] != sha256(wav).hexdigest():
        raise GenerationError("Music renderer returned an inconsistent WAV digest")
    if (report["frames"] != score.plan.frame(score.plan.ticks)
            or report["sample_rate"] != SAMPLE_RATE or report["channels"] != 2
            or len(wav) != 44 + 4 * report["frames"]):
        raise GenerationError("Music renderer returned inconsistent stereo PCM geometry")
    return wav, {
        "engine_id": ENGINE_ID,
        "model_id": ENGINE_ID,
        "score_sha256": report["score_sha256"],
        "wav_sha256": report["wav_sha256"],
        "duration_seconds": report["frames"] / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "channels": 2,
        "frames": report["frames"],
        "metrics": report["metrics"],
        "track_metrics": report["tracks"],
        "master_attenuation": report["master_attenuation"],
        "learned_model": False,
        "external_recordings_used": False,
        "disclaimer": DISCLAIMER,
    }
