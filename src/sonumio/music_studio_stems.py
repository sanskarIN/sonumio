"""Bounded, score-pinned individual stem auditions through the shared queue."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal

import numpy as np

from .arrangement import Arrangement, TRACKS, arrangement_digest
from .audio import SAMPLE_RATE, encode_wav_channels, metrics_channels
from .errors import GenerationError
from .music_automation import AutomationPlan, generate_automated_music
from .music_mix import render_stems
from .music_stem_pack import PEAK_CEILING, _attenuation, _inspect_stem
from .music_studio import (
    MusicRenderRequest, Digest, Track, ENGINE_ID, DISCLAIMER,
    MAX_PREVIEW_FRAMES, prepare_score, prepare_automation,
)


class MusicStemRequest(MusicRenderRequest):
    # Auditions are always bound to a previously verified score.
    expected_score_sha256: Digest
    track: Track
    audition: Literal["dry", "solo"] = "dry"


@dataclass(frozen=True, slots=True)
class StemRecipe:
    score: Arrangement
    track: str
    audition: str
    automation: AutomationPlan | None = None

    def __post_init__(self):
        if not isinstance(self.score, Arrangement):
            raise GenerationError("Expected audition score")
        if self.track not in TRACKS or self.audition not in ("dry", "solo"):
            raise GenerationError("Invalid stem audition selection")
        if not 1 <= self.score.plan.frame(self.score.plan.ticks) <= MAX_PREVIEW_FRAMES:
            raise GenerationError("Stem audition exceeds 30-second preview budget")
        if self.audition == "dry":
            if self.automation is not None:
                raise GenerationError("Dry auditions do not use a mixer")
        else:
            if not isinstance(self.automation, AutomationPlan):
                raise GenerationError("Expected audition automation")
            self.automation.validate_score(self.score)


def prepare_stem(request):
    if not isinstance(request, MusicStemRequest):
        raise GenerationError("Expected validated stem request")
    score = prepare_score(request)
    # Validate all supplied settings, even when the dry renderer ignores
    # them. A stale automation pin never silently becomes a new recipe.
    original = prepare_automation(request, score)
    return StemRecipe(score, request.track, request.audition,
                      original if request.audition == "solo" else None)


def describe_stem(recipe):
    if not isinstance(recipe, StemRecipe):
        raise GenerationError("Expected StemRecipe")
    effective = None
    if recipe.automation is not None:
        # Audition is an explicit temporary solo. It does not mutate the
        # studio mixer, and overrides original mute/solo selection only.
        effective = replace(recipe.automation,
                            mix=replace(recipe.automation.mix, mute=(), solo=(recipe.track,)))
    frames = recipe.score.plan.frame(recipe.score.plan.ticks)
    return {
        "protocol": "sonumio-studio-stem-audition-v1",
        "engine_id": ENGINE_ID, "model_id": ENGINE_ID,
        "score_sha256": arrangement_digest(recipe.score), "track": recipe.track,
        "audition": recipe.audition, "frames": frames, "sample_rate": SAMPLE_RATE,
        "channels": 2, "duration_seconds": frames / SAMPLE_RATE,
        "automation_sha256": recipe.automation.sha256 if recipe.automation is not None else None,
        "effective_automation_sha256": effective.sha256 if effective is not None else None,
        "static_mix_applied": effective is not None,
        "room_applied": effective is not None and effective.mix.room != 0,
        "mute_solo_overridden": effective is not None,
        "common_dry_attenuation": recipe.audition == "dry",
        "learned_model": False, "external_recordings_used": False,
        "disclaimer": DISCLAIMER,
    }


def render_stem_preview(recipe):
    report = describe_stem(recipe)
    frames = report["frames"]
    if recipe.audition == "dry":
        # Reuse the dry archive's shared attenuation policy so that this
        # WAV is byte-identical to the corresponding verified ZIP member.
        stems = render_stems(recipe.score)
        if (type(stems) is not dict or set(stems) != set(TRACKS)
                or any(not isinstance(stem, np.ndarray) or stem.dtype != np.float32
                       or stem.shape != (2, frames) or not np.isfinite(stem).all()
                       for stem in stems.values())):
            raise GenerationError("Invalid audition dry stems")
        peak = max(float(np.max(np.abs(stem))) for stem in stems.values())
        scale = _attenuation(peak)
        audio = (stems[recipe.track].astype(np.float64)*scale).astype(np.float32)
        wav = encode_wav_channels(audio, SAMPLE_RATE)
        integer_receipt = _inspect_stem(recipe.track, wav, frames)
        report.update({"metrics": metrics_channels(audio), "peak_ceiling": PEAK_CEILING,
                       "master_attenuation": scale, "dry_source_peak": peak,
                       "pcm_receipt": integer_receipt})
    else:
        effective = replace(recipe.automation,
                            mix=replace(recipe.automation.mix, mute=(), solo=(recipe.track,)))
        wav, details = generate_automated_music(recipe.score, effective)
        if (details["automation_sha256"] != report["effective_automation_sha256"]
                or details["score_sha256"] != report["score_sha256"]
                or details["frames"] != frames or details["channels"] != 2
                or details["sample_rate"] != SAMPLE_RATE):
            raise GenerationError("Solo audition receipt differs from recipe")
        report.update({"metrics": details["metrics"],
                       "peak_ceiling": effective.mix.peak_ceiling,
                       "master_attenuation": details["master_attenuation"]})
    if len(wav) != 44 + 4 * frames:
        raise GenerationError("Stem audition WAV geometry mismatch")
    report["wav_sha256"] = sha256(wav).hexdigest()
    return wav, report
