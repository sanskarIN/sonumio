"""Research-only hybrid score rendering using an in-memory harmonic model.

The original score, note clock, pan, percussion and mix remain unchanged.
Learned synthesis is used only within the model's documented control
domain. Unsupported notes are rendered by the existing procedural engine.
No model loading, checkpoint publication or external audio is involved.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch

from research.instrument_note_data import (
    INSTRUMENTS, MIN_PITCH, MAX_PITCH, MIN_VELOCITY, MAX_VELOCITY,
    MIN_FRAMES, MAX_FRAMES, NoteSpec,
)
from research.instrument_harmonic_model import HarmonicInstrument, fixed_sine
from research.weight_ema import model_state_sha256
from sonumio.arrangement import Arrangement, Event, TRACKS, arrangement_digest
from sonumio.audio import SAMPLE_RATE, encode_wav_channels
from sonumio.music_instruments import render_event
from sonumio.music_mix import MixPlan, _render_stems, mix_stems

PROTOCOL = "sonumio-research-hybrid-score-v1"
MAX_SCORE_FRAMES = 30 * SAMPLE_RATE
MAX_SCORE_EVENTS = 512
MAX_EVENT_SAMPLES = 3_000_000
ARMS = ("procedural", "fixed_sine", "matched", "wrong_instrument")
REASONS = ("percussion", "pitch", "duration", "velocity", "training_overlap")
EPS = 1e-12


@dataclass(frozen=True, slots=True)
class NoteRoute:
    event: Event
    start: int
    stop: int
    spec: NoteSpec | None
    fallback: str | None

    def __post_init__(self):
        if not isinstance(self.event, Event):
            raise ValueError("Expected validated score event")
        if (type(self.start) is not int or type(self.stop) is not int
                or not 0 <= self.start < self.stop <= MAX_SCORE_FRAMES):
            raise ValueError("Invalid score note frame span")
        if self.spec is None:
            if self.fallback not in REASONS:
                raise ValueError("Missing explicit fallback reason")
        elif (not isinstance(self.spec, NoteSpec) or self.fallback is not None
                or self.spec.instrument != self.event.track
                or self.spec.pitch != self.event.pitch
                or self.spec.frames != self.stop - self.start
                or self.spec.velocity_milli != round(self.event.velocity * 1000)):
            raise ValueError("Inconsistent learned score note route")

    @property
    def frames(self):
        return self.stop - self.start


def route_event(score, event, excluded):
    if not isinstance(score, Arrangement) or not isinstance(event, Event):
        raise ValueError("Expected arrangement and score event")
    start = score.plan.frame(event.tick)
    stop = score.plan.frame(event.tick + event.length)
    frames = stop - start
    velocity = round(event.velocity * 1000)
    spec = None
    if event.track not in INSTRUMENTS:
        reason = "percussion"
    elif not MIN_PITCH <= event.pitch <= MAX_PITCH:
        reason = "pitch"
    elif not MIN_FRAMES <= frames <= MAX_FRAMES:
        reason = "duration"
    elif not MIN_VELOCITY <= velocity <= MAX_VELOCITY:
        reason = "velocity"
    else:
        candidate = NoteSpec(event.track, event.pitch, velocity, frames)
        if candidate.identity in excluded:
            reason = "training_overlap"
        else:
            spec, reason = candidate, None
    return NoteRoute(event, start, stop, spec, reason)


def _validated_exclusions(excluded):
    if (type(excluded) is not frozenset or len(excluded) > 1024
            or any(type(value) is not str or len(value) != 64
                   or any(c not in "0123456789abcdef" for c in value)
                   for value in excluded)):
        raise ValueError("Expected bounded note identity exclusion set")
    return excluded


def route_score(score, *, excluded=frozenset()):
    _validated_exclusions(excluded)
    if not isinstance(score, Arrangement):
        raise ValueError("Expected validated arrangement")
    frames = score.plan.frame(score.plan.ticks)
    if not 1 <= frames <= MAX_SCORE_FRAMES or len(score.events) > MAX_SCORE_EVENTS:
        raise ValueError("Hybrid score exceeds preview event or duration budget")
    routes = tuple(route_event(score, event, excluded) for event in score.events)
    if sum(route.frames for route in routes) > MAX_EVENT_SAMPLES:
        raise ValueError("Hybrid score exceeds note sample budget")
    return routes


def describe_routes(routes):
    if type(routes) is not tuple or not routes or any(not isinstance(r, NoteRoute) for r in routes):
        raise ValueError("Expected canonical score routes")
    by_track = {}
    for track in TRACKS:
        items = [r for r in routes if r.event.track == track]
        eligible = [r for r in items if r.spec is not None]
        by_track[track] = {
            "events": len(items),
            "eligible_events": len(eligible),
            "event_samples": sum(r.frames for r in items),
            "eligible_samples": sum(r.frames for r in eligible),
            "fallback_reasons": dict(sorted(Counter(r.fallback for r in items if r.fallback).items())),
        }
    return {
        "events": len(routes),
        "eligible_events": sum(r.spec is not None for r in routes),
        "event_samples": sum(r.frames for r in routes),
        "eligible_samples": sum(r.frames for r in routes if r.spec is not None),
        "fallback_reasons": dict(sorted(Counter(r.fallback for r in routes if r.fallback).items())),
        "tracks": by_track,
    }


@torch.no_grad()
def _synthesize(route, model, arm, offset):
    spec = route.spec
    if spec is None:
        raise ValueError("Cannot synthesize a fallback note with learned model")
    index = spec.instrument_id
    if arm == "wrong_instrument":
        index = (index + offset) % len(INSTRUMENTS)
    label = torch.tensor([index], dtype=torch.long)
    pitch = torch.tensor([spec.pitch], dtype=torch.long)
    velocity = torch.tensor([spec.velocity_milli], dtype=torch.long)
    wave = (fixed_sine(label, pitch, velocity, spec.frames) if arm == "fixed_sine"
            else model(label, pitch, velocity, spec.frames))
    if (not isinstance(wave, torch.Tensor) or wave.shape != (1, spec.frames)
            or wave.device.type != "cpu" or wave.dtype != torch.float32
            or not bool(torch.isfinite(wave).all())):
        raise ValueError("Invalid hybrid note waveform")
    # Recover exact score articulation rather than rounding its final gain.
    # Only the conditioning velocity remains quantized to one thousandth.
    signal = wave[0].numpy().astype(np.float64)
    signal *= route.event.velocity / spec.velocity
    if np.max(np.abs(signal)) > 1.000001:
        raise ValueError("Hybrid note exceeds bounded waveform range")
    return signal


def render_hybrid_stems(score, *, arm, model=None, excluded=frozenset(), wrong_offset=1):
    if type(arm) is not str or arm not in ARMS:
        raise ValueError("Unknown hybrid renderer arm")
    if arm in ("matched", "wrong_instrument"):
        if not isinstance(model, HarmonicInstrument):
            raise ValueError("Expected in-memory harmonic instrument model")
    elif model is not None:
        raise ValueError("Procedural and sine controls do not accept a model")
    if (type(wrong_offset) is not int or not 1 <= wrong_offset < len(INSTRUMENTS)):
        raise ValueError("Invalid wrong instrument offset")

    routes = route_score(score, excluded=excluded)
    coverage = describe_routes(routes)
    index = 0
    before = model_state_sha256(model) if model is not None else None
    previous_mode = model.training if model is not None else None
    if model is not None:
        model.eval()

    def renderer(event, frames, seed):
        nonlocal index
        if index >= len(routes):
            raise ValueError("Unexpected additional score event")
        route = routes[index]
        index += 1
        if event != route.event or frames != route.frames or seed != score.plan.seed:
            raise ValueError("Score event routing changed during rendering")
        if arm == "procedural" or route.spec is None:
            return render_event(event, frames, seed)
        return _synthesize(route, model, arm, wrong_offset)

    try:
        stems = _render_stems(score, renderer)
    finally:
        if model is not None:
            model.train(previous_mode)
    if index != len(routes):
        raise ValueError("Incomplete hybrid score event traversal")
    if model is not None and before != model_state_sha256(model):
        raise ValueError("Hybrid rendering mutated neural model state")
    coverage["substituted_events"] = 0 if arm == "procedural" else coverage["eligible_events"]
    coverage["substituted_samples"] = 0 if arm == "procedural" else coverage["eligible_samples"]
    coverage["procedural_events"] = coverage["events"] - coverage["substituted_events"]
    coverage["procedural_samples"] = coverage["event_samples"] - coverage["substituted_samples"]
    coverage["arm"] = arm
    coverage["wrong_offset"] = wrong_offset if arm == "wrong_instrument" else None
    coverage["model_sha256"] = before
    coverage["score_sha256"] = arrangement_digest(score)
    return stems, coverage


def _pcm_stats(candidate, reference):
    if (not isinstance(candidate, np.ndarray) or not isinstance(reference, np.ndarray)
            or candidate.dtype != np.float32 or reference.dtype != np.float32
            or candidate.shape != reference.shape or candidate.ndim != 2
            or candidate.shape[0] != 2 or not 1 <= candidate.shape[1] <= MAX_SCORE_FRAMES
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError("Invalid paired stereo score PCM")
    error = ref_energy = candidate_energy = dot = 0.
    peak = 0.
    for start in range(0, candidate.shape[1], 16384):
        a = candidate[:, start:start+16384].astype(np.float64)
        b = reference[:, start:start+16384].astype(np.float64)
        error += float(np.sum((a-b)**2))
        ref_energy += float(np.sum(b*b))
        candidate_energy += float(np.sum(a*a))
        dot += float(np.sum(a*b))
        peak = max(peak, float(np.max(np.abs(a))))
    n = candidate.size
    return {
        "mse": error / n,
        "relative_wave_mse": error / ref_energy if ref_energy > EPS else None,
        "rms_ratio": math.sqrt(candidate_energy / ref_energy) if ref_energy > EPS else None,
        "waveform_cosine": max(-1., min(1., dot / math.sqrt(candidate_energy * ref_energy)))
        if candidate_energy > EPS and ref_energy > EPS else None,
        "peak": peak,
        "exact_pcm_match": np.array_equal(candidate, reference),
    }


def compare_stems(candidate, reference, coverage):
    if (type(candidate) is not dict or type(reference) is not dict
            or set(candidate) != set(TRACKS) or set(reference) != set(TRACKS)):
        raise ValueError("Expected eight paired score stems")
    report = {}
    for track in TRACKS:
        stats = _pcm_stats(candidate[track], reference[track])
        substituted = coverage["tracks"][track]["eligible_events"] > 0 and coverage["arm"] != "procedural"
        if not substituted and not stats["exact_pcm_match"]:
            raise ValueError("Hybrid renderer changed a fallback-only track")
        report[track] = {**stats, "substituted": substituted}
    return report


def compare_hybrid_score(score, *, arm, model=None, excluded=frozenset(), wrong_offset=1, mix=None):
    """Compare one whole score to its exact original procedural reference."""
    mix = MixPlan() if mix is None else mix
    if not isinstance(mix, MixPlan):
        raise ValueError("Expected original MixPlan")
    reference, baseline = render_hybrid_stems(score, arm="procedural", excluded=excluded)
    candidate, coverage = render_hybrid_stems(score, arm=arm, model=model,
                                              excluded=excluded, wrong_offset=wrong_offset)
    if baseline["score_sha256"] != coverage["score_sha256"]:
        raise ValueError("Paired scores changed identity")
    by_track = compare_stems(candidate, reference, coverage)
    reference_audio, reference_mix = mix_stems(reference, mix)
    candidate_audio, candidate_mix = mix_stems(candidate, mix)
    reference_wav = encode_wav_channels(reference_audio, SAMPLE_RATE)
    candidate_wav = encode_wav_channels(candidate_audio, SAMPLE_RATE)
    return {
        "protocol": PROTOCOL, "routing": coverage, "track_comparisons": by_track,
        "mix_comparison": _pcm_stats(candidate_audio, reference_audio),
        "reference_wav_sha256": hashlib.sha256(reference_wav).hexdigest(),
        "candidate_wav_sha256": hashlib.sha256(candidate_wav).hexdigest(),
        "reference_master_attenuation": reference_mix["master_attenuation"],
        "candidate_master_attenuation": candidate_mix["master_attenuation"],
        "frames": score.plan.frame(score.plan.ticks), "sample_rate": SAMPLE_RATE,
        "learned_events_used": arm in ("matched", "wrong_instrument") and coverage["substituted_events"] > 0,
        "external_recordings_used": False, "perceptual_quality_established": False,
        "generated_audio_written": False, "model_weights_published": False,
    }
