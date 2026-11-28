"""In-memory multi-model arrangement rendering and paired diagnostics."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import statistics

import numpy as np
import torch

from research.instrument_harmonic_model import HarmonicInstrument, fixed_sine
from research.instrument_long_model import WindowedHarmonicInstrument, render_complete, sine_window
from research.percussion_model import PercussionModel, broadband_control, diagnostics, feature_errors
from research.instrument_note_data import INSTRUMENTS
from research.percussion_data import DRUMS
from research.multimodel_music_routes import (
    FAMILIES, Exclusions, route_score, describe_routes, event_excitation,
)
from research.hybrid_music_render import _pcm_stats
from research.weight_ema import model_state_sha256
from sonumio.arrangement import TRACKS, arrangement_digest
from sonumio.music_instruments import render_event
from sonumio.music_mix import MixPlan, _render_stems, mix_stems
from sonumio.audio import SAMPLE_RATE, encode_wav_channels

PROTOCOL = "sonumio-multimodel-music-v1"
ARMS = ("reference", "generic", "learned")
METRIC_BLOCK = 4096


@dataclass(frozen=True, slots=True)
class Models:
    short: HarmonicInstrument
    sustained: WindowedHarmonicInstrument
    percussion: PercussionModel

    def __post_init__(self):
        if (not isinstance(self.short, HarmonicInstrument)
                or isinstance(self.short, WindowedHarmonicInstrument)
                or not isinstance(self.sustained, WindowedHarmonicInstrument)
                or not isinstance(self.percussion, PercussionModel)):
            raise ValueError("Invalid multi-model synthesis bundle")

    def states(self):
        return {family: model_state_sha256(getattr(self, family)) for family in FAMILIES}


@dataclass(frozen=True, slots=True)
class RenderPlan:
    arm: str = "reference"
    families: tuple[str, ...] = FAMILIES
    replica: int = 0
    pitched_offset: int = 0
    drum_offset: int = 0

    def __post_init__(self):
        if type(self.arm) is not str or self.arm not in ARMS:
            raise ValueError("Unknown multi-model rendering arm")
        if (type(self.families) is not tuple or not self.families
                or self.families != tuple(f for f in FAMILIES if f in self.families)):
            raise ValueError("Expected canonical enabled synthesis families")
        if (type(self.replica) is not int or not 0 <= self.replica < 2**32
                or type(self.pitched_offset) is not int or not 0 <= self.pitched_offset < len(INSTRUMENTS)
                or type(self.drum_offset) is not int or not 0 <= self.drum_offset < len(DRUMS)):
            raise ValueError("Invalid multi-model replica or label offset")
        if self.arm != "learned" and (self.pitched_offset or self.drum_offset):
            raise ValueError("Only learned controls may change instrument labels")


@contextmanager
def _evaluation(models):
    if models is None:
        yield None
        return
    before = models.states()
    modes = {family: getattr(models, family).training for family in FAMILIES}
    try:
        for family in FAMILIES:
            getattr(models, family).eval()
        yield before
    finally:
        for family in FAMILIES:
            getattr(models, family).train(modes[family])
        if models.states() != before:
            raise ValueError("Multi-model inference mutated model state")


@torch.no_grad()
def _render_note(route, plan, models):
    event = route.event
    family = route.family
    v = torch.tensor([route.velocity_milli], dtype=torch.long)
    pitch = torch.tensor([event.pitch], dtype=torch.long)
    n = route.frames
    if family == "percussion":
        index = (DRUMS.index(event.track) + plan.drum_offset) % len(DRUMS)
        labels = torch.tensor([index], dtype=torch.long)
        noise = torch.from_numpy(event_excitation(route, plan.replica))[None,:]
        output = (broadband_control(labels, v, n, noise) if plan.arm == "generic"
                  else models.percussion(labels, v, n, noise))
    else:
        index = (INSTRUMENTS.index(event.track) + plan.pitched_offset) % len(INSTRUMENTS)
        labels = torch.tensor([index], dtype=torch.long)
        if family == "short":
            output = (fixed_sine(labels, pitch, v, n) if plan.arm == "generic"
                      else models.short(labels, pitch, v, n))
        elif family == "sustained":
            if plan.arm == "generic":
                blocks = [sine_window(labels, pitch, v, n, start, min(4096,n-start))
                          for start in range(0,n,4096)]
            else:
                blocks = [block for _,block in render_complete(models.sustained,labels,pitch,v,n)]
            output = torch.cat(blocks, dim=1)
        else:
            raise ValueError("Cannot synthesize unsupported route")
    if (not isinstance(output,torch.Tensor) or output.device.type != "cpu"
            or output.dtype != torch.float32 or output.shape != (1,n)
            or not bool(torch.isfinite(output).all())):
        raise ValueError("Invalid multi-model note waveform")
    # Preserve original articulation amplitude despite millivelocity rounding.
    signal = output[0].numpy().astype(np.float64) * (event.velocity / (route.velocity_milli/1000.))
    if np.max(np.abs(signal)) > 1.000001:
        raise ValueError("Multi-model note exceeds amplitude bound")
    return signal


def _pitched_error(candidate, target):
    denominator = float(np.sum(target**2))
    if denominator <= 1e-12:
        raise ValueError("Silent pitched reference")
    return {"relative_wave_mse": float(np.sum((candidate-target)**2))/denominator}


def _drum_error(candidate, target):
    a = torch.from_numpy(candidate.astype(np.float32))[None,:]
    b = torch.from_numpy(target.astype(np.float32))[None,:]
    return {key:float(value[0]) for key,value in diagnostics(a,b).items()}


def _summarize_notes(notes):
    result = {}
    for family in FAMILIES:
        rows = [row for row in notes if row["family"] == family]
        tracks = {}
        for track in TRACKS:
            selected = [row["metrics"] for row in rows if row["track"] == track]
            if selected:
                tracks[track] = {"notes":len(selected),
                    "metrics":{key:statistics.fmean(x[key] for x in selected) for key in selected[0]}}
        if rows:
            macro = {key:statistics.fmean(x["metrics"][key] for x in tracks.values())
                     for key in rows[0]["metrics"]}
        else:
            macro = None
        result[family] = {"notes":len(rows),"by_track":tracks,"track_macro":macro}
    return result


def render_score(score, plan=None, *, models=None, exclusions=None):
    plan = RenderPlan() if plan is None else plan
    exclusions = Exclusions() if exclusions is None else exclusions
    if not isinstance(plan, RenderPlan):
        raise ValueError("Expected multi-model RenderPlan")
    if (plan.arm == "learned" and not isinstance(models,Models)
            or plan.arm != "learned" and models is not None):
        raise ValueError("Model bundle must match the selected rendering arm")
    routes = route_score(score, exclusions)
    coverage = describe_routes(routes)
    used = dict.fromkeys(FAMILIES,0)
    used_samples = dict.fromkeys(FAMILIES,0)
    substituted_tracks = dict.fromkeys(TRACKS,0)
    notes = []
    cursor = 0

    def renderer(event, frames, seed):
        nonlocal cursor
        if cursor >= len(routes):
            raise ValueError("Unexpected extra multi-model event")
        route = routes[cursor]
        cursor += 1
        if route.event != event or route.frames != frames or seed != score.plan.seed:
            raise ValueError("Multi-model event traversal changed")
        original = render_event(event,frames,seed)
        if plan.arm == "reference" or route.family not in plan.families:
            return original
        candidate = _render_note(route,plan,models)
        used[route.family] += 1
        used_samples[route.family] += frames
        substituted_tracks[event.track] += 1
        metrics = (_drum_error(candidate,original) if route.family == "percussion"
                   else _pitched_error(candidate,original))
        notes.append({"event_sha256":route.identity,"track":event.track,
                      "family":route.family,"frames":frames,"metrics":metrics})
        return candidate

    with _evaluation(models) as state:
        stems = _render_stems(score,renderer)
    if cursor != len(routes) or len(notes) != sum(used.values()):
        raise ValueError("Incomplete multi-model render traversal")
    return stems, {
        "protocol":PROTOCOL,"score_sha256":arrangement_digest(score),"arm":plan.arm,
        "eligible":coverage,"substituted_events":used,"substituted_samples":used_samples,
        "substituted_by_track":substituted_tracks,"procedural_events":len(routes)-sum(used.values()),
        "procedural_samples":coverage["event_samples"]-sum(used_samples.values()),
        "model_sha256":state,"replica":plan.replica,
        "pitched_offset":plan.pitched_offset,"drum_offset":plan.drum_offset,
        "enabled_families":plan.families,"notes":notes,"note_diagnostics":_summarize_notes(notes),
    }


@torch.no_grad()
def mix_feature_comparison(candidate, reference):
    """Phase-insensitive full-mix features over bounded stereo windows.

    Both channels are evaluated separately. The final block is zero-padded
    equally for both arms and weighted by its actual frame count.
    """
    if (not isinstance(candidate,np.ndarray) or not isinstance(reference,np.ndarray)
            or candidate.shape != reference.shape or candidate.ndim != 2
            or candidate.shape[0] != 2 or candidate.shape[1] == 0
            or candidate.dtype != np.float32 or reference.dtype != np.float32
            or not np.isfinite(candidate).all() or not np.isfinite(reference).all()):
        raise ValueError("Invalid paired full-mix features")
    totals = {}
    frames = candidate.shape[1]
    for start in range(0,frames,METRIC_BLOCK):
        count = min(METRIC_BLOCK,frames-start)
        a = np.zeros((2,METRIC_BLOCK),dtype=np.float32)
        b = np.zeros_like(a)
        a[:,:count] = candidate[:,start:start+count]
        b[:,:count] = reference[:,start:start+count]
        values = feature_errors(torch.from_numpy(a),torch.from_numpy(b))
        for key,value in values.items():
            totals[key] = totals.get(key,0.)+float(value.mean())*count
    result = {key:value/frames for key,value in totals.items()}
    result["objective"] = result["envelope"]+result["spectrum"]+result["band_envelope"]+2*result["centroid"]
    return result


def compare_score(score, plan, *, models=None, exclusions=None, mix=None):
    mix = MixPlan() if mix is None else mix
    if not isinstance(mix,MixPlan):
        raise ValueError("Expected original MixPlan")
    reference, baseline = render_score(score,exclusions=exclusions)
    candidate, report = render_score(score,plan,models=models,exclusions=exclusions)
    if report["score_sha256"] != baseline["score_sha256"] or report["eligible"] != baseline["eligible"]:
        raise ValueError("Multi-model paired routing changed")
    track_report = {}
    for track in TRACKS:
        values = _pcm_stats(candidate[track],reference[track])
        if not report["substituted_by_track"][track] and not values["exact_pcm_match"]:
            raise ValueError("Multi-model renderer changed a procedural-only track")
        track_report[track] = values
    a, am = mix_stems(candidate,mix)
    b, bm = mix_stems(reference,mix)
    return {
        "routing":report,"tracks":track_report,"mix_waveform":_pcm_stats(a,b),
        "mix_features":mix_feature_comparison(a,b),
        "reference_wav_sha256":hashlib.sha256(encode_wav_channels(b,SAMPLE_RATE)).hexdigest(),
        "candidate_wav_sha256":hashlib.sha256(encode_wav_channels(a,SAMPLE_RATE)).hexdigest(),
        "candidate_master_attenuation":am["master_attenuation"],
        "reference_master_attenuation":bm["master_attenuation"],
        "perceptual_quality_established":False,"weights_published":False,
        "generated_audio_written":False,"external_recordings_used":False,
    }
