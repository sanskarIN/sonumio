"""Self-contained sound-design extension to the original render recipe.

An extended recipe preserves the v1 recipe's project, expressive timing
and effects. It additionally binds pitched and percussion patch banks.
All costs are computed from the actual performed sample intervals before
allocating stems. The original v1 document and renderer stay unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np

from .arrangement import TRACKS, arrangement_digest
from .audio import SAMPLE_RATE, encode_wav_channels
from .errors import GenerationError
from .music_automation import mix_automated_stems
from .music_drum_bank import DrumBank, encode_drum_bank, decode_drum_bank
from .music_drum_patches import drum_cost, render_drum_patch
from .music_instruments import render_event
from .music_mix import _render_stems, MAX_STEM_BYTES
from .music_patch_bank import PatchBank, encode_patch_bank, decode_patch_bank
from .music_patch_synthesis import patch_cost, render_patch, active_partials
from .music_performance import compile_performance, PerformedNote
from .music_rack import mix_rack_stems
from .music_render_recipe import (
    RenderRecipe, encode_recipe, decode_recipe, describe_recipe,
    encode_receipt, decode_receipt, _canonical, _sha, _digest,
)
from .music_render_wav import inspect_pcm16
from .music_score_io import _object, _unique, _nonfinite

FORMAT = "sonumio-complete-sound-recipe-v1"
MAX_BYTES = 4 * 1024 * 1024
MAX_PITCHED_COST = 120_000_000
MAX_DRUM_COST = 100_000_000
MAX_TOTAL_COST = 150_000_000
MAX_RENDER_NOTE_FRAMES = 12 * 60 * SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class SoundRecipe:
    recipe: RenderRecipe
    pitched: PatchBank | None = None
    drums: DrumBank | None = None

    def __post_init__(self):
        if not isinstance(self.recipe, RenderRecipe):
            raise GenerationError("Sound recipe requires a validated base recipe")
        for bank, kind in ((self.pitched, PatchBank), (self.drums, DrumBank)):
            if bank is not None:
                if not isinstance(bank, kind):
                    raise GenerationError("Invalid sound recipe patch bank")
                if bank.score_sha256 != self.recipe.project.score_sha256:
                    raise GenerationError("Sound recipe bank score SHA256 mismatch")
        if not any(bank is not None and bank.routes for bank in (self.pitched, self.drums)):
            raise GenerationError("Sound recipe requires at least one assigned timbre")
        compile_sound(self)

    @property
    def identity(self):
        return {
            "base_recipe_sha256": self.recipe.sha256,
            **self.recipe.identity,
            "pitched_bank_sha256": self.pitched.sha256 if self.pitched is not None else None,
            "drum_bank_sha256": self.drums.sha256 if self.drums is not None else None,
        }

    @property
    def sha256(self):
        return _sha(FORMAT.encode("ascii") + b"\0" + _canonical(self.identity))


@dataclass(frozen=True, slots=True)
class SoundPlan:
    notes: tuple[PerformedNote, ...]
    pitched_cost: int
    drum_cost: int
    track_cost: tuple[int, ...]
    suppressed_partials: tuple[int, ...]
    custom_events: tuple[int, ...]
    track_frames: tuple[int, ...]

    def report(self):
        return {
            "pitched_cost": self.pitched_cost, "drum_cost": self.drum_cost,
            "combined_cost": self.pitched_cost + self.drum_cost,
            "maximum_cost": MAX_TOTAL_COST,
            "note_frames": sum(self.track_frames),
            "tracks": {
                track: {
                    "cost": self.track_cost[i], "custom_events": self.custom_events[i],
                    "suppressed_partials": self.suppressed_partials[i],
                    "note_frames": self.track_frames[i],
                }
                for i, track in enumerate(TRACKS)
            },
        }


def _maps(recipe):
    pitched = {row.track: row.patch for row in recipe.pitched.routes} if recipe.pitched is not None else {}
    drums = {row.track: row.patch for row in recipe.drums.routes} if recipe.drums is not None else {}
    return pitched, drums


def compile_sound(recipe):
    if not isinstance(recipe, SoundRecipe):
        raise GenerationError("Expected SoundRecipe")
    score = recipe.recipe.project.score
    if recipe.recipe.performance is not None:
        notes = compile_performance(score, recipe.recipe.performance)
    else:
        notes = tuple(PerformedNote(i, score.plan.frame(event.tick),
            score.plan.frame(event.tick + event.length), event.velocity, event.pan)
            for i, event in enumerate(score.events))
    total = sum(note.stop - note.start for note in notes)
    frames = score.plan.frame(score.plan.ticks)
    if total > MAX_RENDER_NOTE_FRAMES or frames * len(TRACKS) * 8 > MAX_STEM_BYTES:
        raise GenerationError("Sound recipe exceeds sample or stem budget")
    pitched, drums = _maps(recipe)
    cost = [0] * len(TRACKS)
    suppressed = [0] * len(TRACKS)
    custom = [0] * len(TRACKS)
    durations = [0] * len(TRACKS)
    for note in notes:
        event = score.events[note.index]
        i = TRACKS.index(event.track)
        length = note.stop - note.start
        durations[i] += length
        if event.track in pitched:
            patch = pitched[event.track]
            cost[i] += patch_cost(event, length, patch)
            custom[i] += 1
            for bank in patch.banks:
                if bank.gain:
                    suppressed[i] += (sum(partial.amplitude > 0 for partial in bank.partials)
                                      - len(active_partials(event.pitch, bank)))
        elif event.track in drums:
            cost[i] += drum_cost(event, length, drums[event.track])
            custom[i] += 1
    pc = sum(cost[TRACKS.index(t)] for t in pitched)
    dc = sum(cost[TRACKS.index(t)] for t in drums)
    if pc > MAX_PITCHED_COST or dc > MAX_DRUM_COST or pc + dc > MAX_TOTAL_COST:
        raise GenerationError("Sound recipe exceeds actual performed synthesis budget")
    return SoundPlan(notes, pc, dc, tuple(cost), tuple(suppressed), tuple(custom), tuple(durations))


def encode_sound_recipe(recipe):
    if not isinstance(recipe, SoundRecipe):
        raise GenerationError("Expected SoundRecipe")
    data = _canonical({
        "format": FORMAT, "sound_recipe_sha256": recipe.sha256, "identity": recipe.identity,
        "base_recipe": json.loads(encode_recipe(recipe.recipe)),
        "pitched": json.loads(encode_patch_bank(recipe.pitched)) if recipe.pitched is not None else None,
        "drums": json.loads(encode_drum_bank(recipe.drums)) if recipe.drums is not None else None,
    })
    if len(data) > MAX_BYTES:
        raise GenerationError("Sound recipe exceeds byte budget")
    return data


def decode_sound_recipe(data, *, expected_sha256=None):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_BYTES:
        raise GenerationError("Invalid sound recipe byte count")
    if expected_sha256 is not None:
        _digest(expected_sha256, "expected sound recipe")
    try:
        raw = _object(json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                 parse_constant=_nonfinite),
                      ("format","sound_recipe_sha256","identity","base_recipe","pitched","drums"))
        if raw["format"] != FORMAT:
            raise GenerationError("Unknown complete sound recipe format")
        recipe = SoundRecipe(
            decode_recipe(_canonical(raw["base_recipe"])),
            decode_patch_bank(_canonical(raw["pitched"])) if raw["pitched"] is not None else None,
            decode_drum_bank(_canonical(raw["drums"])) if raw["drums"] is not None else None,
        )
        if (raw["sound_recipe_sha256"] != recipe.sha256
                or _canonical(raw["identity"]) != _canonical(recipe.identity)
                or (expected_sha256 is not None and expected_sha256 != recipe.sha256)):
            raise GenerationError("Complete sound recipe identity mismatch")
        if _canonical(raw) != encode_sound_recipe(recipe):
            raise GenerationError("Embedded sound recipe documents are not canonical")
        return recipe
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed complete sound recipe") from exc


def read_sound_recipe(path, *, expected_sha256=None):
    with open(path,"rb") as stream:
        return decode_sound_recipe(stream.read(MAX_BYTES+1),expected_sha256=expected_sha256)


def describe_sound(recipe):
    plan = compile_sound(recipe)
    base = describe_recipe(recipe.recipe)
    return {
        "format": FORMAT, "sound_recipe_sha256": recipe.sha256, **recipe.identity,
        "frames": base["frames"], "sample_rate": SAMPLE_RATE, "channels": 2,
        "base_engine": base["engine"], "revision": base["revision"], "plan": plan.report(),
        "rendered": False, "learned_model": False, "external_recordings_used": False,
        "authenticated": False, "strictly_bandlimited_claimed": False,
    }


def render_sound_stems(recipe):
    plan = compile_sound(recipe)
    score = recipe.recipe.project.score
    pitched, drums = _maps(recipe)

    def voice(event, frames, seed):
        if event.track in pitched:
            return render_patch(event, frames, seed, pitched[event.track])
        if event.track in drums:
            return render_drum_patch(event, frames, seed, drums[event.track])
        return render_event(event, frames, seed)

    if recipe.recipe.performance is None:
        return _render_stems(score, voice)

    frames = score.plan.frame(score.plan.ticks)
    stems = {track: np.zeros((2, frames), dtype=np.float32) for track in TRACKS}
    for note in plan.notes:
        event = score.events[note.index]
        # Keep symbolic tick/pitch for independent percussion excitation.
        source = replace(event, velocity=note.velocity, pan=note.pan)
        mono = voice(source, note.stop-note.start, score.plan.seed)
        if (not isinstance(mono,np.ndarray) or mono.dtype not in (np.float32,np.float64)
                or mono.shape != (note.stop-note.start,) or not np.isfinite(mono).all()
                or np.max(np.abs(mono)) > 1.000001):
            raise GenerationError("Invalid performed custom note waveform")
        pan = (note.pan+1)*math.pi/4
        stems[event.track][0,note.start:note.stop] += (mono*math.cos(pan)).astype(np.float32)
        stems[event.track][1,note.start:note.stop] += (mono*math.sin(pan)).astype(np.float32)
    for stem in stems.values():
        if not np.isfinite(stem).all():
            raise GenerationError("Invalid performed custom stem")
        stem.setflags(write=False)
    return stems


def render_sound(recipe):
    report = describe_sound(recipe)
    project = recipe.recipe.project
    stems = render_sound_stems(recipe)
    if recipe.recipe.rack is not None:
        audio, details = mix_rack_stems(project.score, stems, recipe.recipe.rack)
    else:
        audio, details = mix_automated_stems(project.score, stems, project.automation)
    wav = encode_wav_channels(audio, SAMPLE_RATE)
    inspect_pcm16(wav, report["frames"])
    report.update(rendered=True, wav_sha256=_sha(wav), wav_bytes=len(wav), details=details)
    report["receipt_sha256"] = _sha(_canonical(report))
    encode_receipt(report)
    return wav, report


def verify_sound(recipe, wav, report, *, replay=False):
    if type(replay) is not bool:
        raise GenerationError("Invalid sound replay option")
    encode_receipt(report)
    expected = describe_sound(recipe)
    for key, value in expected.items():
        if key != "rendered" and report.get(key) != value:
            raise GenerationError("Sound receipt differs from recipe: " + key)
    if (type(wav) is not bytes or len(wav) != 44 + expected["frames"]*4
            or report.get("wav_bytes") != len(wav) or report.get("wav_sha256") != _sha(wav)):
        raise GenerationError("Sound WAV checksum or byte count mismatch")
    pcm = inspect_pcm16(wav, expected["frames"])
    if replay:
        actual, receipt = render_sound(recipe)
        if actual != wav or encode_receipt(receipt) != encode_receipt(report):
            raise GenerationError("Sound replay differs from supplied artifacts")
    return {"sound_recipe_sha256":recipe.sha256,"wav_sha256":_sha(wav),
            "receipt_sha256":report["receipt_sha256"],"pcm":pcm,
            "replayed":replay,"authenticated":False}
