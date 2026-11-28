"""Strict JSON recipe interchange for user-authored original MIDI patterns.

A recipe contains only explicit numerical rhythm/pitch parameters; it cannot
reference an external sound library, URL, model, script or output path. An
optional SHA-256 pin protects the exact input bytes. Source timestamps and
identities are checked by the shared bounded file reader.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from .contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
    assert_keys, canonical_json, parse_json, read_regular_file,
)
from .patterns import PatternCell, PatternPlan, PatternVoice, render_patterns

PATTERN_PROTOCOL = "sonumio-original-pattern-recipe-v1"
PATTERN_MAX_BYTES = 256 * 1024
_PLAN_KEYS = {
    "ppqn", "bpm", "bars", "numerator", "denominator", "steps_per_quarter",
    "seed", "title", "force_first_pass",
}
_VOICE_KEYS = {
    "id", "name", "channel", "program", "cells", "repeats_every_bars",
    "rotate_steps_per_bar", "bank_msb", "bank_lsb", "muted", "solo",
}
_CELL_KEYS = {
    "step", "pitch", "duration_steps", "velocity", "probability_percent",
    "accent_percent", "microshift_ticks", "pan", "label",
}


@dataclass(frozen=True, slots=True)
class PatternRecipe:
    plan: PatternPlan
    voices: tuple[PatternVoice, ...]

    def __post_init__(self):
        if not isinstance(self.plan, PatternPlan):
            raise MidiWorkstationError("Recipe needs a typed PatternPlan")
        if type(self.voices) is not tuple or any(
            not isinstance(voice, PatternVoice) for voice in self.voices
        ):
            raise MidiWorkstationError("Recipe needs typed PatternVoice tuple")
        # Validate complete topology and theoretical event budget before
        # accepting or publishing a recipe, even if all notes are probabilistic.
        from .patterns import _check_plan
        _check_plan(self.plan, self.voices)

    def to_bytes(self):
        payload = {
            "protocol": PATTERN_PROTOCOL,
            "plan": asdict(self.plan),
            "voices": [asdict(voice) for voice in self.voices],
        }
        return canonical_json(payload, limit=PATTERN_MAX_BYTES)

    @property
    def sha256(self):
        return sha256(self.to_bytes()).hexdigest()

    def render(self):
        return render_patterns(self.plan, self.voices)


def decode_pattern_recipe(raw: bytes) -> PatternRecipe:
    payload = assert_keys(
        parse_json(raw, limit=PATTERN_MAX_BYTES),
        {"protocol", "plan", "voices"}, context="pattern recipe",
    )
    if payload["protocol"] != PATTERN_PROTOCOL:
        raise MidiFormatError("Unsupported original pattern recipe version")
    plan_values = assert_keys(payload["plan"], _PLAN_KEYS,
                              context="pattern plan")
    plan = PatternPlan(**plan_values)
    voice_rows = payload["voices"]
    if type(voice_rows) is not list or not 1 <= len(voice_rows) <= 48:
        raise MidiFormatError("Recipe must specify 1..48 typed voices")
    voices = []
    for row in voice_rows:
        values = dict(assert_keys(row, _VOICE_KEYS, context="pattern voice"))
        cell_rows = values["cells"]
        if type(cell_rows) is not list or len(cell_rows) > 1024:
            raise MidiFormatError("Pattern cells must be a bounded JSON array")
        values["cells"] = tuple(PatternCell(**assert_keys(
            item, _CELL_KEYS, context="pattern cell",
        )) for item in cell_rows)
        voices.append(PatternVoice(**values))
    return PatternRecipe(plan, tuple(voices))


def load_pattern_recipe(path: str | Path, *, expected_sha256=None):
    snapshot = read_regular_file(path, limit=PATTERN_MAX_BYTES,
                                 expected_sha256=expected_sha256)
    recipe = decode_pattern_recipe(snapshot.content)
    return recipe, {
        "protocol": "sonumio-original-pattern-snapshot-v1",
        "source_sha256": snapshot.sha256,
        "source_bytes": snapshot.size,
        "canonical_sha256": recipe.sha256,
        "verified_source_immutable": True,
        "external_assets_loaded": False,
        "network_uses": 0,
    }
