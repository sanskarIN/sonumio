"""Self-contained project render recipes integrating expression and effects.

The existing v1 project stays unchanged. A separate immutable recipe binds
a complete project, optional performance overlay and optional processing
rack. Rack automation must agree with the project's embedded automation.
No external paths or executable plugin configuration are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from .audio import SAMPLE_RATE, encode_wav_channels
from .errors import GenerationError
from .music_project import MusicProject, encode_project, decode_project, render_project
from .music_performance import (
    Performance, encode_performance, decode_performance, describe_performance,
    render_performance_stems, generate_performance_music,
)
from .music_rack import RackPlan, encode_rack, decode_rack, describe_rack, mix_rack_stems
from .music_mix import render_stems
from .music_score_io import _unique, _nonfinite, _object

FORMAT = "sonumio-project-render-recipe-v1"
MAX_RECIPE_BYTES = 3 * 1024 * 1024
MAX_RECEIPT_BYTES = 256 * 1024


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value, label):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise GenerationError("Invalid render " + label + " SHA256")
    return value


@dataclass(frozen=True, slots=True)
class RenderRecipe:
    project: MusicProject
    performance: Performance | None = None
    rack: RackPlan | None = None

    def __post_init__(self):
        if not isinstance(self.project, MusicProject):
            raise GenerationError("Render recipe requires a verified project")
        if self.performance is not None:
            if not isinstance(self.performance, Performance):
                raise GenerationError("Expected performance overlay")
            describe_performance(self.project.score, self.performance)
        if self.rack is not None:
            if not isinstance(self.rack, RackPlan):
                raise GenerationError("Expected processing rack")
            if self.rack.automation.sha256 != self.project.automation_sha256:
                raise GenerationError("Rack automation differs from project automation")
            self.rack.validate_score(self.project.score)

    @property
    def identity(self):
        return {
            "project_sha256": self.project.sha256,
            "score_sha256": self.project.score_sha256,
            "automation_sha256": self.project.automation_sha256,
            "performance_sha256": self.performance.sha256 if self.performance is not None else None,
            "rack_sha256": self.rack.sha256 if self.rack is not None else None,
        }

    @property
    def sha256(self):
        return _sha(FORMAT.encode("ascii") + b"\0" + _canonical(self.identity))

    @property
    def engine(self):
        if self.rack is not None:
            return "performance-rack" if self.performance is not None else "rack"
        return "performance" if self.performance is not None else "project"


def encode_recipe(recipe):
    if not isinstance(recipe, RenderRecipe):
        raise GenerationError("Expected RenderRecipe")
    data = _canonical({
        "format": FORMAT, "recipe_sha256": recipe.sha256, "identity": recipe.identity,
        "project": json.loads(encode_project(recipe.project)),
        "performance": (json.loads(encode_performance(recipe.performance))
                        if recipe.performance is not None else None),
        "rack": json.loads(encode_rack(recipe.rack)) if recipe.rack is not None else None,
    })
    if len(data) > MAX_RECIPE_BYTES:
        raise GenerationError("Render recipe exceeds byte budget")
    return data


def decode_recipe(data, *, expected_sha256=None):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_RECIPE_BYTES:
        raise GenerationError("Invalid render recipe byte count")
    if expected_sha256 is not None:
        _digest(expected_sha256, "expected recipe")
    try:
        raw = _object(json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                 parse_constant=_nonfinite),
                      ("format", "recipe_sha256", "identity", "project", "performance", "rack"))
        if raw["format"] != FORMAT:
            raise GenerationError("Unknown project render recipe format")
        recipe = RenderRecipe(
            decode_project(_canonical(raw["project"])),
            decode_performance(_canonical(raw["performance"])) if raw["performance"] is not None else None,
            decode_rack(_canonical(raw["rack"])) if raw["rack"] is not None else None,
        )
        if (raw["recipe_sha256"] != recipe.sha256
                or _canonical(raw["identity"]) != _canonical(recipe.identity)
                or (expected_sha256 is not None and expected_sha256 != recipe.sha256)):
            raise GenerationError("Render recipe identity mismatch")
        # Reject normalized nested content which did not use canonical fields.
        if _canonical(raw) != encode_recipe(recipe):
            raise GenerationError("Embedded render recipe documents are not canonical")
        return recipe
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed project render recipe") from exc


def read_recipe(path, *, expected_sha256=None):
    with open(path, "rb") as stream:
        return decode_recipe(stream.read(MAX_RECIPE_BYTES + 1), expected_sha256=expected_sha256)


def describe_recipe(recipe):
    if not isinstance(recipe, RenderRecipe):
        raise GenerationError("Expected RenderRecipe")
    project = recipe.project
    return {
        "format": FORMAT, "recipe_sha256": recipe.sha256, **recipe.identity,
        "engine": recipe.engine, "frames": project.score.plan.frame(project.score.plan.ticks),
        "sample_rate": SAMPLE_RATE, "channels": 2, "revision": project.revision,
        "performance": (describe_performance(project.score, recipe.performance)
                        if recipe.performance is not None else None),
        "rack": describe_rack(project.score, recipe.rack) if recipe.rack is not None else None,
        "rendered": False, "learned_model": False, "external_recordings_used": False,
        "authenticated": False,
    }


def render_recipe(recipe):
    report = describe_recipe(recipe)
    project = recipe.project
    if recipe.rack is not None:
        stems = (render_performance_stems(project.score, recipe.performance)
                 if recipe.performance is not None else render_stems(project.score))
        audio, details = mix_rack_stems(project.score, stems, recipe.rack)
        wav = encode_wav_channels(audio, SAMPLE_RATE)
    elif recipe.performance is not None:
        wav, details = generate_performance_music(project.score, recipe.performance, project.automation)
    else:
        wav, details = render_project(project)
    if (type(wav) is not bytes or len(wav) != 44 + report["frames"] * 4
            or details.get("score_sha256") != project.score_sha256):
        raise GenerationError("Integrated renderer output geometry or identity mismatch")
    digest = _sha(wav)
    if "wav_sha256" in details and details["wav_sha256"] != digest:
        raise GenerationError("Integrated renderer WAV digest mismatch")
    # Each delegated renderer reports its own processing details.
    report.update(rendered=True, wav_sha256=digest, wav_bytes=len(wav), details=details)
    report["receipt_sha256"] = _sha(_canonical(report))
    encode_receipt(report)
    return wav, report


def encode_receipt(report):
    if type(report) is not dict or report.get("rendered") is not True:
        raise GenerationError("Expected completed render receipt")
    try:
        body = {k: v for k, v in report.items() if k != "receipt_sha256"}
        if report.get("receipt_sha256") != _sha(_canonical(body)):
            raise GenerationError("Render receipt digest mismatch")
        data = _canonical(report)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed render receipt") from exc
    if len(data) > MAX_RECEIPT_BYTES:
        raise GenerationError("Render receipt exceeds byte budget")
    return data


def decode_receipt(data):
    if type(data) is not bytes or not 1 <= len(data) <= MAX_RECEIPT_BYTES:
        raise GenerationError("Invalid render receipt byte count")
    try:
        report = json.loads(data.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
        encode_receipt(report)
        return report
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError) as exc:
        if isinstance(exc, GenerationError):
            raise
        raise GenerationError("Malformed render receipt JSON") from exc


def verify_render(recipe, wav, report, *, replay=False):
    """Verify content and geometry, optionally repeat the complete synthesis.

    Hash checks alone do not prove the audio was produced by this engine.
    Replay compares both the complete WAV and deterministic receipt.
    """
    if type(replay) is not bool:
        raise GenerationError("Invalid replay option")
    encode_receipt(report)
    expected = describe_recipe(recipe)
    for key, value in expected.items():
        if key != "rendered" and report.get(key) != value:
            raise GenerationError("Render receipt differs from recipe: " + key)
    if (type(wav) is not bytes or len(wav) != 44 + expected["frames"] * 4
            or report.get("wav_bytes") != len(wav) or report.get("wav_sha256") != _sha(wav)):
        raise GenerationError("Render WAV checksum or size mismatch")
    from .music_render_wav import inspect_pcm16
    pcm = inspect_pcm16(wav, expected["frames"])
    if replay:
        actual, receipt = render_recipe(recipe)
        if actual != wav or encode_receipt(receipt) != encode_receipt(report):
            raise GenerationError("Render replay differs from provided artifacts")
    return {"recipe_sha256": recipe.sha256, "wav_sha256": _sha(wav),
            "receipt_sha256": report["receipt_sha256"], "pcm": pcm,
            "replayed": replay, "authenticated": False}
