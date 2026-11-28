"""Canonical, digest-bound per-track synthesis settings for offline previews.

The recipe is data, not a project file or executable plugin. It references a
specific immutable workstation score by SHA-256, never a file path, URL,
model checkpoint or sample library. Only source-original mathematical
waveforms are supported by the downstream renderer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256 as _sha256
from pathlib import Path

from .contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
    assert_keys, canonical_json, parse_json, read_regular_file, sha256,
)
from .multitrack_preview import MultiTrackPreviewSettings, TrackPatch
from .score import Score

RECIPE_PROTOCOL = "sonumio-multitrack-patch-recipe-v1"
MAX_RECIPE_BYTES = 32 * 1024
PATCH_KEYS = frozenset(TrackPatch.__dataclass_fields__)
RENDER_KEYS = frozenset(MultiTrackPreviewSettings.__dataclass_fields__)


@dataclass(frozen=True, slots=True)
class MultitrackRecipe:
    """An immutable rendering plan pinned to one canonical symbolic score."""

    score_sha256: str
    settings: MultiTrackPreviewSettings

    def __post_init__(self) -> None:
        sha256(self.score_sha256, "bound source score")
        if not isinstance(self.settings, MultiTrackPreviewSettings):
            raise MidiWorkstationError("Multitrack recipe requires typed render settings")

    def to_bytes(self) -> bytes:
        value = {
            "protocol": RECIPE_PROTOCOL,
            "score_sha256": self.score_sha256,
            "render": {
                **{key: getattr(self.settings, key) for key in RENDER_KEYS - {"patches"}},
                "patches": [asdict(patch) for patch in self.settings.patches],
            },
        }
        return canonical_json(value, limit=MAX_RECIPE_BYTES)

    @property
    def sha256(self) -> str:
        return _sha256(self.to_bytes()).hexdigest()

    def check_score(self, score: Score) -> None:
        if not isinstance(score, Score):
            raise MidiWorkstationError("Recipe expects immutable workstation Score")
        if score.sha256 != self.score_sha256:
            raise MidiIntegrityError("Multitrack patch recipe binds another score digest")


def decode_multitrack_recipe(raw: bytes) -> MultitrackRecipe:
    """Reject unsupported fields, duplicate keys, unsafe values and size abuse."""
    if type(raw) is not bytes:
        raise MidiFormatError("Multitrack recipe must be UTF-8 bytes")
    if len(raw) > MAX_RECIPE_BYTES:
        raise MidiLimitError("Multitrack patch recipe exceeds byte budget")
    document = assert_keys(
        parse_json(raw, limit=MAX_RECIPE_BYTES),
        {"protocol", "score_sha256", "render"},
        context="multitrack recipe",
    )
    if document["protocol"] != RECIPE_PROTOCOL:
        raise MidiFormatError("Unsupported multitrack recipe protocol")
    render = assert_keys(document["render"], set(RENDER_KEYS),
                         context="multitrack settings")
    patches = render["patches"]
    if type(patches) is not list or not 1 <= len(patches) <= 16:
        raise MidiFormatError("Multitrack recipe needs 1..16 patch objects")
    typed_patches = []
    for value in patches:
        validated = assert_keys(value, set(PATCH_KEYS), context="instrument patch")
        typed_patches.append(TrackPatch(**validated))
    settings = MultiTrackPreviewSettings(
        patches=tuple(typed_patches),
        **{key: render[key] for key in RENDER_KEYS - {"patches"}},
    )
    return MultitrackRecipe(document["score_sha256"], settings)


def load_multitrack_recipe(
    source: str | Path, *, expected_sha256: str | None = None,
) -> tuple[MultitrackRecipe, dict]:
    snapshot = read_regular_file(
        source, limit=MAX_RECIPE_BYTES, expected_sha256=expected_sha256,
    )
    recipe = decode_multitrack_recipe(snapshot.content)
    return recipe, {
        "protocol": "sonumio-multitrack-recipe-source-v1",
        "source_sha256": snapshot.sha256,
        "source_bytes": snapshot.size,
        "canonical_sha256": recipe.sha256,
        "score_sha256": recipe.score_sha256,
        "canonical": snapshot.content == recipe.to_bytes(),
        "remote_assets_loaded": False,
        "trained_model_inference": False,
    }
