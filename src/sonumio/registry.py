"""An explicit model registry. An engine is not automatically an original AI model."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class ModelEntry:
    id: str
    origin: str
    model_kind: str
    tasks: tuple[str, ...]
    license: str
    enabled_by_default: bool
    description: str
    upstream_url: str | None = None
    original_sonumio_model: bool = False

ENTRIES: dict[str, ModelEntry] = {
    "procedural-sfx-v0": ModelEntry(
        id="procedural-sfx-v0", origin="sonumio", model_kind="procedural DSP engine, NOT AI",
        tasks=("sound-effects", "ambient-tones"), license="Apache-2.0",
        enabled_by_default=True,
        description="Deterministic rain, thunder, wind, water, birds and ambient tones.",
    ),
    "meta-musicgen-small": ModelEntry(
        id="meta-musicgen-small", origin="Meta", model_kind="pretrained third-party Transformer",
        tasks=("text-to-music",), license="CC-BY-NC-4.0 (model weights)",
        enabled_by_default=False,
        description="Optional research-only local inference; weights are not bundled.",
        upstream_url="https://huggingface.co/facebook/musicgen-small",
    ),
}

def list_models() -> list[dict]:
    return [asdict(entry) for entry in ENTRIES.values()]
