"""Rights-gated *local* WAV ingestion for experimental Sonumio diffusion.

All inputs remain offline. A valid manifest declares permissions but does not prove
legal ownership: a human must review the original source and consent documents.
Only uncompressed 16-bit PCM mono/stereo WAV is supported. Speech/voice datasets
are intentionally excluded. This loader is not an automatic voice detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import wave

import numpy as np

from research.dataset_manifest import ManifestError, validate_manifest
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, SAMPLE_RATE
from sonumio.engines.procedural import TOKENS

MAX_ASSET_BYTES = 20 * 1024 * 1024
MAX_ASSETS = 10_000
CATEGORY_TO_DSP = {"ocean": "water", "ambient": "drone"}


@dataclass(frozen=True)
class LoadedSplit:
    """Decoded clips, labels and immutable provenance summary for one split."""

    clips: np.ndarray  # float32, shape [N, 1, CLIP_FRAMES]
    labels: np.ndarray  # int64, shape [N]
    asset_count: int
    fingerprint: str


def caption_category(caption: str) -> int:
    """Map an explicitly recognized *single* sound category to a class label.

    This is limited keyword matching, NOT an open-vocabulary text encoder.
    Ambiguous prompts are rejected rather than silently selecting one class.
    """
    if not isinstance(caption, str) or len(caption) > 800:
        raise ManifestError("caption must be text of at most 800 characters")
    matches = []
    for idx, category in enumerate(CATEGORIES):
        words = TOKENS[CATEGORY_TO_DSP.get(category, category)]
        if any((re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", caption, re.I)
                if word.isascii() else word in caption) for word in words):
            matches.append(idx)
    if len(matches) != 1:
        raise ManifestError("caption must describe exactly one of rain/thunder/wind/forest/ocean/ambient")
    return matches[0]


def _confined_path(root: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename or "\\" in filename or "\x00" in filename:
        raise ManifestError("local_path must be a non-empty POSIX relative file path")
    rel = PurePosixPath(filename)
    if (rel.is_absolute() or any(part in (".", "..") for part in filename.split("/"))
            or ":" in filename or filename.startswith("~")):
        raise ManifestError("local_path must not escape the configured audio root")
    destination = (root / filename).resolve(strict=False)
    if not destination.is_relative_to(root.resolve()):
        raise ManifestError("audio asset resolves outside the configured audio root")
    if not destination.is_file():
        raise ManifestError(f"local WAV file does not exist: {filename}")
    return destination


def _decode_pcm16(wav_bytes: bytes, record: dict) -> tuple[np.ndarray, int]:
    try:
        with wave.open(BytesIO(wav_bytes), "rb") as stream:
            channels, width, rate, frames = (stream.getnchannels(), stream.getsampwidth(),
                                             stream.getframerate(), stream.getnframes())
            if stream.getcomptype() != "NONE" or width != 2 or channels not in (1, 2):
                raise ManifestError("only uncompressed 16-bit PCM mono/stereo WAV is supported")
            if rate != record["sample_rate"]:
                raise ManifestError("WAV sample rate differs from the manifest")
            if frames <= 0 or frames / rate > 600:
                raise ManifestError("WAV length is outside supported training limits")
            if abs(frames / rate - record["duration_seconds"]) > 1.1 / rate:
                raise ManifestError("WAV duration differs from the manifest")
            data = stream.readframes(frames)
    except (wave.Error, EOFError, OSError, ValueError, ZeroDivisionError) as exc:
        raise ManifestError("malformed or unsupported WAV file") from exc
    if len(data) != frames * channels * 2:
        raise ManifestError("truncated PCM data")
    pcm = np.frombuffer(data, dtype="<i2").reshape(-1, channels).astype(np.float32) / 32768
    mono = pcm.mean(axis=1)
    if not np.any(np.abs(mono) > 1e-6):
        raise ManifestError("fully silent recording is not eligible for training")
    return mono, rate


def _crop_clips(signal: np.ndarray, rate: int, per_asset: int) -> list[np.ndarray]:
    # Local import: scipy is required only for rights-gated research ingestion.
    from scipy.signal import resample_poly
    import math as _math

    segment_size = math.ceil(CLIP_FRAMES * rate / SAMPLE_RATE) + 128
    segments: list[np.ndarray] = []
    for index in range(per_asset):
        portion = (index + 0.5) / per_asset
        start = int(max(0, len(signal) - segment_size) * portion)
        cropped = signal[start:start + segment_size]
        factor = _math.gcd(SAMPLE_RATE, rate)
        resampled = resample_poly(cropped, SAMPLE_RATE // factor, rate // factor)
        result = np.zeros(CLIP_FRAMES, dtype=np.float32)
        frames = min(CLIP_FRAMES, len(resampled))
        result[:frames] = resampled[:frames].astype(np.float32)
        segments.append(result)
    return segments


def load_reviewed_splits(manifest: str | Path, audio_root: str | Path, *,
                         clips_per_asset: int = 2) -> tuple[dict[str, LoadedSplit], dict]:
    """Verify rights declarations, bytes and WAV layout before creating splits.

    The test split is read but never used by the training function. Every asset
    must be manually reviewed; this routine does not independently verify truth.
    """
    if not isinstance(clips_per_asset, int) or isinstance(clips_per_asset, bool) or not 1 <= clips_per_asset <= 8:
        raise ValueError("clips_per_asset must be an integer from 1 to 8")
    manifest = Path(manifest)
    root = Path(audio_root).resolve()
    if not root.is_dir():
        raise ManifestError("audio root must be an existing local directory")
    declaration = validate_manifest(manifest)
    if declaration["records"] > MAX_ASSETS:
        raise ManifestError("audio manifest exceeds the bounded asset limit")
    clips: dict[str, list[np.ndarray]] = {split: [] for split in ("train", "validation", "test")}
    labels: dict[str, list[int]] = {split: [] for split in clips}
    fingerprints = {split: sha256() for split in clips}
    counts = {split: 0 for split in clips}

    with manifest.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                path = _confined_path(root, row.get("local_path"))
                if path.suffix.lower() != ".wav":
                    raise ManifestError("only .wav files are permitted")
                if path.stat().st_size > MAX_ASSET_BYTES:
                    raise ManifestError("WAV asset exceeds 20 MiB safety limit")
                wav_bytes = path.read_bytes()
                if sha256(wav_bytes).hexdigest() != row["sha256"]:
                    raise ManifestError("WAV SHA256 does not match the manifest")
                label = caption_category(row["caption"])
                mono, rate = _decode_pcm16(wav_bytes, row)
                pieces = _crop_clips(mono, rate, clips_per_asset)
                split = row["split"]
                clips[split].extend(pieces)
                labels[split].extend([label] * len(pieces))
                counts[split] += 1
                fingerprints[split].update((row["asset_id"] + ":" + row["sha256"] + ":" +
                                            str(label) + ":" + str(clips_per_asset)).encode("utf-8"))
            except ManifestError as exc:
                raise ManifestError(f"line {line_no}: {exc}") from exc
    if not counts["train"] or not counts["validation"]:
        raise ManifestError("both train and validation recording groups must exist")

    loaded = {
        split: LoadedSplit(np.stack(clips[split])[:, None].astype(np.float32)
                           if clips[split] else np.empty((0, 1, CLIP_FRAMES), dtype=np.float32),
                           np.asarray(labels[split], dtype=np.int64), counts[split],
                           fingerprints[split].hexdigest())
        for split in clips
    }
    report = {
        "declaration": declaration,
        "asset_counts": counts,
        "clip_counts": {split: len(value.labels) for split, value in loaded.items()},
        "clips_per_asset": clips_per_asset,
        "wav_bytes_verified": True,
        "human_rights_review_remains_required": True,
        "voice_detection_performed": False,
        "split_fingerprints": {split: value.fingerprint for split, value in loaded.items()},
    }
    return loaded, report
