"""Leakage-audited adjacent-window synthetic audio for continuation research.

Unlike shuffled 1.024-second clips, each pair is cropped from ONE longer
first-party DSP recording. Previous[-prefix:] equals future[:prefix] exactly,
sample-for-sample, before any model or waveform processing.

The two English/Hindi captions for a recording share the SAME waveform
and must never count as independent acoustic observations. Seeds identify
recording groups; all descendants of a recording stay in the same split.

This benchmark is narrow and synthetic: related source combinations and
deterministic DSP tones limit conclusions about out-of-family audio.
No downloaded audio, speech recordings, model weights or WAV files.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
import torch

from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS
from research.velocity_prefix_continuation import _prefix_frames
from research.text_audio import (
    ByteTextTokenizer, SYNTHETIC_COMBINATIONS, _synth_prompts,
)
from research.wave_diffusion import (
    CATEGORIES, SAMPLE_RATE, _downsample_24k_to_8k,
)
from sonumio.engines.procedural import generate as dsp_generate

RECIPE = "sonumio-adjacent-dsp-window-pairs-v1"
SOURCE_DURATION = 3.0
SOURCE_FRAMES = int(SAMPLE_RATE * SOURCE_DURATION)
MIN_CROP_OFFSET = 400
MAX_CROP_OFFSET = 1600
MAX_SEEDS = 8
MAX_GROUPS = len(SYNTHETIC_COMBINATIONS) * MAX_SEEDS


def _validate_members(members: tuple[int, ...]) -> None:
    if type(members) is not tuple or members not in SYNTHETIC_COMBINATIONS:
        raise ValueError("Unknown or unsupported first-party DSP source family")


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Recording seeds must be unsigned 32-bit integers")


def _sample_sha(data: np.ndarray) -> str:
    if (not isinstance(data, np.ndarray) or data.dtype != np.float32
            or data.ndim != 1 or not np.isfinite(data).all()):
        raise ValueError("Cannot fingerprint invalid waveform bytes")
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def _offset(members: tuple[int, ...], seed: int) -> int:
    """Deterministic crop position even for seed-invariant ambient oscillators."""
    fingerprint = hashlib.sha256(
        b"sonumio-adjacent-offset-v1\0"
        + bytes(members) + seed.to_bytes(4, "little")
    ).digest()
    return MIN_CROP_OFFSET + (
        int.from_bytes(fingerprint[:8], "little")
        % (MAX_CROP_OFFSET - MIN_CROP_OFFSET + 1)
    )


@dataclass(frozen=True)
class AdjacentRecording:
    """One acoustic observation, with no duplicated language samples."""
    members: tuple[int, ...]
    seed: int
    offset: int
    prefix_frames: int
    previous: np.ndarray
    future: np.ndarray
    source_sha256: str
    previous_sha256: str
    future_sha256: str
    join_sha256: str

    @property
    def new_frames(self) -> int:
        return FRAMES - self.prefix_frames

    @property
    def reconstructed_pair_frames(self) -> int:
        return FRAMES + self.new_frames


def render_adjacent_recording(
    members: tuple[int, ...], seed: int, *,
    prefix_frames: int = 1024,
) -> AdjacentRecording:
    """Create two overlapping real windows from a single 3-second DSP source.

    No resampling is performed on windows individually. That is important:
    independently FIR-downsampled windows would have different boundary
    artifacts and would not prove exact real recording adjacency.
    """
    _validate_members(members)
    _validate_seed(seed)
    _prefix_frames(prefix_frames)
    prompt = " and ".join(CATEGORIES[i] for i in members)
    source = dsp_generate(prompt, SOURCE_DURATION, seed).samples
    if (not isinstance(source, np.ndarray)
            or source.ndim != 1
            or source.shape != (int(24_000 * SOURCE_DURATION),)
            or source.dtype != np.float32
            or not np.isfinite(source).all()):
        raise ValueError("Unexpected first-party DSP source")
    downsampled = _downsample_24k_to_8k(source)
    # Some first-party oscillator sound families ignore the DSP seed.
    # Apply a gentle, seed-derived time envelope to the *whole source*
    # BEFORE extracting both windows so each recording group has unique
    # underlying samples. This is deterministic synthetic augmentation,
    # not evidence of independent real-world acoustic content.
    key = hashlib.sha256(
        b"sonumio-adjacent-envelope-v1\0"
        + bytes(members) + seed.to_bytes(4, "little")
    ).digest()
    freq = 0.15 + (int.from_bytes(key[:2], "little") / 65535.0) * 0.45
    phase = 2 * math.pi * int.from_bytes(key[2:4], "little") / 65535.0
    time = np.arange(len(downsampled), dtype=np.float64) / SAMPLE_RATE
    envelope = (0.91 + 0.07 * np.sin(2 * math.pi * freq * time + phase))
    downsampled = np.ascontiguousarray(
        (downsampled.astype(np.float64) * envelope).astype(np.float32)
    )
    if (downsampled.shape != (SOURCE_FRAMES,)
            or downsampled.dtype != np.float32
            or not np.isfinite(downsampled).all()
            or np.max(np.abs(downsampled)) > 1.0001):
        raise ValueError("Unexpected downsampled synthetic DSP source")
    offset = _offset(members, seed)
    shift = FRAMES - prefix_frames
    first = np.ascontiguousarray(downsampled[offset:offset + FRAMES]).copy()
    second = np.ascontiguousarray(
        downsampled[offset + shift:offset + shift + FRAMES]
    ).copy()
    if (first.shape != (FRAMES,) or second.shape != (FRAMES,)
            or offset + shift + FRAMES > len(downsampled)
            or not np.array_equal(first[-prefix_frames:], second[:prefix_frames])):
        raise ValueError("Adjacent windows are not exactly contiguous")
    complete = np.concatenate((first, second[prefix_frames:]))
    expected = downsampled[offset:offset + FRAMES + shift]
    if (complete.shape != (FRAMES + shift,)
            or not np.array_equal(complete, expected)):
        raise ValueError("Adjacent clip provenance cannot reconstruct source")
    return AdjacentRecording(
        members=members, seed=seed, offset=offset,
        prefix_frames=prefix_frames,
        previous=first, future=second,
        source_sha256=_sample_sha(downsampled),
        previous_sha256=_sample_sha(first),
        future_sha256=_sample_sha(second),
        join_sha256=_sample_sha(complete),
    )


@dataclass(frozen=True)
class AdjacentDataset:
    """Caption examples plus recording-group provenance, not a train split."""
    previous: torch.Tensor
    future: torch.Tensor
    tokens: torch.Tensor
    captions: tuple[str, ...]
    group_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    languages: tuple[str, ...]
    recording_seeds: tuple[int, ...]
    prefix_frames: int
    group_count: int
    fingerprint: str
    exact_adjacency_verified: bool

    @property
    def examples(self) -> int:
        return len(self.captions)

    def check(self) -> None:
        count = self.examples
        if (type(count) is not int or not 1 <= count <= 2 * MAX_GROUPS
                or self.previous.shape != (count, 1, FRAMES)
                or self.future.shape != (count, 1, FRAMES)
                or self.previous.dtype != torch.float32
                or self.future.dtype != torch.float32
                or self.previous.device.type != "cpu"
                or self.future.device.type != "cpu"
                or not torch.isfinite(self.previous).all()
                or not torch.isfinite(self.future).all()
                or self.tokens.shape[0] != count or self.tokens.dtype != torch.long
                or len(self.languages) != count
                or len(self.group_ids) != count
                or len(self.source_hashes) != count
                or len(self.recording_seeds) != count):
            raise ValueError("Invalid adjacent dataset batch or provenance")
        if (not torch.equal(
                self.previous[:, :, -self.prefix_frames:],
                self.future[:, :, :self.prefix_frames]
            ) or len(set(self.group_ids)) != self.group_count
                or not self.exact_adjacency_verified):
            raise ValueError("Adjacent dataset lacks exact boundary identity")
        for group in set(self.group_ids):
            ids = [i for i, value in enumerate(self.group_ids) if value == group]
            if (len(ids) != 2
                    or {self.languages[i] for i in ids} != {"en", "hi"}
                    or len({self.source_hashes[i] for i in ids}) != 1
                    or len({self.recording_seeds[i] for i in ids}) != 1
                    or not torch.equal(
                        self.previous[ids[0]], self.previous[ids[1]]
                    ) or not torch.equal(
                        self.future[ids[0]], self.future[ids[1]]
                    )):
                raise ValueError("Bilingual group must duplicate one recording")


def _validate_seed_tuple(seeds: tuple[int, ...]) -> None:
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= MAX_SEEDS
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Use 1..8 distinct unsigned recording seeds")
    for seed in seeds:
        _validate_seed(seed)


def adjacent_bilingual_dataset(
    seeds: tuple[int, ...] = TRAIN_SEEDS, *,
    prefix_frames: int = 1024,
    combinations: tuple[tuple[int, ...], ...] = SYNTHETIC_COMBINATIONS,
) -> AdjacentDataset:
    """Render 21 DSP source families x N seeds x 2 caption languages.

    All audio used by both language captions is identical, so model-seed
    replication or source-group bootstrap must count each waveform once.
    """
    _validate_seed_tuple(seeds)
    _prefix_frames(prefix_frames)
    if (type(combinations) is not tuple
            or not 1 <= len(combinations) <= len(SYNTHETIC_COMBINATIONS)
            or len(set(combinations)) != len(combinations)):
        raise ValueError("Adjacent source combinations must be unique")
    for members in combinations:
        _validate_members(members)

    prior, future, captions, languages = [], [], [], []
    groups, source_hashes, recording_seeds = [], [], []
    fingerprint = hashlib.sha256(RECIPE.encode("ascii"))
    group_fingerprints: set[str] = set()
    seen_source_hashes: set[str] = set()
    for members in combinations:
        for seed in seeds:
            recording = render_adjacent_recording(
                members, seed, prefix_frames=prefix_frames,
            )
            group = hashlib.sha256(
                b"sonumio-adjacent-group-v1\0"
                + bytes(members) + seed.to_bytes(4, "little")
            ).hexdigest()
            if (group in group_fingerprints
                    or recording.source_sha256 in seen_source_hashes):
                raise ValueError("Duplicate independent source recording")
            group_fingerprints.add(group)
            seen_source_hashes.add(recording.source_sha256)
            fingerprint.update(
                bytes(members) + seed.to_bytes(4, "little")
                + prefix_frames.to_bytes(2, "little")
                + recording.source_sha256.encode("ascii")
                + recording.previous_sha256.encode("ascii")
                + recording.future_sha256.encode("ascii")
            )
            for lang in ("en", "hi"):
                caption = _synth_prompts(members, lang, seed)
                captions.append(caption)
                languages.append(lang)
                prior.append(torch.from_numpy(recording.previous.copy()).view(1, FRAMES))
                future.append(torch.from_numpy(recording.future.copy()).view(1, FRAMES))
                groups.append(group)
                source_hashes.append(recording.source_sha256)
                recording_seeds.append(seed)
                fingerprint.update(
                    lang.encode("ascii") + caption.encode("utf-8")
                )
    result = AdjacentDataset(
        previous=torch.stack(prior),
        future=torch.stack(future),
        tokens=ByteTextTokenizer().encode(captions),
        captions=tuple(captions),
        group_ids=tuple(groups),
        source_hashes=tuple(source_hashes),
        languages=tuple(languages),
        recording_seeds=tuple(recording_seeds),
        prefix_frames=prefix_frames,
        group_count=len(group_fingerprints),
        fingerprint=fingerprint.hexdigest(),
        exact_adjacency_verified=True,
    )
    result.check()
    return result


def audit_adjacent_split(train: AdjacentDataset, heldout: AdjacentDataset) -> dict:
    """Fail closed on exact source/descendant leakage across named splits."""
    if not isinstance(train, AdjacentDataset) or not isinstance(
        heldout, AdjacentDataset
    ):
        raise ValueError("Adjacent split requires validated paired datasets")
    train.check()
    heldout.check()
    if train.prefix_frames != heldout.prefix_frames:
        raise ValueError("Mixed adjacency overlap lengths in evaluation")
    exact_source_overlap = set(train.source_hashes) & set(heldout.source_hashes)
    shared_recording_seeds = set(train.recording_seeds) & set(
        heldout.recording_seeds
    )
    shared_group_ids = set(train.group_ids) & set(heldout.group_ids)
    # Distinct seeds cannot establish that two related procedural source
    # families are independent acoustic tasks. Flag this limitation.
    if (exact_source_overlap or shared_recording_seeds or shared_group_ids
            or train.fingerprint == heldout.fingerprint):
        raise ValueError("Adjacent train/heldout source or seed leakage detected")
    return {
        "protocol": RECIPE,
        "train_caption_examples": train.examples,
        "heldout_caption_examples": heldout.examples,
        "train_independent_recordings": train.group_count,
        "heldout_independent_recordings": heldout.group_count,
        "train_fingerprint": train.fingerprint,
        "heldout_fingerprint": heldout.fingerprint,
        "training_seeds": sorted(set(train.recording_seeds)),
        "heldout_seeds": sorted(set(heldout.recording_seeds)),
        "prefix_frames": train.prefix_frames,
        "exact_sample_overlap_between_neighbors": True,
        "source_recording_groups_not_split_across_languages": True,
        "exact_audio_source_leakage_detected": False,
        "shared_seed_leakage_detected": False,
        "independent_acoustic_families_verified": False,
        "rights": "sonumio_first_party_procedural_dsp",
        "seeded_envelope_augmentation": True,
        "warning": (
            "A seed-disjoint procedural benchmark is not a real-world "
            "distribution split. Related DSP sound families and deterministic "
            "tones can share acoustic structure despite different bytes; "
            "seeded gain-envelope augmentation does not make source families independent."
        ),
    }


def paired_continuity_probes(data: AdjacentDataset) -> dict:
    """Numerical ground-truth seam controls; absolutely no listening claims."""
    if not isinstance(data, AdjacentDataset):
        raise ValueError("Requires an adjacent dataset")
    data.check()
    prefix = data.prefix_frames
    groups = {}
    for i, group in enumerate(data.group_ids):
        if group in groups:
            continue
        previous = data.previous[i, 0].double()
        future = data.future[i, 0].double()
        exact_known = torch.equal(
            previous[-prefix:], future[:prefix]
        )
        joined = torch.cat((previous, future[prefix:]))
        hard_boundary = float(torch.abs(previous[-1] - future[0]))
        new_boundary = float(torch.abs(future[prefix] - future[prefix - 1]))
        if not exact_known:
            raise ValueError("A supposedly adjacent source has a corrupt prefix")
        groups[group] = {
            "recording_sha256": data.source_hashes[i],
            "known_prefix_exact": True,
            "hard_clip_concat_jump": hard_boundary,
            "true_first_unknown_sample_jump": new_boundary,
            "stitched_frames": joined.numel(),
            "stitched_sha256": hashlib.sha256(
                joined.float().numpy().tobytes()
            ).hexdigest(),
        }
    means = {
        "hard_clip_concat_jump": float(np.mean(
            [v["hard_clip_concat_jump"] for v in groups.values()]
        )),
        "true_first_unknown_sample_jump": float(np.mean(
            [v["true_first_unknown_sample_jump"] for v in groups.values()]
        )),
    }
    if not all(math.isfinite(x) and x >= 0 for x in means.values()):
        raise ValueError("Nonfinite adjacency continuity metric")
    return {
        "recordings_counted_once": len(groups),
        "caption_examples": data.examples,
        "prefix_frames": prefix,
        "mean_jump": means,
        "ground_truth_contiguous_windows_not_generated": True,
        "perceptual_audio_quality_measured": False,
        "semantic_continuity_or_prompt_adherence_measured": False,
    }
