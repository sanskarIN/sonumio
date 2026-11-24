"""Quality and coverage diagnostics for already rights-gated local audio.

Readiness policies are explicit engineering thresholds, not perceptual scores
or proof of provenance. The audit never emits captions, asset IDs, group IDs,
local paths, audio, or weights. It does not modify the input dataset.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from research.audio_dataset import LoadedSplit, load_reviewed_splits, _cropped_clip_digest
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, SAMPLE_RATE

PROTOCOL = "sonumio-reviewed-corpus-readiness-v1"
SPLITS = ("train", "validation", "test")
LANGUAGES = ("en", "hi")


@dataclass(frozen=True)
class ReadinessPolicy:
    min_train_groups_per_category: int = 2
    min_validation_groups_per_category: int = 1
    min_centered_rms_dbfs: float = -55.0
    min_active_fraction: float = 0.1
    max_near_clip_fraction: float = 0.01
    max_absolute_dc: float = 0.05

    def __post_init__(self):
        for value in (self.min_train_groups_per_category, self.min_validation_groups_per_category):
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError("Minimum group count must be an integer in [1, 10000]")
        limits = (
            (self.min_centered_rms_dbfs, -100, 0),
            (self.min_active_fraction, 0, 1),
            (self.max_near_clip_fraction, 0, 1),
            (self.max_absolute_dc, 0, 1),
        )
        if any(type(x) not in (int, float) or not math.isfinite(x) or not lo <= x <= hi
               for x, lo, hi in limits):
            raise ValueError("Invalid readiness policy threshold")

    def as_dict(self) -> dict:
        return {
            "min_train_groups_per_category": self.min_train_groups_per_category,
            "min_validation_groups_per_category": self.min_validation_groups_per_category,
            "min_centered_rms_dbfs": self.min_centered_rms_dbfs,
            "min_active_fraction": self.min_active_fraction,
            "max_near_clip_fraction": self.max_near_clip_fraction,
            "max_absolute_dc": self.max_absolute_dc,
        }


def _validate_split(name: str, split: LoadedSplit) -> None:
    if not isinstance(split, LoadedSplit) or split.split_name != name:
        raise ValueError("Expected provenance-bearing loaded split")
    count = len(split.labels)
    if (not isinstance(split.clips, np.ndarray) or split.clips.shape != (count, 1, CLIP_FRAMES)
            or split.clips.dtype != np.float32 or split.labels.shape != (count,)
            or split.labels.dtype != np.int64 or len(split.provenance) != count
            or len(split.captions) != count or count > 80000
            or np.any(split.labels < 0) or np.any(split.labels >= len(CATEGORIES))):
        raise ValueError("Loaded audio and provenance rows are not aligned")
    asset_metadata = {}
    clip_identities = set()
    for row, label, caption in zip(split.provenance, split.labels, split.captions, strict=True):
        if (not row.asset_id or not row.group_id or row.language not in LANGUAGES
                or type(row.source_line) is not int or row.source_line < 1
                or type(row.crop_index) is not int or not 1 <= row.crop_index <= 8):
            raise ValueError("Invalid per-clip provenance")
        identity = (row.asset_id, row.crop_index)
        if identity in clip_identities:
            raise ValueError("Duplicate crop provenance")
        clip_identities.add(identity)
        metadata = (row.group_id, row.language, row.source_line, int(label), caption)
        if row.asset_id in asset_metadata and asset_metadata[row.asset_id] != metadata:
            raise ValueError("Inconsistent metadata within one recording")
        asset_metadata[row.asset_id] = metadata
    if len(asset_metadata) != split.asset_count:
        raise ValueError("Asset count does not match provenance")


def clip_quality(clip: np.ndarray, policy: ReadinessPolicy) -> dict:
    if (not isinstance(clip, np.ndarray) or clip.shape != (CLIP_FRAMES,)
            or clip.dtype != np.float32 or not np.isfinite(clip).all()):
        raise ValueError("Expected finite float32 model-input clip")
    wave = clip.astype(np.float64)
    dc = float(wave.mean())
    centered = wave - dc
    rms = float(np.sqrt(np.mean(centered ** 2)))
    rms_dbfs = 20 * math.log10(rms) if rms > 0 else None
    # 4 ms frames on the actual model input; use centered samples so a
    # constant DC offset is not misidentified as useful signal activity.
    window = SAMPLE_RATE // 250
    complete = len(centered) // window
    frames = centered[:complete * window].reshape(complete, window)
    activity = float(np.mean(np.sqrt(np.mean(frames ** 2, axis=1)) >=
                             10 ** (policy.min_centered_rms_dbfs / 20)))
    near_clip = float(np.mean(np.abs(wave) >= 0.999))
    reasons = []
    if rms_dbfs is None or rms_dbfs < policy.min_centered_rms_dbfs:
        reasons.append("low_centered_rms")
    if activity < policy.min_active_fraction:
        reasons.append("low_activity")
    if near_clip > policy.max_near_clip_fraction:
        reasons.append("near_clipping")
    if abs(dc) > policy.max_absolute_dc:
        reasons.append("dc_offset")
    return {
        "centered_rms_dbfs": rms_dbfs, "active_fraction": activity,
        "near_clip_fraction": near_clip, "absolute_dc": abs(dc),
        "peak": float(np.abs(wave).max()), "reasons": reasons,
    }


def _eligible_fingerprint(split: LoadedSplit, indices: tuple[int, ...]) -> str:
    """Bind sampler indices to the actual audited audio and recording identities."""
    digest = hashlib.sha256(b"sonumio-eligible-crop-provenance-v1\0")
    for index in indices:
        p = split.provenance[index]
        record = json.dumps([index, int(split.labels[index]), p.asset_id, p.group_id,
                             p.language, p.source_line, p.crop_index, split.captions[index]],
                            ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest.update(len(record).to_bytes(4, "little"))
        digest.update(record)
        digest.update(bytes.fromhex(_cropped_clip_digest(split.clips[index, 0])))
    return digest.hexdigest()


@dataclass(frozen=True)
class ReadinessResult:
    report: dict
    eligible_indices: dict[str, tuple[int, ...]]
    eligible_fingerprints: dict[str, str]


def audit_loaded_splits(splits: dict[str, LoadedSplit], policy: ReadinessPolicy | None = None) -> ReadinessResult:
    policy = policy if policy is not None else ReadinessPolicy()
    if not isinstance(policy, ReadinessPolicy) or set(splits) != set(SPLITS):
        raise ValueError("Expected train, validation and test loaded splits and readiness policy")
    for name in SPLITS:
        _validate_split(name, splits[name])
    group_splits = {}
    asset_splits = {}
    hash_splits = {}
    results = {}
    eligible_by_split = {}
    for name in SPLITS:
        split = splits[name]
        categories = {}
        eligible = []
        reasons_total = Counter()
        seen_hashes = set()
        all_groups = defaultdict(set)
        eligible_groups = defaultdict(set)
        all_assets = defaultdict(set)
        eligible_assets = defaultdict(set)
        all_languages = defaultdict(Counter)
        eligible_languages = defaultdict(Counter)
        counts = Counter()
        accepted_counts = Counter()
        signal_metrics = defaultdict(list)
        for index, (clip, label, provenance) in enumerate(zip(
            split.clips, split.labels, split.provenance, strict=True,
        )):
            category = CATEGORIES[int(label)]
            for registry, key in ((group_splits, provenance.group_id), (asset_splits, provenance.asset_id)):
                if key in registry and registry[key] != name:
                    raise ValueError("Recording identity crosses split boundaries")
                registry[key] = name
            waveform = clip[0]
            quality = clip_quality(waveform, policy)
            digest = _cropped_clip_digest(waveform)
            if digest in hash_splits and hash_splits[digest] != name:
                raise ValueError("Exact audio crop crosses split boundaries")
            hash_splits[digest] = name
            if digest in seen_hashes:
                quality["reasons"].append("repeated_exact_crop")
            seen_hashes.add(digest)
            counts[category] += 1
            all_groups[category].add(provenance.group_id)
            all_assets[category].add(provenance.asset_id)
            all_languages[category][provenance.language] += 1
            signal_metrics[category].append(quality)
            reasons_total.update(quality["reasons"])
            if not quality["reasons"]:
                eligible.append(index)
                accepted_counts[category] += 1
                eligible_groups[category].add(provenance.group_id)
                eligible_assets[category].add(provenance.asset_id)
                eligible_languages[category][provenance.language] += 1
        for category in CATEGORIES:
            rows = signal_metrics[category]
            metrics = {}
            for key in ("centered_rms_dbfs", "active_fraction", "near_clip_fraction", "absolute_dc", "peak"):
                values = [row[key] for row in rows if row[key] is not None]
                metrics[key] = {
                    "defined_clips": len(values),
                    "minimum": min(values) if values else None,
                    "median": float(np.median(values)) if values else None,
                    "maximum": max(values) if values else None,
                }
            categories[category] = {
                "clips": counts[category], "eligible_clips": accepted_counts[category],
                "recording_groups": len(all_groups[category]),
                "eligible_recording_groups": len(eligible_groups[category]),
                "assets": len(all_assets[category]), "eligible_assets": len(eligible_assets[category]),
                "clips_by_language": {lang: all_languages[category][lang] for lang in LANGUAGES},
                "eligible_clips_by_language": {lang: eligible_languages[category][lang] for lang in LANGUAGES},
                "signal_metrics": metrics,
            }
        eligible_by_split[name] = tuple(eligible)
        results[name] = {
            "split_fingerprint": split.fingerprint,
            "clips": len(split.labels), "eligible_clips": len(eligible),
            "recording_groups": len(set(p.group_id for p in split.provenance)),
            "eligible_recording_groups": len(set(split.provenance[i].group_id for i in eligible)),
            "rejection_reason_counts": dict(sorted(reasons_total.items())),
            "by_category": categories,
        }
    failures = []
    for name, threshold in (
        ("train", policy.min_train_groups_per_category),
        ("validation", policy.min_validation_groups_per_category),
    ):
        for category in CATEGORIES:
            groups = results[name]["by_category"][category]["eligible_recording_groups"]
            if groups < threshold:
                failures.append({"split": name, "category": category, "groups": groups, "required": threshold})
    report = {
        "protocol": PROTOCOL, "policy": policy.as_dict(),
        "splits": results, "coverage_failures": failures,
        "readiness_gate_passed": not failures,
        "gate_scope": "synthetic engineering thresholds for reviewed six-category audio",
        "exact_cross_split_crop_overlap": False,
        "rights_and_voice_claims_independently_verified": False,
        "model_quality_established": False,
        "audio_or_weights_saved": False,
    }
    return ReadinessResult(report, eligible_by_split, {
        name: _eligible_fingerprint(splits[name], indices)
        for name, indices in eligible_by_split.items()
    })


class GroupBalancedSampler:
    """Deterministic training-only category/group/asset/crop hierarchy.

    Category order is shuffled once per complete six-category cycle; within a
    category, groups, then assets, then eligible crops are chosen uniformly.
    Sampling with replacement is intentional. Large recording groups cannot
    dominate solely because they contain more clips. No held-out indices enter.
    """

    def __init__(self, splits: dict[str, LoadedSplit], readiness: ReadinessResult, *, seed: int):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Sampler seed must be an unsigned 32-bit integer")
        if not readiness.report.get("readiness_gate_passed"):
            raise ValueError("Cannot sample a corpus that failed readiness")
        train = splits["train"]
        _validate_split("train", train)
        if readiness.report["splits"]["train"]["split_fingerprint"] != train.fingerprint:
            raise ValueError("Sampler corpus does not match audited training fingerprint")
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        indices = readiness.eligible_indices["train"]
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(train.labels) for i in indices):
            raise ValueError("Invalid eligible training indices")
        if _eligible_fingerprint(train, indices) != readiness.eligible_fingerprints["train"]:
            raise ValueError("Eligible audio or provenance changed since audit")
        for index in indices:
            p = train.provenance[index]
            tree[int(train.labels[index])][p.group_id][p.asset_id].append(index)
        if set(tree) != set(range(len(CATEGORIES))):
            raise ValueError("Balanced sampling requires every category")
        self.tree = tree
        self.rng = np.random.default_rng(seed)
        self.cycle = []
        self.trace = hashlib.sha256(b"sonumio-group-balanced-index-trace-v1\0")
        self.draws = 0

    def _choose(self, values):
        return values[int(self.rng.integers(len(values)))]

    def draw(self, count: int) -> np.ndarray:
        if type(count) is not int or not 1 <= count <= 100000:
            raise ValueError("Draw count must be in [1, 100000]")
        result = []
        for _ in range(count):
            if not self.cycle:
                self.cycle = self.rng.permutation(len(CATEGORIES)).tolist()
            label = self.cycle.pop()
            group = self._choose(sorted(self.tree[label]))
            asset = self._choose(sorted(self.tree[label][group]))
            index = self._choose(self.tree[label][group][asset])
            result.append(index)
            self.trace.update(index.to_bytes(8, "little"))
        self.draws += count
        return np.asarray(result, dtype=np.int64)

    def describe(self) -> dict:
        return {"draws": self.draws, "index_trace_sha256": self.trace.hexdigest(),
                "sampling_with_replacement": True, "training_only": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("--clips-per-asset", type=int, default=2)
    parser.add_argument("--min-train-groups", type=int, default=2)
    parser.add_argument("--min-validation-groups", type=int, default=1)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    splits, ingestion = load_reviewed_splits(args.manifest, args.audio_root, clips_per_asset=args.clips_per_asset)
    policy = ReadinessPolicy(min_train_groups_per_category=args.min_train_groups,
                             min_validation_groups_per_category=args.min_validation_groups)
    result = audit_loaded_splits(splits, policy)
    report = {**result.report, "manifest_fingerprint": ingestion["declaration"]["manifest_fingerprint"],
              "wav_bytes_verified": ingestion["wav_bytes_verified"]}
    print(json.dumps(report, allow_nan=False, indent=2))
    if args.require_ready and not report["readiness_gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
