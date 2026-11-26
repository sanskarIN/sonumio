"""Development-only replication threshold stress test on reviewed archive crops.

Compare known transformed training members and nonmember validation controls
against the same complete eligible training gallery. Select one anchor per
recording group to avoid overweighting groups with many crops. No threshold
is fitted, no test split scored and no model training or generation occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np

from research.archive_replication_screen import (
    FRAMES, SAMPLE_RATE, ScreenPolicy, Signature, signature, similarities,
)

PROTOCOL = "sonumio-replication-control-study-v1"
MAX_GROUPS_PER_SPLIT = 64
EDIT_NAMES = ("identity", "gain_dc_polarity", "shift_64", "shift_128",
              "noise_30db", "noise_15db", "lowpass_3tap", "pcm16")
THRESHOLDS = (.9, .95, .98, .99, .995, .999, .9995)


@dataclass(frozen=True, slots=True)
class ControlPolicy:
    seed: int = 61
    pair_budget: int = 50_000

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid control study seed")
        ScreenPolicy(pair_budget=self.pair_budget)


def edit_waveform(waveform, edit, *, seed):
    """Deterministic deliberately related examples; no artificial identity noise."""
    if (not isinstance(waveform, np.ndarray) or waveform.shape != (FRAMES,)
            or waveform.dtype != np.float32 or not np.isfinite(waveform).all()
            or np.max(np.abs(waveform)) > 1):
        raise ValueError("Expected finite bounded float32 training crop")
    if edit not in EDIT_NAMES or type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Unknown edit or invalid control seed")
    x = waveform.astype(np.float64)
    if edit == "identity":
        y = x
    elif edit == "gain_dc_polarity":
        y = -.4*x + .03
    elif edit.startswith("shift_"):
        shift = int(edit.split("_")[1])
        y = np.zeros_like(x)
        y[shift:] = x[:-shift]
    elif edit.startswith("noise_"):
        db = int(edit.split("_")[1][:-2])
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(FRAMES)
        noise -= noise.mean()
        rms = np.sqrt(np.mean((x-x.mean())**2))
        noise *= rms * 10**(-db/20) / np.sqrt(np.mean(noise**2))
        y = x+noise
    elif edit == "lowpass_3tap":
        y = np.convolve(x, [.25, .5, .25], mode="same")
    elif edit == "pcm16":
        y = np.clip(np.rint(x*32768), -32768, 32767)/32768
    else:
        raise AssertionError("Unimplemented edit")
    # No clamping except the explicitly defined PCM16 encoding.
    return y.astype(np.float32)


def _anchors(eligible):
    """Pick the lowest crop index in each group, independent of shard layout."""
    result = {}
    for item in eligible:
        row = item.row
        if row.group_id not in result or row.index < result[row.group_id].row.index:
            result[row.group_id] = item
    if not 1 <= len(result) <= MAX_GROUPS_PER_SPLIT:
        raise ValueError("Control study group count exceeds supported range")
    return {item.row.index for item in result.values()}


@dataclass(slots=True)
class _Query:
    panel: str
    features: Signature | None
    source_index: int | None
    nearest: list[float | None] = field(default_factory=lambda: [None, None])
    known: list[float | None] = field(default_factory=lambda: [None, None])
    seen_source: bool = False
    best_joint: float | None = None
    known_joint: float | None = None

    def compare(self, reference, index):
        own = self.source_index is not None and index == self.source_index
        if own:
            if self.seen_source:
                raise ValueError("Duplicate source anchor")
            self.seen_source = True
        if self.features is None or reference is None:
            return
        scores = similarities(self.features, reference)
        if all(value is not None for value in scores):
            joint = min(scores)
            self.best_joint = joint if self.best_joint is None else max(self.best_joint, joint)
            if own:
                self.known_joint = joint
        for i, value in enumerate(scores):
            if value is None:
                continue
            if not math.isfinite(value):
                raise ValueError("Nonfinite control similarity")
            self.nearest[i] = value if self.nearest[i] is None else max(self.nearest[i], value)
            if own:
                self.known[i] = value


def _metric(values, thresholds):
    scored = [v for v in values if v is not None]
    return {
        "queries": len(values), "scorable": len(scored),
        "unscorable": len(values)-len(scored),
        "minimum": min(scored) if scored else None,
        "median": float(np.median(scored)) if scored else None,
        "maximum": max(scored) if scored else None,
        "thresholds": [
            {"threshold": t, "candidate_count": sum(v >= t for v in scored),
             "fraction_of_scorable": sum(v >= t for v in scored)/len(scored) if scored else None}
            for t in thresholds
        ],
    }


def _panel_report(queries):
    result = {}
    for i, name in enumerate(("waveform", "mel")):
        entry = {"nearest_gallery": _metric([q.nearest[i] for q in queries], THRESHOLDS)}
        if all(q.source_index is not None for q in queries):
            entry["known_source"] = _metric([q.known[i] for q in queries], THRESHOLDS)
        result[name] = entry
    result["joint_same_reference"] = {
        "nearest_gallery": _metric([q.best_joint for q in queries], THRESHOLDS),
    }
    if all(q.source_index is not None for q in queries):
        result["joint_same_reference"]["known_source"] = _metric(
            [q.known_joint for q in queries], THRESHOLDS)
    return result


def study_controls(reader, readiness, *, policy=None):
    """Aggregate stress-test diagnostics, retaining query signatures only.

    All eligible training gallery rows are read and verified. Test split
    data is never used for scoring, although archive/readiness verification
    covers the whole archive. The complete pair budget is checked before
    signal feature extraction.
    """
    from research.archive_signal_reference import _audited_rows, _require_ready

    policy = ControlPolicy() if policy is None else policy
    if not isinstance(policy, ControlPolicy):
        raise ValueError("Expected ControlPolicy")
    _require_ready(reader, readiness)
    meta = reader.report()
    if meta["sample_rate"] != SAMPLE_RATE or meta["clip_frames"] != FRAMES:
        raise ValueError("Unsupported archive crop geometry")
    train_ids = _anchors(readiness.eligible["train"])
    validation_ids = _anchors(readiness.eligible["validation"])
    gallery_count = len(readiness.eligible["train"])
    query_count = len(train_ids)*len(EDIT_NAMES)+len(validation_ids)
    expected_pairs = gallery_count*query_count
    if expected_pairs > policy.pair_budget:
        raise ValueError("Control study exceeds complete gallery pair budget")

    queries = []
    for row, clip in _audited_rows(reader, readiness, "train"):
        if row.index in train_ids:
            seed = (policy.seed << 32) | row.index
            for name in EDIT_NAMES:
                features = signature(edit_waveform(clip, name, seed=seed))
                queries.append(_Query(name, features, row.index))
    for row, clip in _audited_rows(reader, readiness, "validation"):
        if row.index in validation_ids:
            queries.append(_Query("validation_nonmember", signature(clip), None))
    if len(queries) != query_count:
        raise ValueError("Incomplete control query panel")

    compared = 0
    observed_gallery = 0
    scorable_gallery = 0
    for row, clip in _audited_rows(reader, readiness, "train"):
        observed_gallery += 1
        reference = signature(clip)
        scorable_gallery += int(reference is not None)
        for query in queries:
            query.compare(reference, row.index)
            compared += 1
    if (compared != expected_pairs or observed_gallery != gallery_count
            or any(q.source_index is not None and not q.seen_source for q in queries)):
        raise ValueError("Incomplete control comparison")
    if reader.report()["archive_sha256"] != meta["archive_sha256"]:
        raise ValueError("Archive identity changed")

    names = EDIT_NAMES + ("validation_nonmember",)
    panels = {}
    for name in names:
        panel = [q for q in queries if q.panel == name]
        expected = len(validation_ids) if name == "validation_nonmember" else len(train_ids)
        if len(panel) != expected:
            raise ValueError("Control panel count mismatch")
        panels[name] = _panel_report(panel)
    return {
        "protocol": PROTOCOL,
        "archive_sha256": meta["archive_sha256"],
        "eligible_training_sha256": readiness.eligible_fingerprints["train"],
        "eligible_validation_sha256": readiness.eligible_fingerprints["validation"],
        "sample_rate": SAMPLE_RATE, "clip_frames": FRAMES,
        "seed": policy.seed, "pair_budget": policy.pair_budget,
        "planned_pairs": expected_pairs, "examined_pairs": compared,
        "gallery_crops": observed_gallery, "scorable_gallery_crops": scorable_gallery,
        "training_anchor_groups": len(train_ids),
        "validation_anchor_groups": len(validation_ids),
        "anchor_selection": "lowest_eligible_crop_index_per_recording_group",
        "panels": panels,
        "thresholds_fitted": False, "test_split_scored": False,
        "all_eligible_training_crops_compared": True,
        "validation_independence_established": False,
        "memorization_established": False, "files_written": 0,
    }


def main(argv=None):
    from research.archive_readiness import audit_archive
    from research.corpus_archive import CorpusArchiveReader
    from research.corpus_archive_format import require_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--pair-budget", type=int, default=50_000)
    args = parser.parse_args(argv)
    try:
        policy = ControlPolicy(args.seed, args.pair_budget)
        digest = require_sha256(args.expected_sha256)
        with CorpusArchiveReader(args.archive, expected_sha256=digest) as reader:
            result = study_controls(reader, audit_archive(reader), policy=policy)
    except (ValueError, OSError, RuntimeError):
        parser.exit(2, "Replication control study failed verification or budget.\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
