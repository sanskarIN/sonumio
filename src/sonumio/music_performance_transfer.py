"""Verified migration of sample-clock expression overlays to revised scores.

Expressions address canonical event indices, which can change when an
editor adds or removes events. Migration resolves each source event by
an explicit identity policy, records every mapping and validates the
result against the new score's sample-clock polyphony.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

from .arrangement import Arrangement, arrangement_digest, TRACKS
from .errors import GenerationError
from .music_performance import (
    Performance, Expression, compile_performance, _digest,
)

PROTOCOL = "sonumio-performance-transfer-v1"
MATCH_MODES = ("exact", "geometry")
REMOVAL_MODES = ("reject", "drop")
CLOCK_MODES = ("identical", "sample-offset")


@dataclass(frozen=True, slots=True)
class TransferPolicy:
    match: str = "exact"
    missing: str = "reject"
    clock: str = "identical"

    def __post_init__(self):
        if type(self.match) is not str or self.match not in MATCH_MODES:
            raise GenerationError("Unknown performance match policy")
        if type(self.missing) is not str or self.missing not in REMOVAL_MODES:
            raise GenerationError("Unknown missing-expression policy")
        if type(self.clock) is not str or self.clock not in CLOCK_MODES:
            raise GenerationError("Unknown performance clock policy")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _geometry(event):
    return (event.track, event.pitch, event.tick, event.length)


def _event_identity(event, mode):
    return _canonical(asdict(event) if mode == "exact" else _geometry(event))


def _index(score, mode):
    result = {}
    for index, event in enumerate(score.events):
        key = _event_identity(event, mode)
        result.setdefault(key, []).append(index)
    return result


def _clock(score):
    return (score.plan.bpm, round(score.plan.swing*1000))


def _frames(score, event):
    return score.plan.frame(event.tick), score.plan.frame(event.tick+event.length)


def _receipt(source, target, performance, policy, rows, conflicts, result):
    record = {
        "protocol": PROTOCOL,
        "source_score_sha256": arrangement_digest(source),
        "target_score_sha256": arrangement_digest(target),
        "source_performance_sha256": performance.sha256,
        "result_performance_sha256": result.sha256 if result is not None else None,
        "policy": asdict(policy),
        "mappings": rows,
        "conflicts": conflicts,
        "transferred": sum(row["status"] == "mapped" for row in rows),
        "dropped": sum(row["status"] == "dropped" for row in rows),
        "clock_changed": _clock(source) != _clock(target),
        "percussion_seed_changed": source.plan.seed != target.plan.seed,
        "result_validated": result is not None,
        "symbolic_scores_changed": False,
        "audio_rendered": False,
        "authenticated": False,
    }
    return {**record, "receipt_sha256": hashlib.sha256(_canonical(record)).hexdigest()}


def transfer_performance(source, target, performance, policy=TransferPolicy()):
    """Return (validated overlay or None, deterministic diagnostic receipt).

    Exact mode preserves all original event fields. Geometry mode allows
    velocity and pan edits, but never changes track, pitch, onset or
    duration. The original expression remains relative to the revised
    event's velocity and pan in that mode.

    Identical clock mode requires the same quantized swing and BPM.
    Sample-offset mode explicitly retains absolute sample offsets even
    if the clock changes. It does not scale offsets to musical beats.
    """
    if not isinstance(source, Arrangement) or not isinstance(target, Arrangement):
        raise GenerationError("Expected validated source and target scores")
    if not isinstance(performance, Performance) or not isinstance(policy, TransferPolicy):
        raise GenerationError("Expected performance and transfer policy")
    # Reject stale pins and invalid source overlays before any mapping.
    old_notes = compile_performance(source, performance)
    target_index = _index(target, policy.match)
    source_index = _index(source, policy.match)
    rows, conflicts, expressions = [], [], []
    if policy.clock == "identical" and _clock(source) != _clock(target):
        conflicts.append({"domain": "clock", "reason": "score BPM or quantized swing changed"})
    used = set()
    for expression in performance.expressions:
        old_index = expression.index
        event = source.events[old_index]
        key = _event_identity(event, policy.match)
        candidates = target_index.get(key, ())
        row = {
            "source_index": old_index, "target_index": None,
            "event_identity_sha256": hashlib.sha256(key).hexdigest(),
            "track": event.track, "pitch": event.pitch,
            "tick": event.tick, "length": event.length,
        }
        if len(source_index[key]) != 1 or len(candidates) > 1:
            row["status"] = "conflict"
            conflicts.append({"domain": "expression", "source_index": old_index,
                              "reason": "ambiguous event identity"})
        elif not candidates:
            row["status"] = "dropped" if policy.missing == "drop" else "conflict"
            if policy.missing == "reject":
                conflicts.append({"domain": "expression", "source_index": old_index,
                                  "reason": "matching target event missing"})
        else:
            index = candidates[0]
            updated = target.events[index]
            if index in used:
                row["status"] = "conflict"
                conflicts.append({"domain": "expression", "source_index": old_index,
                                  "reason": "target event already mapped"})
            else:
                used.add(index)
                row["target_index"] = index
                row["status"] = "mapped"
                row["base_velocity_changed"] = event.velocity != updated.velocity
                row["base_pan_changed"] = event.pan != updated.pan
                row["source_base_frames"] = list(_frames(source, event))
                row["target_base_frames"] = list(_frames(target, updated))
                row["source_performed_frames"] = [old_notes[old_index].start, old_notes[old_index].stop]
                expressions.append(replace(expression, index=index))
        rows.append(row)
    result = None
    if not conflicts:
        candidate = Performance(arrangement_digest(target),
                                tuple(sorted(expressions, key=lambda item: item.index)))
        try:
            notes = compile_performance(target, candidate)
        except GenerationError as exc:
            conflicts.append({"domain": "target_performance", "reason": str(exc)})
        else:
            result = candidate
            for row in rows:
                if row["status"] == "mapped":
                    note = notes[row["target_index"]]
                    row["target_performed_frames"] = [note.start, note.stop]
                    row["target_performed_velocity"] = note.velocity
                    row["target_performed_pan"] = note.pan
    return result, _receipt(source, target, performance, policy, rows, conflicts, result)


def verify_transfer(source, target, original, result, receipt, policy=TransferPolicy()):
    if not isinstance(result, Performance) or type(receipt) is not dict:
        raise GenerationError("Expected performance and transfer receipt")
    expected, report = transfer_performance(source, target, original, policy)
    if expected is None or expected.sha256 != result.sha256:
        raise GenerationError("Transferred performance differs from replay")
    if _canonical(report) != _canonical(receipt):
        raise GenerationError("Performance transfer receipt differs from replay")
    return report


def merge_performances(score, base, left, right):
    """Three-way reconciliation of independent per-event expressions.

    Inputs are pinned to the same immutable score. A conflicting change
    to any field on the same note is intentionally not split into
    separate timing and articulation domains.
    """
    for overlay in (base, left, right):
        compile_performance(score, overlay)
    maps = [{row.index: row for row in overlay.expressions} for overlay in (base, left, right)]
    merged, conflicts, decisions = [], [], []
    for index in sorted(set().union(*(mapping.keys() for mapping in maps))):
        b, l, r = (mapping.get(index) for mapping in maps)
        if l == r:
            chosen, origin = l, "base" if b == l else "both"
        elif b == l:
            chosen, origin = r, "right"
        elif b == r:
            chosen, origin = l, "left"
        else:
            conflicts.append({"index": index, "reason": "different concurrent expression edits"})
            decisions.append({"index": index, "origin": "conflict"})
            continue
        decisions.append({"index": index, "origin": origin, "removed": chosen is None})
        if chosen is not None:
            merged.append(chosen)
    result = None
    if not conflicts:
        candidate = Performance(arrangement_digest(score), tuple(merged))
        try:
            compile_performance(score, candidate)
        except GenerationError as exc:
            conflicts.append({"domain": "combined_performance", "reason": str(exc)})
        else:
            result = candidate
    record = {
        "protocol": "sonumio-performance-merge-v1",
        "score_sha256": arrangement_digest(score),
        "base_performance_sha256": base.sha256,
        "left_performance_sha256": left.sha256,
        "right_performance_sha256": right.sha256,
        "result_performance_sha256": result.sha256 if result is not None else None,
        "decisions": decisions, "conflicts": conflicts,
        "combined_validated": result is not None, "authenticated": False,
    }
    return result, {**record, "receipt_sha256": hashlib.sha256(_canonical(record)).hexdigest()}
