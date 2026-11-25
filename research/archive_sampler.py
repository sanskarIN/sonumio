"""Versioned, random-access group-balanced sampling for audited archive clips.

The sampling sequence is derived from SHA256 domain-separated counters,
rather than relying on a mutable NumPy or Python RNG implementation. State
restoration replays the bounded prefix and verifies its trace. This is
sampler replay, not a model/optimizer training checkpoint.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import struct

import numpy as np

from research.archive_readiness import ArchiveReadinessResult, eligible_fingerprint
from research.corpus_archive_format import ArchiveError, canonical_json, require_sha256, MAX_ROWS
from research.wave_diffusion import CATEGORIES

PROTOCOL = "sonumio-counter-group-sampler-v1"
MAX_DRAWS = 400_000
MAX_DRAW_BATCH = 1024
_DOMAIN = b"sonumio-counter-group-sampler-v1\0"
_TRACE = b"sonumio-counter-group-trace-v1\0"


def _strict_int(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ArchiveError(f"Invalid sampler {name}")
    return value


class CounterGroupSampler:
    """Training-only uniform category/group/asset/crop hierarchy.

    Every complete category cycle is a permutation of all categories.
    Absolute draw indices make random-access chunks equal to sequential
    draws. Samples are drawn with replacement at the group/asset/crop levels.
    """

    def __init__(self, readiness: ArchiveReadinessResult, *, seed: int):
        _strict_int(seed, 0, 2**32-1, "seed")
        if not isinstance(readiness, ArchiveReadinessResult):
            raise ArchiveError("Expected archive readiness result")
        if readiness.report.get("readiness_gate_passed") is not True:
            raise ArchiveError("Cannot sample archive that failed readiness")
        train = readiness.eligible["train"]
        fingerprint = eligible_fingerprint(train)
        if fingerprint != readiness.eligible_fingerprints["train"]:
            raise ArchiveError("Eligible training fingerprint changed")
        if fingerprint != readiness.report["eligible_split_fingerprints"]["train"]:
            raise ArchiveError("Training eligibility report mismatch")
        if not 1 <= len(train) <= MAX_ROWS:
            raise ArchiveError("Invalid eligible training row count")
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        seen = set()
        for clip in train:
            row = clip.row
            if row.index in seen or not 0 <= row.label < len(CATEGORIES):
                raise ArchiveError("Invalid eligible training row")
            seen.add(row.index)
            tree[row.label][row.group_id][row.asset_id].append(row.index)
        if set(tree) != set(range(len(CATEGORIES))):
            raise ArchiveError("Training schedule requires every category")
        # Stable sorted identifiers ensure hierarchy order does not depend
        # on dict insertion, hash seed, or the original shard partition.
        self._tree = tuple(
            tuple(
                tuple(tuple(sorted(tree[label][group][asset]))
                      for asset in sorted(tree[label][group]))
                for group in sorted(tree[label])
            )
            for label in range(len(CATEGORIES))
        )
        self.seed = seed
        self.eligible_sha256 = fingerprint
        self.plan_sha256 = hashlib.sha256(canonical_json({
            "protocol": PROTOCOL, "seed": seed,
            "eligible_training_sha256": fingerprint,
            "categories": list(CATEGORIES),
            "sampling": "uniform-category-cycle/group/asset/crop",
        })).hexdigest()
        self._key = bytes.fromhex(self.plan_sha256)
        self._position = 0
        self._trace = hashlib.sha256(_TRACE + self._key)

    @property
    def position(self) -> int:
        return self._position

    def _randbelow(self, count: int, counter: int, purpose: int) -> int:
        # Rejection avoids modulo bias for group/asset counts that are not
        # powers of two. Domain separation makes each hierarchy choice
        # independent of how many retries another choice needed.
        if count == 1:
            return 0
        modulus = 1 << 64
        limit = modulus - modulus % count
        for attempt in range(128):
            raw = hashlib.sha256(
                _DOMAIN + self._key + struct.pack("<QII", counter, purpose, attempt),
            ).digest()
            value = int.from_bytes(raw[:8], "little")
            if value < limit:
                return value % count
        raise ArchiveError("Sampler rejection budget exhausted")

    def category_order(self, cycle: int) -> tuple[int, ...]:
        _strict_int(cycle, 0, MAX_DRAWS, "cycle")
        order = list(range(len(CATEGORIES)))
        for i in range(len(order)-1, 0, -1):
            j = self._randbelow(i+1, cycle, 100+i)
            order[i], order[j] = order[j], order[i]
        return tuple(order)

    def draw_at(self, start: int, count: int) -> np.ndarray:
        _strict_int(start, 0, MAX_DRAWS-1, "offset")
        _strict_int(count, 1, MAX_DRAW_BATCH, "batch count")
        if start + count > MAX_DRAWS:
            raise ArchiveError("Sampler draw budget exceeded")
        result = np.empty(count, dtype=np.int64)
        previous_cycle = -1
        order = ()
        for j, index in enumerate(range(start, start+count)):
            cycle, slot = divmod(index, len(CATEGORIES))
            if cycle != previous_cycle:
                order = self.category_order(cycle)
                previous_cycle = cycle
            label = order[slot]
            groups = self._tree[label]
            assets = groups[self._randbelow(len(groups), index, 1)]
            crops = assets[self._randbelow(len(assets), index, 2)]
            result[j] = crops[self._randbelow(len(crops), index, 3)]
        return result

    def draw(self, count: int) -> np.ndarray:
        indices = self.draw_at(self._position, count)
        self._trace.update(np.asarray(indices, dtype="<i8").tobytes())
        self._position += count
        return indices

    def describe(self) -> dict:
        return {
            "protocol": PROTOCOL, "seed": self.seed, "draws": self._position,
            "plan_sha256": self.plan_sha256,
            "eligible_training_sha256": self.eligible_sha256,
            "index_trace_sha256": self._trace.hexdigest(),
            "training_only": True, "sampling_with_replacement": True,
            "model_optimizer_state_included": False,
        }

    @classmethod
    def restore(cls, readiness: ArchiveReadinessResult, state: dict) -> "CounterGroupSampler":
        """Validate a serialized sampler state against a freshly audited plan."""
        if not isinstance(state, dict):
            raise ArchiveError("Expected serialized sampler state")
        sampler = cls(readiness, seed=state.get("seed"))
        keys = set(sampler.describe())
        if set(state) != keys or state.get("protocol") != PROTOCOL:
            raise ArchiveError("Invalid sampler state fields")
        count = _strict_int(state["draws"], 0, MAX_DRAWS, "restored draw count")
        for key in ("plan_sha256", "eligible_training_sha256", "index_trace_sha256"):
            require_sha256(state[key])
        if (sampler.plan_sha256 != state["plan_sha256"]
                or sampler.eligible_sha256 != state["eligible_training_sha256"]):
            raise ArchiveError("Sampler state belongs to a different audited plan")
        while sampler.position < count:
            sampler.draw(min(MAX_DRAW_BATCH, count-sampler.position))
        if sampler.describe() != state:
            raise ArchiveError("Sampler state trace does not match replay")
        return sampler
