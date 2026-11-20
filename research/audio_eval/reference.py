"""Strict acoustic evaluation-pair provenance and split separation.

Metadata validation is NOT legal verification of creator rights/consent:
human review and the original license source must remain outside this
prototype. Evaluation recordings should never be silently substituted for
training data. Source-identical copies and repeated group_id across splits
are rejected. Synthetic stress seeds shared with neural training are NOT
a globally untouched test split.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Sequence

import numpy as np

from sonumio.audio_core.contracts import AudioBuffer
from sonumio.audio_core.errors import AudioContractError
from sonumio.audio_core.hashing import canonical_json_bytes, digest_audio

_IDENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
_SPLITS = frozenset({"train", "validation", "test", "stress"})
_KINDS = frozenset({"first-party-synthetic", "owned", "cc0", "direct-permission"})
_LICENSES = frozenset({
    "CC0-1.0", "Apache-2.0", "CC-BY-4.0",
    "proprietary-with-permission", "first-party-test-fixture"
})


@dataclass(frozen=True, slots=True)
class RightsDeclaration:
    """Human-declared rights, never implied to be externally authenticated."""

    source_kind: str
    license_id: str
    rights_reviewed: bool
    voice_present: bool
    creator_permission: bool
    attribution: str = ""
    original_source_url: str = ""

    def __post_init__(self):
        if self.source_kind not in _KINDS:
            raise AudioContractError("Unknown evaluation recording source kind")
        if self.license_id not in _LICENSES:
            raise AudioContractError("Unsupported stated evaluation license")
        for value in (self.rights_reviewed, self.voice_present,
                      self.creator_permission):
            if type(value) is not bool:
                raise AudioContractError("Rights declarations must be booleans")
        if not isinstance(self.attribution, str) or len(self.attribution) > 1024:
            raise AudioContractError("Invalid attribution field")
        if not isinstance(self.original_source_url, str) or len(
            self.original_source_url
        ) > 2048:
            raise AudioContractError("Invalid original source URL")
        if self.source_kind == "direct-permission" and not self.creator_permission:
            raise AudioContractError("Direct-permission source needs declared permission")

    def admissible_for_synthetic_acoustic_test(self) -> bool:
        return bool(self.rights_reviewed and not self.voice_present and (
            self.creator_permission or self.source_kind in (
                "first-party-synthetic", "owned", "cc0"
            )
        ))

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AudioReference:
    """One immutable, checksum-addressed evaluation reference waveform."""

    asset_id: str
    buffer: AudioBuffer
    group_id: str
    split: str
    language: str
    caption: str
    rights: RightsDeclaration
    source_sha256: str | None = None

    def __post_init__(self):
        for name, value in (("asset_id", self.asset_id),
                            ("group_id", self.group_id)):
            if not isinstance(value, str) or _IDENT.fullmatch(value) is None:
                raise AudioContractError("Invalid evaluation " + name)
        if not isinstance(self.buffer, AudioBuffer):
            raise AudioContractError("Reference needs immutable AudioBuffer")
        if self.split not in _SPLITS:
            raise AudioContractError("Unknown evaluation split")
        if self.language not in ("en", "hi", "nonverbal"):
            raise AudioContractError("Unsupported declared evaluation language")
        if not isinstance(self.caption, str) or not 1 <= len(self.caption) <= 800:
            raise AudioContractError("Reference caption must be 1..800 chars")
        if not isinstance(self.rights, RightsDeclaration):
            raise AudioContractError("Reference needs a rights declaration")
        actual = digest_audio(self.buffer)
        if self.source_sha256 is not None and self.source_sha256 != actual:
            raise AudioContractError("Reference source hash does not match actual PCM")
        object.__setattr__(self, "source_sha256", actual)

    def metadata(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "group_id": self.group_id,
            "split": self.split,
            "language": self.language,
            "caption": self.caption,
            "source_sha256": self.source_sha256,
            "source_frames": self.buffer.frames,
            "sample_rate": self.buffer.sample_rate,
            "channels": self.buffer.channels,
            "rights": self.rights.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationPair:
    """Candidate audio compared with reference, never auto-assumed licensed."""

    case_id: str
    reference: AudioReference
    candidate: AudioBuffer
    candidate_engine: str
    candidate_training_split: str = "unknown"

    def __post_init__(self):
        if not isinstance(self.case_id, str) or _IDENT.fullmatch(self.case_id) is None:
            raise AudioContractError("Invalid evaluation case id")
        if not isinstance(self.reference, AudioReference) or not isinstance(
            self.candidate, AudioBuffer
        ):
            raise AudioContractError("EvaluationPair needs validated audio")
        if self.candidate.sample_rate != self.reference.buffer.sample_rate:
            raise AudioContractError("Evaluation source and candidate clocks differ")
        if self.candidate.channels != self.reference.buffer.channels:
            raise AudioContractError("Evaluation source and candidate channels differ")
        if self.candidate.frames != self.reference.buffer.frames:
            raise AudioContractError("Evaluation source and candidate frame counts differ")
        if (not isinstance(self.candidate_engine, str)
                or _IDENT.fullmatch(self.candidate_engine) is None):
            raise AudioContractError("Invalid candidate engine ID")
        if self.candidate_training_split not in _SPLITS | {"unknown"}:
            raise AudioContractError("Unknown candidate training split")

    @property
    def group_id(self) -> str:
        return self.reference.group_id

    @property
    def split(self) -> str:
        return self.reference.split

    def metadata(self) -> dict:
        return {
            "case_id": self.case_id,
            "reference": self.reference.metadata(),
            "candidate_sha256": digest_audio(self.candidate),
            "candidate_engine": self.candidate_engine,
            "candidate_training_split": self.candidate_training_split,
            "requires_external_rights_review": True,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSet:
    """Evaluation split with group and exact-PCM isolation gates."""

    pairs: tuple[EvaluationPair, ...]
    protocol: str = "sonumio-numeric-audio-eval-v1"
    strict_rights: bool = True

    def __post_init__(self):
        if not isinstance(self.pairs, tuple) or not 1 <= len(self.pairs) <= 10_000:
            raise AudioContractError("EvaluationSet needs 1..10000 pairs")
        if any(not isinstance(x, EvaluationPair) for x in self.pairs):
            raise AudioContractError("Expected EvaluationPair entries")
        if self.protocol != "sonumio-numeric-audio-eval-v1":
            raise AudioContractError("Unknown evaluation metric protocol version")
        if type(self.strict_rights) is not bool:
            raise AudioContractError("strict_rights must be boolean")
        seen_cases: set[str] = set()
        seen_assets: dict[str, str] = {}
        seen_hashes: dict[str, str] = {}
        group_splits: dict[str, str] = {}
        for pair in self.pairs:
            ref = pair.reference
            if pair.case_id in seen_cases:
                raise AudioContractError("Duplicate evaluation case ID")
            seen_cases.add(pair.case_id)
            if self.strict_rights and not ref.rights.admissible_for_synthetic_acoustic_test():
                raise AudioContractError("Reference rights declaration not approved for this test")
            if ref.asset_id in seen_assets and seen_assets[ref.asset_id] != ref.split:
                raise AudioContractError("Same asset occurs across evaluation splits")
            seen_assets[ref.asset_id] = ref.split
            if ref.source_sha256 in seen_hashes and seen_hashes[ref.source_sha256] != ref.split:
                raise AudioContractError("Identical PCM source leaked across splits")
            seen_hashes[ref.source_sha256] = ref.split
            if ref.group_id in group_splits and group_splits[ref.group_id] != ref.split:
                raise AudioContractError("Speaker/source group reused across splits")
            group_splits[ref.group_id] = ref.split

    @property
    def fingerprint(self) -> str:
        data = [p.metadata() for p in self.pairs]
        return hashlib.sha256(canonical_json_bytes({
            "protocol": self.protocol, "pairs": data, "strict_rights": self.strict_rights
        })).hexdigest()

    def splits(self) -> dict[str, tuple[EvaluationPair, ...]]:
        return {
            split: tuple(p for p in self.pairs if p.split == split)
            for split in sorted({p.split for p in self.pairs})
        }

    def groups(self) -> dict[str, tuple[EvaluationPair, ...]]:
        return {
            group: tuple(p for p in self.pairs if p.group_id == group)
            for group in sorted({p.group_id for p in self.pairs})
        }

    def metadata(self) -> dict:
        return {
            "protocol": self.protocol, "fingerprint": self.fingerprint,
            "cases": len(self.pairs),
            "unique_groups": len(self.groups()),
            "unique_assets": len({x.reference.asset_id for x in self.pairs}),
            "split_sizes": {
                k: len(v) for k, v in self.splits().items()
            },
            "declared_rights_strict": self.strict_rights,
            "external_rights_authentication_performed": False,
            "claim_of_globally_unseen_neural_evaluation": False,
        }


def group_provenance_report(samples: EvaluationSet) -> dict:
    if not isinstance(samples, EvaluationSet):
        raise AudioContractError("Expected validated EvaluationSet")
    groups = samples.groups()
    return {
        "cases": len(samples.pairs),
        "group_count": len(groups),
        "group_sizes": {k: len(v) for k, v in groups.items()},
        "group_splits": {k: v[0].split for k, v in groups.items()},
        "distinct_candidate_engines": sorted({
            p.candidate_engine for p in samples.pairs
        }),
        "dataset_fingerprint": samples.fingerprint,
        "human_rights_review_still_required": True,
        "neural_training_not_performed": True,
    }


def blind_pair_labels(pair: EvaluationPair, *, seed: int) -> dict:
    """Deterministic anonymized A/B order, not audio publication or quality."""
    import random
    if not isinstance(pair, EvaluationPair) or type(seed) is not int or not 0 <= seed < 2**32:
        raise AudioContractError("Blind order needs valid pair and 32-bit seed")
    decision = random.Random(
        hashlib.sha256(f"{seed}:{pair.case_id}".encode("utf8")).digest()
    ).randint(0, 1)
    return {
        "case_id": pair.case_id,
        "a": "reference" if decision == 0 else "candidate",
        "b": "candidate" if decision == 0 else "reference",
        "blinding_seed": seed,
        "listener_preference_recorded": False,
    }
