"""Grouped paired uncertainty, sign-flip tests and multiplicity control.

Acoustic clips from the same original recording/group are correlated.
Bootstrap resampling at per-clip level would overstate confidence. This
module requires explicit groups and only resamples their aggregated means.
It does not claim clinical/statistical significance or calibrated MOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from sonumio.audio_core.errors import AudioContractError

# 2**16 = 65,536 complete sign assignments; bounded and practical on CPU.
# A small group study has a discrete p-value floor, not arbitrary precision.
EXACT_SIGN_FLIP_MAX_GROUPS = 16


def _finite_values(values: Sequence[float], *, minimum_count: int = 1
                   ) -> np.ndarray:
    if not isinstance(values, (tuple, list)) or not minimum_count <= len(values) <= 100_000:
        raise AudioContractError("Expected bounded nonempty numerical sample")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise AudioContractError("All evaluation measurements must be finite")
    return np.asarray(values, dtype=np.float64)


def _seed(value: int) -> int:
    if type(value) is not int or not 0 <= value < 2**32:
        raise AudioContractError("Explicit deterministic random seed required")
    return value


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    groups: int
    samples: int
    iterations: int
    seed: int
    resampling_unit: str = "source_group"

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedComparison:
    difference: float
    p_value: float
    permutations: int
    groups: int
    direction: str
    report_protocol: str = "sonumio-grouped-paired-test-v2"
    method: str = "monte_carlo"
    requested_permutations: int = 0
    resampling_unit: str = "source_group"

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def _group_means(values: Sequence[float], groups: Sequence[str]
                 ) -> tuple[np.ndarray, int]:
    rows = _finite_values(values)
    if not isinstance(groups, (tuple, list)) or len(groups) != len(rows):
        raise AudioContractError("Every numerical measurement requires a group")
    grouped = defaultdict(list)
    for value, group in zip(rows, groups):
        if not isinstance(group, str) or not 1 <= len(group) <= 96:
            raise AudioContractError("Invalid source group id")
        grouped[group].append(float(value))
    if len(grouped) > 10000:
        raise AudioContractError("Too many independent acoustic groups")
    means = np.asarray([
        float(np.mean(grouped[k])) for k in sorted(grouped)
    ], dtype=np.float64)
    return means, len(rows)


def grouped_bootstrap(values: Sequence[float], groups: Sequence[str], *,
                      confidence: float = .95,
                      iterations: int = 2000, seed: int = 0
                      ) -> BootstrapInterval:
    level = float(confidence) if type(confidence) in (float, int) else float("nan")
    if not math.isfinite(level) or not .5 <= level < 1:
        raise AudioContractError("Confidence must lie in [.5,1)")
    if type(iterations) is not int or not 100 <= iterations <= 100_000:
        raise AudioContractError("Expected 100..100000 bootstrap repetitions")
    rng = np.random.default_rng(_seed(seed))
    independent, n = _group_means(values, groups)
    count = len(independent)
    if count < 2:
        raise AudioContractError("At least two independent groups needed for uncertainty")
    estimates = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, count, count)
        estimates[i] = float(np.mean(independent[idx]))
    alpha = (1 - level) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return BootstrapInterval(
        float(np.mean(independent)), float(low), float(high),
        level, count, n, iterations, seed,
    )


def paired_sign_flip_test(a: Sequence[float], b: Sequence[float],
                          groups: Sequence[str], *,
                          permutations: int = 5000,
                          seed: int = 0,
                          direction: str = "two-sided") -> PairedComparison:
    """Paired source-group inference: exact small-N, Monte Carlo otherwise.

    The null distribution flips the sign of each INDEPENDENT source-group
    mean once, never individual audio clips. For 2..16 groups all 2**N
    assignments are enumerated, so small studies do not claim precision
    that a seeded Monte Carlo sample cannot establish. With >16 groups
    use bounded Monte Carlo and a plus-one p-value correction.
    """
    if direction not in ("two-sided", "a-greater", "b-greater"):
        raise AudioContractError("Invalid paired permutation direction")
    if type(permutations) is not int or not 100 <= permutations <= 100_000:
        raise AudioContractError("Expected 100..100000 sign-flip permutations")
    _seed(seed)  # Reject invalid provenance seed even for exhaustive inference.
    aa, bb = _finite_values(a), _finite_values(b)
    if aa.shape != bb.shape:
        raise AudioContractError("Paired samples require same case count")
    differences, _ = _group_means((aa - bb).tolist(), groups)
    n = len(differences)
    if n < 2:
        raise AudioContractError("Paired test requires at least two independent groups")
    observed = math.fsum(map(float, differences)) / n
    if not math.isfinite(observed):
        raise AudioContractError("Nonfinite aggregate paired effect")

    # Reuse one threshold rule for exact and randomized paths, including ties.
    # Floating comparisons tolerate roundoff when the observed assignment
    # is reconstructed under another sign configuration.
    tolerance = 1e-12
    def extreme(trial: float) -> bool:
        if direction == "two-sided":
            return abs(trial) >= abs(observed) - tolerance
        if direction == "a-greater":
            return trial >= observed - tolerance
        return trial <= observed + tolerance

    if n <= EXACT_SIGN_FLIP_MAX_GROUPS:
        trials = 1 << n
        extreme_count = 0
        # math.fsum avoids order-dependent accumulation and 2**16 assignments
        # are small enough to enumerate without a large temporary matrix.
        for signs in product((-1.0, 1.0), repeat=n):
            candidate = math.fsum(float(x) * sign
                                  for x, sign in zip(differences, signs)) / n
            extreme_count += extreme(candidate)
        # This is the entire discrete null distribution; no plus-one
        # Monte Carlo correction and no false zero-probability claim.
        p_value = extreme_count / trials
        method = "exact"
    else:
        trials = permutations
        rng = np.random.default_rng(seed)
        extreme_count = 0
        for _ in range(trials):
            signs = rng.choice(np.array([-1.0, 1.0]), size=n)
            candidate = math.fsum(float(x) * float(sign)
                                  for x, sign in zip(differences, signs)) / n
            extreme_count += extreme(candidate)
        # Include the observed configuration to prevent zero Monte Carlo p.
        p_value = (extreme_count + 1) / (trials + 1)
        method = "monte_carlo"
    return PairedComparison(
        observed, p_value, trials, n, direction,
        method=method, requested_permutations=permutations,
    )

def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down FWER adjustment, with original order preserved."""
    p = _finite_values(p_values)
    if np.any((p < 0) | (p > 1)):
        raise AudioContractError("Expected p-values in [0,1]")
    ordered = np.argsort(p)
    adjusted = np.zeros(len(p), dtype=np.float64)
    running = 0.0
    n = len(p)
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (n - rank) * float(p[index])))
        adjusted[index] = running
    return adjusted.tolist()


def group_effect_report(reference: Sequence[float], candidate: Sequence[float],
                        groups: Sequence[str], *, seed: int = 0,
                        confidence: float = .95,
                        bootstrap_iterations: int = 2000,
                        permutations: int = 5000) -> dict:
    """Two-level paired differences with epistemic limits spelled out."""
    before = _finite_values(reference)
    after = _finite_values(candidate)
    if before.shape != after.shape:
        raise AudioContractError("Aligned comparisons need equal metric rows")
    difference = (after - before).tolist()
    interval = grouped_bootstrap(
        difference, groups, confidence=confidence,
        iterations=bootstrap_iterations, seed=seed,
    )
    test = paired_sign_flip_test(
        after.tolist(), before.tolist(), groups,
        permutations=permutations, seed=seed,
    )
    return {
        "protocol": "sonumio-group-level-audio-metric-effect-v2",
        "direction": "candidate-minus-reference",
        "interval": interval.as_dict(),
        "sign_flip": test.as_dict(),
        "independent_groups": interval.groups,
        "measurements": interval.samples,
        "group_pseudoreplication_guard": True,
        "is_human_listening_quality_score": False,
        "external_dataset_rights_verified": False,
        "multiple_comparisons_adjusted": False,
        "caution": (
            "Intervals assume the defined groups are independent. "
            "Exact small-group p-values have a discrete resolution floor; "
            "Monte Carlo p-values have simulation uncertainty. "
            "These tests do not make metric changes perceptually meaningful; "
            "multiple comparisons require prespecified correction."
        ),
    }
