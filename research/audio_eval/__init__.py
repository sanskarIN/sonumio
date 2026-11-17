"""Inspectably reproducible acoustic evaluation without model weights.

These modules use local in-memory first-party or explicitly consent-cleared
reference recordings. No external audio, person likeness, hidden pretrained
embeddings, weights, or model training are bundled or downloaded.
"""

from .reference import (
    AudioReference, EvaluationPair, EvaluationSet, RightsDeclaration,
    group_provenance_report, blind_pair_labels,
)
from .alignment import (
    AlignmentResult, align_pair, estimate_lag, shift_preserving_length,
)
from .metrics import (
    MetricSet, compare_signals, si_sdr_db, spectral_log_distance,
    normalized_correlation, multi_resolution_distance,
)
from .statistics import (
    BootstrapInterval, PairedComparison, grouped_bootstrap,
    paired_sign_flip_test, holm_adjust, group_effect_report,
)

__all__ = [
    "AudioReference", "EvaluationPair", "EvaluationSet", "RightsDeclaration",
    "group_provenance_report", "blind_pair_labels", "AlignmentResult",
    "align_pair", "estimate_lag", "shift_preserving_length",
    "MetricSet", "compare_signals", "si_sdr_db",
    "spectral_log_distance", "normalized_correlation",
    "multi_resolution_distance", "BootstrapInterval",
    "PairedComparison", "grouped_bootstrap", "paired_sign_flip_test",
    "holm_adjust", "group_effect_report",
]
