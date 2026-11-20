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


from .protocol import (
    EvaluationPlan, MetricRule, metric_catalog, preregistration_report,
)
from .runner import (
    EvaluationResult, EvaluationRow, run_evaluation,
    compare_engine_reports, numeric_benchmark_summary,
)
from .report_io import (
    NumericReport, NumericReportBundle,
    serialize_numeric_report, deserialize_numeric_report,
    read_numeric_report, write_numeric_report,
)

__all__ = [
    "EvaluationPlan",
    "MetricRule",
    "metric_catalog",
    "preregistration_report",
    "EvaluationResult",
    "EvaluationRow",
    "run_evaluation",
    "compare_engine_reports",
    "numeric_benchmark_summary",
    "NumericReport",
    "NumericReportBundle",
    "serialize_numeric_report",
    "deserialize_numeric_report",
    "read_numeric_report",
    "write_numeric_report",
    "AudioReference", "EvaluationPair", "EvaluationSet", "RightsDeclaration",
    "group_provenance_report", "blind_pair_labels", "AlignmentResult",
    "align_pair", "estimate_lag", "shift_preserving_length",
    "MetricSet", "compare_signals", "si_sdr_db",
    "spectral_log_distance", "normalized_correlation",
    "multi_resolution_distance", "BootstrapInterval",
    "PairedComparison", "grouped_bootstrap", "paired_sign_flip_test",
    "holm_adjust", "group_effect_report",
]
