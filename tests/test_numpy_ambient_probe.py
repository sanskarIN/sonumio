"""Ambient NumPy-version probe tests: precise stage ordering and provenance."""
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("torch")

from research import numpy_ambient_probe as study


@pytest.fixture(scope="module")
def report():
    return study.build_probe()


def test_ambient_expression_matches_original_without_generator_patch(report):
    study.validate_probe(report)
    assert report["protocol"] == study.PROTOCOL
    assert report["sample_rate"] == 24000
    assert report["frames"] == 36000
    assert report["matches_original_drone_exactly"] is True
    assert report["source_generator_modified"] is False
    assert report["audio_files_saved"] is False
    assert report["model_weights_saved"] is False
    assert report["stage_order"][0] == "time_grid"
    assert report["stage_order"][-1] == "original_drone_output"


def test_comparison_preserves_execution_order_after_sorted_json_serialization(report):
    text = json.dumps(report, sort_keys=True)
    restored = json.loads(text)
    study.validate_probe(restored)
    result = study.compare_probes(report, restored)
    assert result["first_divergent_stage"] is None
    assert result["different_stages"] == []
    assert result["same_final_drone_bytes"] is True
    assert result["same_cpu_model"] is True


def test_first_difference_is_temporal_and_reports_other_differences(report):
    mutated = copy.deepcopy(report)
    mutated["stages"]["sin_110"]["sha256"] = "f" * 64
    mutated["stages"]["wave"]["sha256"] = "e" * 64
    mutated["stages"]["original_drone_output"]["sha256"] = "d" * 64
    mutated = json.loads(json.dumps(mutated, sort_keys=True))
    compared = study.compare_probes(report, mutated)
    assert compared["first_divergent_stage"] == "sin_110"
    assert compared["different_stages"] == [
        "sin_110", "wave", "original_drone_output"
    ]
    assert compared["same_final_drone_bytes"] is False


def test_metadata_stage_order_and_corruption_rejected(report):
    broken = copy.deepcopy(report)
    broken["audio_files_saved"] = True
    with pytest.raises(ValueError, match="metadata"):
        study.validate_probe(broken)
    broken = copy.deepcopy(report)
    broken["stage_order"].pop()
    with pytest.raises(ValueError, match="stages"):
        study.validate_probe(broken)
    broken = copy.deepcopy(report)
    broken["stages"]["phase_110"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="stage"):
        study.validate_probe(broken)
    broken = copy.deepcopy(report)
    broken["stages"]["sin_110"]["dtype"] = "complex64"
    with pytest.raises(ValueError, match="stage"):
        study.validate_probe(broken)
