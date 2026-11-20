"""Exact-stage NumPy version drift localization, no audio files."""
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("torch")

from research import numpy_stage_probe as study


def test_probe_inputs_are_restricted():
    for members, seed in (
        ((0,), True), ((0,), 3), ((0, 0), 0),
        ((99,), 0), ((0,), -1),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            study.trace_one(members, seed)


def test_independent_actual_renderer_matches_manual_stages():
    row = study.trace_one((0, 1), 101)
    assert row["sources"] == ["rain", "thunder"]
    assert row["seed"] == 101
    assert row["crop_offset"] >= 400
    for name in (
        "time_grid", "layer_rain", "layer_thunder", "mix_after_rain",
        "mix_after_thunder", "fade_ramp", "normalized_24k",
        "fir_kernel", "fir_filtered_24k", "downsampled_8k", "cropped_8k",
    ):
        assert len(row["stages"][name]) == 64


@pytest.fixture(scope="module")
def trace():
    return study.build_trace()


def test_complete_stage_probe_and_same_report_comparison(trace):
    study.validate_trace(trace)
    assert trace["protocol"] == study.PROTOCOL
    assert len(trace["recordings"]) == 40
    assert trace["source_recipe_unchanged"] is True
    assert trace["candidate_not_used_for_training"] is True
    assert trace["audio_files_saved"] is False
    assert trace["weights_saved"] is False
    result = study.compare_traces(trace, copy.deepcopy(trace))
    assert result["cases"] == 40
    assert result["fully_identical_cases"] == 40
    assert result["first_divergent_stage_per_case"] == {}
    assert result["same_cpu_model"] is True


def test_stage_hash_mutation_is_detected_even_with_recomputed_aggregate(trace):
    drift = copy.deepcopy(trace)
    row = drift["recordings"][0]
    row["stages"]["fir_kernel"] = "f" * 64
    with pytest.raises(ValueError, match="aggregate"):
        study.validate_trace(drift)
    drift["stage_aggregate"] = study._aggregate(drift["recordings"])
    result = study.compare_traces(trace, drift)
    assert result["fully_identical_cases"] == 39
    assert result["first_divergent_stage_per_case"]["rain@0"] == "fir_kernel"
    assert "fir_kernel" in result["differing_stages"]


def test_stage_identity_grid_and_nonfinite_metadata_fail_closed(trace):
    bad = copy.deepcopy(trace)
    bad["candidate_not_used_for_training"] = False
    with pytest.raises(ValueError, match="metadata"):
        study.validate_trace(bad)
    bad = copy.deepcopy(trace)
    bad["recordings"][0]["crop_offset"] = 0
    with pytest.raises(ValueError, match="offset"):
        study.validate_trace(bad)
    bad = copy.deepcopy(trace)
    bad["recordings"][0]["stages"]["time_grid"] = "invalid"
    with pytest.raises(ValueError, match="hashes"):
        study.validate_trace(bad)
    bad = copy.deepcopy(trace)
    bad["recordings"][0]["seed"] = bad["recordings"][1]["seed"]
    with pytest.raises(ValueError, match="source trace"):
        study.validate_trace(bad)


def test_first_differing_stage_uses_temporal_order_after_sorted_json(trace):
    # JSON sort_keys is used by the CLI and MUST NOT reorder the diagnostic
    # execution order when comparing original and changed stage bytes.
    drift = copy.deepcopy(trace)
    stages = drift["recordings"][0]["stages"]
    stages["layer_rain"] = "f" * 64
    stages["cropped_8k"] = "e" * 64
    drift["stage_aggregate"] = study._aggregate(drift["recordings"])
    encoded = json.dumps(drift, sort_keys=True)
    restored = json.loads(encoded)
    study.validate_trace(restored)
    result = study.compare_traces(trace, restored)
    assert result["first_divergent_stage_per_case"]["rain@0"] == "layer_rain"
    assert result["paired_differing_recordings_by_stage"]["layer_rain"] == 1
    assert result["paired_differing_recordings_by_stage"]["cropped_8k"] == 1
