"""Version-specific RNG and sound-layer fingerprint diagnostic regression."""
from __future__ import annotations

import copy
import json

import pytest

pytest.importorskip("torch")

from research import portable_seed_stage_probe as probe


def test_identity_and_provenance_rejects_global_neural_holdout_claim():
    assert len(probe.GRID) == 105
    assert len(probe.METHODS) == 2
    assert probe.TEXT_AUDIO_TRAIN_OVERLAP == (3,)
    for args in [
        (0, (0,), "numpy_float64"),
        (3, (0, 0), "numpy_float64"),
        (7, (99,), "numpy_float64"),
        (17, (0,), "unknown"),
        (True, (0,), "numpy_float64"),
        (3, [0], "numpy_float64"),
    ]:
        with pytest.raises(ValueError, match="Invalid"):
            probe.trace_one(*args)


@pytest.mark.parametrize("method", probe.METHODS)
def test_one_seeded_mixed_waveform_has_rng_and_crop_stages(method):
    row = probe.trace_one(7, (0, 5), method)
    assert row["sources"] == ["rain", "ambient"]
    assert row["seed"] == 7
    assert row["method"] == method
    assert row["stage_order"][0] == "time_grid"
    assert row["stage_order"][1] == "initial_rng_state"
    assert row["stage_order"][-1] == "crop_8k"
    assert "layer_rain" in row["stages"]
    assert "rng_after_rain" in row["stages"]
    assert "layer_drone" in row["stages"]
    assert "rng_after_drone" in row["stages"]
    assert len(row["stages"]["normalized_24k"]) == 64
    assert 400 <= row["crop_offset"] <= 1600


@pytest.fixture(scope="module")
def report():
    return probe.build_report()


def test_complete_portability_stage_corpus_is_originally_scoped(report):
    probe.validate_report(report)
    assert report["protocol"] == probe.PROTOCOL
    assert report["records_per_method"] == 105
    assert len(report["rows"]) == 210
    assert report["text_audio_training_seed_overlap"] == [3]
    assert report["globally_heldout_for_neural_model_evaluation"] is False
    assert report["original_renderer_unchanged"] is True
    assert report["candidate_used_for_neural_training"] is False
    assert report["audio_files_saved"] is False
    assert report["model_weights_saved"] is False
    compared = probe.compare_reports(report, json.loads(json.dumps(report, sort_keys=True)))
    assert compared["fully_matching_method_records"] == 210
    assert compared["total_method_records"] == 210
    assert compared["same_cpu_model"] is True
    assert compared["first_differing_cases_capped_at_32"] == []
    for method in probe.METHODS:
        assert all(
            metrics["cases"] == 21
            and metrics["full_stage_exact"] == 21
            and metrics["different_24k"] == 0
            and metrics["different_8k_crop"] == 0
            and metrics["first_stage_counts"] == {}
            for metrics in compared["by_method_seed"][method].values()
        )


def test_recomputed_valid_mutation_is_localized_to_first_temporal_stage(report):
    changed = copy.deepcopy(report)
    row = next(r for r in changed["rows"]
               if r["method"] == "numpy_float64" and r["seed"] == 3
               and r["sources"] == ["rain"])
    row["stages"]["layer_rain"] = "a" * 64
    row["stages"]["crop_8k"] = "b" * 64
    probe.validate_report(changed)
    compared = probe.compare_reports(report, json.loads(json.dumps(changed, sort_keys=True)))
    by_seed = compared["by_method_seed"]["numpy_float64"]["3"]
    assert compared["fully_matching_method_records"] == 209
    assert by_seed["cases"] == 21
    assert by_seed["different_24k"] == 0
    assert by_seed["different_8k_crop"] == 1
    assert by_seed["first_stage_counts"] == {"layer_rain": 1}
    example = compared["first_differing_cases_capped_at_32"][0]
    assert example["first_divergent_stage"] == "layer_rain"
    assert example["sources"] == ["rain"]
    assert example["seed"] == 3


def test_fail_closed_on_invalid_stage_hash_order_and_metadata(report):
    changed = copy.deepcopy(report)
    changed["globally_heldout_for_neural_model_evaluation"] = True
    with pytest.raises(ValueError, match="metadata"):
        probe.validate_report(changed)
    changed = copy.deepcopy(report)
    changed["rows"][0]["stage_order"] = list(reversed(changed["rows"][0]["stage_order"]))
    # Semantically distinct but metadata-valid order; reject comparison
    probe.validate_report(changed)
    with pytest.raises(ValueError, match="execution order"):
        probe.compare_reports(report, changed)
    changed = copy.deepcopy(report)
    changed["rows"][0]["stages"]["crop_8k"] = "invalid"
    with pytest.raises(ValueError, match="Invalid stage"):
        probe.validate_report(changed)
    changed = copy.deepcopy(report)
    changed["rows"][0]["seed"] = 101
    with pytest.raises(ValueError, match="Invalid stage"):
        probe.validate_report(changed)
    changed = copy.deepcopy(report)
    changed["rows"][0]["stage_order"].pop()
    with pytest.raises(ValueError, match="Invalid stage"):
        probe.validate_report(changed)
