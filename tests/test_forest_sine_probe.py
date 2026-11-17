"""NumPy forest chirp numerical localization and original-oracle tests."""
from __future__ import annotations

import copy
import json
import pytest

pytest.importorskip("torch")

from research import forest_sine_probe as f


def test_bounded_identity_and_seed_41_context_grid():
    assert len(f.SEEDS) == 9
    assert len(f.PREFIXES) == 4
    assert len(f.CHIRP_METHODS) == 3
    for seed, prefix in ((41, ("unknown",)), (42, ("thunder",)),
                         (True, ()), (41, ["wind"]), (101, ("forest",))):
        with pytest.raises(ValueError, match="Invalid"):
            f.trace_one(seed, prefix)


@pytest.mark.parametrize("prefix", f.PREFIXES)
def test_seed_41_forest_original_reconstruction_preserves_rng(prefix):
    row = f.trace_one(41, prefix)
    assert row["seed"] == 41
    assert row["prefix"] == list(prefix)
    assert row["original_rng_exact"] is True
    assert row["original_forest_exact"] is True
    assert row["chirp_count"] == 2
    assert row["chirps"][0]["stage_order"][0] == "local_time"
    assert row["chirps"][0]["stage_order"][-1] == "sum_python_math_sin"
    for method in f.CHIRP_METHODS:
        assert len(row["layer_hashes"][method]) == 64


@pytest.fixture(scope="module")
def report():
    return f.build_report()


def test_all_forest_contexts_are_first_party_hash_only(report):
    f.validate_report(report)
    assert report["protocol"] == f.PROTOCOL
    assert len(report["records"]) == 36
    assert report["source_frames"] == 36000
    assert report["original_generator_modified"] is False
    assert report["candidate_used_for_neural_training"] is False
    assert report["globally_new_neural_evaluation_seeds"] is False
    assert report["audio_files_saved"] is False
    assert report["model_weights_saved"] is False
    compared = f.compare_reports(report, json.loads(json.dumps(report, sort_keys=True)))
    assert compared["records"] == 36
    assert compared["same_cpu_model"] is True
    assert compared["first_differences"] == []
    assert set(compared["by_seed"]) == set(map(str, f.SEEDS))
    assert all(row["contexts"] == 4 for row in compared["by_seed"].values())


def test_forest_first_temporal_difference_survives_json_sorting(report):
    changed = copy.deepcopy(report)
    row = next(r for r in changed["records"]
               if r["seed"] == 41 and r["prefix"] == ["thunder"])
    row["chirps"][0]["hashes"]["sine_original"] = "a" * 64
    row["chirps"][0]["hashes"]["contribution_original"] = "b" * 64
    row["layer_hashes"]["original"] = "c" * 64
    compared = f.compare_reports(report, json.loads(json.dumps(changed, sort_keys=True)))
    actual = next(r for r in compared["first_differences"]
                  if r["seed"] == 41 and r["prefix"] == "thunder")
    assert actual["chirps"][0]["first_different_stage"] == "sine_original"
    assert actual["different_layers"] == {
        "original": True, "numpy_float64": False,
        "python_math_sin": False,
    }
    assert compared["by_seed"]["41"]["original_different"] == 1


def test_bad_provenance_and_stage_hashes_are_rejected(report):
    wrong = copy.deepcopy(report)
    wrong["globally_new_neural_evaluation_seeds"] = True
    with pytest.raises(ValueError, match="metadata"):
        f.validate_report(wrong)
    wrong = copy.deepcopy(report)
    wrong["records"][0]["layer_hashes"]["original"] = "bad"
    with pytest.raises(ValueError, match="fingerprint"):
        f.validate_report(wrong)
    wrong = copy.deepcopy(report)
    wrong["records"][0]["chirps"][0]["stage_order"].pop()
    with pytest.raises(ValueError, match="fingerprint"):
        f.validate_report(wrong)
    wrong = copy.deepcopy(report)
    wrong["records"][0]["candidate_errors"]["numpy_float64"]["max_absolute_error"] = float("nan")
    with pytest.raises(ValueError, match="fingerprint"):
        f.validate_report(wrong)
    wrong = copy.deepcopy(report)
    wrong["records"][0]["seed"] = 42
    with pytest.raises(ValueError, match="fingerprint"):
        f.validate_report(wrong)
