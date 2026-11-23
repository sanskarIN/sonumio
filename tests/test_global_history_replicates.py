"""Adversarial and real two-seed full-history neural replication evidence tests."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.global_history_replicates import (
    PROTOCOL, _check_seeds, _verified_run, replicate_studies,
)
from research.global_history_study import (
    run_study, PROTOCOL as SINGLE_PROTOCOL,
)


def common() -> dict:
    return {
        "updates": 1, "prefix_frames": 512,
        "diffusion_steps": 8, "evaluations": 2,
        "generation_seed": 301, "learning_rate": 0.0005,
        "batch_size": 2, "training_seeds": [0],
        "heldout_seeds": [101],
    }


def options() -> dict:
    cfg = common()
    return {
        "model_seeds": (17, 29),
        **(cfg | {
            "training_seeds": tuple(cfg["training_seeds"]),
            "heldout_seeds": tuple(cfg["heldout_seeds"]),
        }),
    }


@pytest.fixture(scope="module")
def seed17():
    settings = common() | {
        "model_seed": 17,
        "training_seeds": (0,), "heldout_seeds": (101,),
    }
    return run_study(**settings)


@pytest.fixture(scope="module")
def replication():
    return replicate_studies(**options())


@pytest.mark.parametrize("seeds", [
    (), (17,), (17, 17), (17, 29, 41, 43, 47),
    (17, True), (-1, 29), (2**32, 29),
    [17, 29], (17.0, 29),
])
def test_invalid_independent_model_seed_sets_rejected(seeds):
    with pytest.raises(ValueError):
        _check_seeds(seeds)


def test_genuine_global_history_study_passes_strict_source_and_model_evidence(seed17):
    checked = _verified_run(
        seed17, seed=17, common=common(),
        expected_source=None,
    )
    assert checked["data_fingerprint"][0] != checked["data_fingerprint"][1]
    assert len(checked["model_initial"]) == 64
    assert len(checked["model_trained"]["full_history"]) == 64
    assert len(checked["model_trained"]["hidden_history"]) == 64
    assert checked["model_trained"]["full_history"] != checked[
        "model_trained"]["hidden_history"]
    assert checked["heldout_mse"]["full_history"] > 0
    assert math.isfinite(checked["effects"]["hidden_minus_full"])
    assert math.isfinite(checked["effects"]["wrong_remote_minus_correct"])
    assert checked["mean_remote_output_sensitivity"] >= 0
    assert len(checked["configuration_sha256"]) == 64


@pytest.mark.parametrize(("path", "value"), [
    (("protocol",), "missing-protocol"),
    (("configuration", "model_seed"), 19),
    (("configuration", "updates"), 32),
    (("configuration_sha256",), "0" * 64),
    (("source_split", "train_independent_recordings"), 1),
    (("source_split", "heldout_caption_examples"), 1),
    (("source_split", "train_fingerprint"), "z" * 64),
    (("source_split", "exact_audio_source_leakage_detected"), True),
    (("source_split", "independent_acoustic_families_verified"), True),
    (("controls", "only_previous_history_information_differs"), False),
    (("controls", "wrong_remote_control_keeps_known_prefix_exact"), False),
    (("neural_model_states", "identical_initial_sha256"), "zz" * 32),
    (("neural_model_states", "trained_sha256", "full_history"), "bad"),
    (("neural_model_states", "distinct_trained_states"), False),
    (("heldout_future_objective", "independent_heldout_recordings"), 1),
    (("heldout_future_objective", "bilingual_caption_examples_scored"), 4),
    (("heldout_future_objective", "wrong_remote_history_has_exactly_same_known_local_overlap"), False),
    (("heldout_future_objective", "mse_by_condition", "full_history"), float("nan")),
    (("heldout_future_objective", "group_descriptive_deltas",
      "wrong_remote_minus_correct", "mean"), 100.0),
    (("global_torch_rng_isolated",), False),
    (("audio_or_model_weights_saved",), True),
    (("model_publication_authorized",), True),
    (("speech_music_or_long_context_quality_established",), True),
])
def test_reject_forged_mixed_or_unethical_research_evidence(seed17, path, value):
    forged = copy.deepcopy(seed17)
    node = forged
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        _verified_run(
            forged, seed=17, common=common(),
            expected_source=None,
        )


def test_mismatched_data_provenance_across_seeds_rejected(seed17):
    known = seed17["source_split"]
    with pytest.raises(ValueError, match="Source bytes"):
        _verified_run(
            seed17, seed=17, common=common(),
            expected_source=(
                "1" * 64, known["heldout_fingerprint"]
            ),
        )


def test_omitted_or_corrupt_generated_future_probe_rejected(seed17):
    broken = copy.deepcopy(seed17)
    broken["numerical_generated_future_probes"].pop()
    with pytest.raises(ValueError, match="sample grid"):
        _verified_run(broken, seed=17, common=common(), expected_source=None)
    broken = copy.deepcopy(seed17)
    broken["numerical_generated_future_probes"][0][
        "exact_local_prefix_identity_in_both_outputs"] = False
    with pytest.raises(ValueError, match="generation"):
        _verified_run(broken, seed=17, common=common(), expected_source=None)


def test_duplicate_model_initializations_not_counted_as_independent(seed17):
    def fake_runner(*, model_seed, **kwargs):
        forged = copy.deepcopy(seed17)
        forged["configuration"]["model_seed"] = model_seed
        forged["configuration_sha256"] = hashlib.sha256(json.dumps(
            forged["configuration"], ensure_ascii=False,
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        return forged

    with pytest.raises(ValueError, match="identical initial weights"):
        replicate_studies(**options(), runner=fake_runner)


def test_real_two_initialization_replicated_training_and_heldout(replication):
    report = replication
    assert report["protocol"] == PROTOCOL
    assert len(report["configuration_sha256"]) == 64
    assert report["configuration"]["model_seeds"] == [17, 29]
    assert report["configuration"]["updates"] == 1
    assert report["configuration"]["training_seeds"] == [0]
    assert report["configuration"]["heldout_seeds"] == [101]
    assert report["configuration"]["single_protocol"] == SINGLE_PROTOCOL
    assert report["model_seeds_counted_as_independent_replicates"] == 2
    assert len(report["replicated_model_runs"]) == 2
    assert report["source_recordings_bitwise_matched_within_runner"] is True
    assert report["distinct_model_initialization_fingerprints_verified"] is True
    assert report["trained_model_weights_distinct_across_seeds"] is True
    assert report["same_training_and_evaluation_configuration_across_runs"] is True
    assert report["bilingual_captions_and_noise_seeds_not_independent_replicates"] is True
    assert report["statistical_significance_or_sound_quality_winner_declared"] is False
    assert report["audio_or_model_weights_saved"] is False
    assert report["model_publication_authorized"] is False
    assert len(set(r["model_initial_sha256"]
                   for r in report["replicated_model_runs"])) == 2
    for row in report["replicated_model_runs"]:
        assert row["model_seed"] in (17, 29)
        assert len(row["configuration_sha256"]) == 64
        assert set(row["paired_heldout_mse"]) == {
            "full_history", "hidden_history", "wrong_remote_history"
        }
        assert all(math.isfinite(x) and x >= 0
                   for x in row["paired_heldout_mse"].values())
        assert math.isfinite(row["remote_history_output_sensitivity_mae"])
    desc = report["descriptive_between_model_seed_effects"]
    assert set(desc) == {
        "hidden_minus_full", "wrong_remote_minus_correct",
        "remote_history_output_sensitivity_mae"
    }
    for name in desc:
        value = desc[name]
        assert value["count"] == 2
        assert math.isfinite(value["mean"])
        assert math.isfinite(value["sample_stddev"])
        assert value["min"] <= value["mean"] <= value["max"]
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
    assert len(encoded) < 30_000
    assert "base64" not in encoded


@pytest.mark.parametrize("change", [
    {"updates": 0}, {"updates": 129},
    {"prefix_frames": 0}, {"prefix_frames": 512.0},
    {"diffusion_steps": 7}, {"evaluations": 1},
    {"batch_size": 0}, {"batch_size": 9},
    {"learning_rate": float("nan")},
    {"training_seeds": (0, 0)},
    {"heldout_seeds": (0,)},
])
def test_replication_rejects_invalid_shared_configuration(change):
    with pytest.raises(ValueError):
        replicate_studies(**(options() | change))
