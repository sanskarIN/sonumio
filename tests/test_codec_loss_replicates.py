"""Provenance, source-level numerical and independent-seed codec replication tests."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.codec_loss_study import run_study, PROTOCOL as SINGLE_PROTOCOL
from research.codec_loss_replicates import (
    PROTOCOL, _config, _verify_single, _seed_list,
    replicate_studies,
)
from research.codec_objectives import (
    CodecObjectiveWeights, CodecSpectralConfig, COMPONENT_NAMES,
)


def common() -> dict:
    return dict(
        updates=1, training_batch=2, learning_rate=0.001,
        training_seeds=(0,), heldout_seeds=(101,),
    )


def settings(seed=17) -> dict:
    return common() | {"model_seed": seed}


def verified_params(seed=17) -> dict:
    return _config(
        seed=seed, **common(),
        weights=CodecObjectiveWeights(),
        spectral=CodecSpectralConfig(),
    )


def options() -> dict:
    return common() | {"model_seeds": (17, 29)}


@pytest.fixture(scope="module")
def first_run():
    return run_study(**settings())


@pytest.fixture(scope="module")
def independent_runs():
    """Two real neural experiments, not just synthesized fixture metadata."""
    return {
        seed: run_study(**settings(seed))
        for seed in (17, 29)
    }


@pytest.mark.parametrize("seeds", [
    (), (17,), (17, 17), (17, 29, 41, 43, 47),
    (17, True), (-1, 29), (2**32, 29),
    [17, 29], (17.0, 29),
])
def test_invalid_model_replicate_seeds_rejected(seeds):
    with pytest.raises(ValueError):
        _seed_list(seeds)


def test_genuine_first_party_codec_report_verifies_all_heldout_groups(first_run):
    verified = _verify_single(
        first_run, seed=17,
        expected=verified_params(),
        previous_acoustics=None,
    )
    assert len(verified["source_identity"]) == 3
    assert verified["source_identity"][0] != verified["source_identity"][1]
    assert len(verified["source_identity"][2]) == 21
    assert len(verified["model_initial"]) == 64
    assert set(verified["model_trained"]) == {"original_loss", "multiresolution_loss"}
    assert all(len(v) == 64 for v in verified["model_trained"].values())
    assert set(verified["component_effects"]) == set(COMPONENT_NAMES)
    assert all(math.isfinite(x) for x in verified["component_effects"].values())
    assert math.isfinite(verified["waveform_mae_effect"])
    assert math.isfinite(verified["waveform_rmse_effect"])
    assert math.isfinite(verified["waveform_snr_effect"])
    assert verified["config_sha"] == first_run["configuration_sha256"]


@pytest.mark.parametrize(("path", "value"), [
    (("protocol",), "wrong-protocol"),
    (("configuration", "model_seed"), 18),
    (("configuration", "training_seeds"), [0, 0]),
    (("configuration_sha256",), "a" * 64),
    (("data", "training_recordings"), 1),
    (("data", "heldout_recordings"), 12),
    (("data", "english_hindi_captions_per_split"), 100),
    (("data", "train_fingerprint"), "x" * 64),
    (("data", "heldout_audio_used_in_training"), True),
    (("data", "exact_recording_overlap"), True),
    (("models", "identical_initial_sha256"), "x" * 64),
    (("models", "trained_model_sha256", "original_loss"), "7" * 12),
    (("models", "unchanged_during_evaluation"), False),
    (("paired_controls", "same_initial_parameters"), False),
    (("paired_controls", "same_training_minibatches"), False),
    (("paired_controls", "only_reconstruction_training_loss_differs"), False),
    (("training_losses_not_directly_comparable", "original_loss", "count"), 0),
    (("training_losses_not_directly_comparable", "multiresolution_loss", "first"), float("nan")),
    (("heldout_reconstruction_diagnostics",
      "independent_acoustic_recordings_scored"), 0),
    (("heldout_reconstruction_diagnostics",
      "bilingual_caption_duplicates_skipped"), 0),
    (("heldout_reconstruction_diagnostics",
      "source_combination_families"), 7),
    (("heldout_reconstruction_diagnostics",
      "matched_heldout_waveform_for_both_codecs"), False),
    (("global_torch_rng_isolated",), False),
    (("no_model_weights_or_audio_saved",), False),
    (("model_publication_authorized",), True),
    (("perceptual_fidelity_or_music_speech_quality_established",), True),
])
def test_reject_bad_model_source_or_publication_evidence(first_run, path, value):
    forged = copy.deepcopy(first_run)
    node = forged
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        _verify_single(
            forged, seed=17, expected=verified_params(),
            previous_acoustics=None,
        )


@pytest.mark.parametrize(("path", "value"), [
    (("recording_seed",), 999),
    (("acoustic_group",), "invented"),
    (("heldout_sha256",), "z" * 64),
    (("independent_acoustic_group_within_this_seed",), False),
    (("sample_aligned_mae", "original_loss"), float("nan")),
    (("sample_aligned_mae", "original_loss"), -1),
    (("sample_aligned_rmse", "multiresolution_loss"), float("inf")),
    (("sample_aligned_snr_db", "original_loss"), float("inf")),
    (("multiresolution_minus_original_components", "waveform_l1"), float("nan")),
])
def test_per_recording_numerical_forgery_detected(first_run, path, value):
    forged = copy.deepcopy(first_run)
    row = forged["heldout_reconstruction_diagnostics"]["per_recording"][0]
    node = row
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        _verify_single(
            forged, seed=17,
            expected=verified_params(), previous_acoustics=None,
        )


def test_omitted_duplicate_or_mislabeled_heldout_recording_detected(first_run):
    missing = copy.deepcopy(first_run)
    missing["heldout_reconstruction_diagnostics"]["per_recording"].pop()
    with pytest.raises(ValueError):
        _verify_single(
            missing, seed=17,
            expected=verified_params(), previous_acoustics=None,
        )
    duplicate = copy.deepcopy(first_run)
    records = duplicate["heldout_reconstruction_diagnostics"]["per_recording"]
    records[1] = copy.deepcopy(records[0])
    with pytest.raises(ValueError):
        _verify_single(
            duplicate, seed=17,
            expected=verified_params(), previous_acoustics=None,
        )
    wrong_group = copy.deepcopy(first_run)
    wrong_group["heldout_reconstruction_diagnostics"]["per_recording"][1][
        "acoustic_group"] = "0"
    with pytest.raises(ValueError):
        _verify_single(
            wrong_group, seed=17,
            expected=verified_params(), previous_acoustics=None,
        )


def test_forged_aggregate_waveform_metrics_rejected(first_run):
    for arm in ("original_loss", "multiresolution_loss"):
        for name in ("sample_aligned_mae", "sample_aligned_rmse", "snr_db"):
            damaged = copy.deepcopy(first_run)
            stats = damaged[
                "heldout_reconstruction_diagnostics"]["aggregate_waveform_metrics"]
            stats[arm][name] += 0.123
            with pytest.raises(ValueError, match="summary"):
                _verify_single(
                    damaged, seed=17, expected=verified_params(),
                    previous_acoustics=None,
                )


def test_forged_source_family_delta_summary_rejected(first_run):
    for field in (
        "mean_over_acoustic_families", "positive_family_deltas",
        "negative_family_deltas",
    ):
        damaged = copy.deepcopy(first_run)
        summary = damaged[
            "heldout_reconstruction_diagnostics"
        ]["source_family_deltas_multires_minus_original"]["waveform_l1"]
        if field == "mean_over_acoustic_families":
            summary[field] += 0.5
        else:
            summary[field] = 99
        with pytest.raises(ValueError):
            _verify_single(
                damaged, seed=17, expected=verified_params(),
                previous_acoustics=None,
            )


def test_full_source_fingerprint_must_match_across_model_seeds(first_run):
    identity = (
        "0" * 64,
        first_run["data"]["heldout_fingerprint"],
        tuple(first_run["data"]["source_recording_sha256_heldout"]),
    )
    with pytest.raises(ValueError, match="drifted"):
        _verify_single(
            first_run, seed=17, expected=verified_params(),
            previous_acoustics=identity,
        )


def test_independent_two_seed_training_produces_only_descriptive_model_level_metrics(
    independent_runs,
):
    def fixture_runner(*, model_seed, **kwargs):
        assert kwargs["updates"] == 1
        return copy.deepcopy(independent_runs[model_seed])

    report = replicate_studies(**options(), runner=fixture_runner)
    assert report["protocol"] == PROTOCOL
    assert report["configuration"]["model_seeds"] == [17, 29]
    assert len(report["configuration_sha256"]) == 64
    assert report["independent_model_seed_count"] == 2
    assert len(report["replicated_model_training_runs"]) == 2
    assert report["matched_source_recordings_across_seeds"] is True
    assert report["source_recording_groups_and_bilingual_duplicates_validated"] is True
    assert report["identical_model_architectures_and_initialization_per_pair"]
    assert report["independent_initial_model_weight_hashes_verified"]
    assert report["independent_final_model_weight_hashes_verified"]
    assert report["model_seeds_not_caption_or_audio_sampling_seeds_are_replicates"]
    assert report["perceptual_or_open_domain_quality_claim"] is False
    assert report["statistical_significance_or_model_winner_declared"] is False
    assert report["no_weights_or_audio_written"] is True
    assert report["model_publication_authorized"] is False
    states = report["replicated_model_training_runs"]
    assert len({x["model_initial_sha256"] for x in states}) == 2
    for arm in ("original_loss", "multiresolution_loss"):
        assert len({x["trained_model_sha256"][arm] for x in states}) == 2
    descriptors = report["descriptive_between_model_seed_effects"]
    assert set(COMPONENT_NAMES).issubset(descriptors)
    for descriptor in descriptors.values():
        assert descriptor["count"] == 2
        assert math.isfinite(descriptor["mean"])
        assert math.isfinite(descriptor["sample_stddev"])
        assert descriptor["min"] <= descriptor["mean"] <= descriptor["max"]
    encoded = json.dumps(report, allow_nan=False, ensure_ascii=False)
    assert len(encoded) < 22_000
    assert "base64" not in encoded


def test_duplicate_initialization_cannot_fake_independent_seeds(first_run):
    def fake_runner(*, model_seed, **kwargs):
        result = copy.deepcopy(first_run)
        result["configuration"]["model_seed"] = model_seed
        result["configuration_sha256"] = hashlib.sha256(json.dumps(
            result["configuration"], sort_keys=True,
            ensure_ascii=False, allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return result

    with pytest.raises(ValueError, match="identical initial"):
        replicate_studies(**options(), runner=fake_runner)


@pytest.mark.parametrize("change", [
    {"model_seeds": (17,)},
    {"model_seeds": (17, 17)},
    {"model_seeds": (17, True)},
    {"model_seeds": (17, 2**32)},
    {"updates": 0}, {"updates": 65},
    {"training_batch": 0}, {"training_batch": 9},
    {"learning_rate": float("nan")},
    {"training_seeds": (0, 0)},
    {"heldout_seeds": (0,)},
    {"weights": {"waveform": 0.5}},
    {"spectral": {"fft_sizes": (256,)}},
    {"runner": False},
])
def test_reject_bad_replication_configuration(change):
    with pytest.raises(ValueError):
        replicate_studies(**(options() | change))
