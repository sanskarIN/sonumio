"""Audited English/Hindi neural alignment replication and false-negative tests."""
from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.audio_text_alignment_replicates import (
    PROTOCOL, METRICS, ALIGNED_LANGUAGES,
    _model_config, _sha, _valid_seeds, _validate_run,
    _validate_retrieval, _expected_source_group,
    replicate_studies,
)
from research.audio_text_alignment_study import run_study
from research.text_audio import SYNTHETIC_COMBINATIONS


def common():
    return {
        "updates": 1, "batch_categories": 4,
        "learning_rate": 0.0005, "temperature": 0.1,
        "training_seeds": (0,), "validation_seeds": (101,),
    }


def settings(seed=17):
    return {"seed": seed} | common()


def expected(seed=17):
    return _model_config(seed=seed, **common())


def options():
    return common() | {"model_seeds": (17, 29)}


@pytest.fixture(scope="module")
def first():
    return run_study(**settings())


@pytest.fixture(scope="module")
def actual_runs():
    return {
        seed: run_study(**settings(seed))
        for seed in (17, 29)
    }


@pytest.mark.parametrize("invalid", [
    (), (17,), (17, 17), (17, True),
    (17, 29, 41, 43, 47), (-1, 29),
    (17.0, 29), ("17", 29), (17, 2**32),
    [17, 29],
])
def test_model_seeds_require_independent_valid_integer_initializations(invalid):
    with pytest.raises(ValueError):
        _valid_seeds(invalid)


def test_real_single_seed_training_validates_original_source_and_ranking(first):
    checked = _validate_run(
        first, seed=17, expected=expected(),
        previous_audio=None,
    )
    assert len(checked["source_identity"]) == 3
    assert checked["source_identity"][0] != checked["source_identity"][1]
    assert len(checked["source_identity"][2]) == 21
    assert len(set(checked["source_identity"][2])) == 21
    assert _sha(checked["initial_model_sha256"])
    assert _sha(checked["trained_model_sha256"])
    assert checked["initial_model_sha256"] != checked["trained_model_sha256"]
    assert set(checked["retrieval"]) == set(ALIGNED_LANGUAGES)
    for language in ALIGNED_LANGUAGES:
        assert set(checked["retrieval"][language]) == set(METRICS)
        for key, value in checked["retrieval"][language].items():
            assert math.isfinite(value)
            if key == "recall_at_1" or key == "recall_at_5":
                assert 0 <= value <= 1
    assert -1 <= checked["bilingual_caption_cosine"] <= 1
    assert checked["loss_first"] > 0
    assert checked["loss_last"] > 0
    assert checked["config_sha"] == first["configuration_sha256"]


@pytest.mark.parametrize(("path", "value"), [
    (("protocol",), "not-original"),
    (("original_model_protocol",), "unknown"),
    (("configuration", "seed"), 29),
    (("configuration", "temperature"), 0.5),
    (("configuration_sha256",), "0" * 64),
    (("data", "unique_training_recordings"), 10),
    (("data", "unique_heldout_recordings"), 10),
    (("data", "training_bilingual_caption_examples"), 1),
    (("data", "train_fingerprint"), "bad"),
    (("data", "train_heldout_exact_audio_overlap"), True),
    (("data", "independent_real_world_categories_proven"), True),
    (("model", "initial_sha256"), "bad"),
    (("model", "trained_sha256"), "0" * 8),
    (("model", "unchanged_during_evaluation"), False),
    (("model", "weights_published"), True),
    (("training", "actual_optimizer_steps"), 0),
    (("training", "same_category_false_negatives_used"), True),
    (("training", "bilingual_audio_recordings_counted_once"), False),
    (("training", "loss_final"), float("nan")),
    (("training", "loss_max"), -1),
    (("heldout_retrieval", "unique_source_categories"), 2),
    (("heldout_retrieval", "same_audio_candidates_for_both_languages"), False),
    (("heldout_retrieval", "heldout_recordings_used_in_training"), True),
    (("heldout_retrieval", "independent_real_world_prompt_semantics_proven"), True),
    (("global_torch_rng_preserved",), False),
    (("no_model_weights_or_audio_saved",), False),
    (("model_publication_authorized",), True),
    (("semantic_generalization_or_perceptual_quality_claim",), True),
])
def test_forged_model_source_training_retrieval_or_rights_rejected(
    first, path, value,
):
    damaged = copy.deepcopy(first)
    node = damaged
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError):
        _validate_run(
            damaged, seed=17,
            expected=expected(), previous_audio=None,
        )


@pytest.mark.parametrize(("field", "value"), [
    ("category_candidates", 0),
    ("rank1_matches", 21),
    ("rank5_matches", 21),
    ("recall_at_1", 10),
    ("recall_at_5", float("nan")),
    ("mean_reciprocal_rank", 100),
    ("mean_matched_minus_other_cosine", float("inf")),
    ("open_vocabulary_or_audio_quality_established", True),
    ("english_hindi_same_recording_not_independent", False),
])
def test_individual_retrieval_ranks_and_claims_checked(first, field, value):
    damaged = copy.deepcopy(first)
    damaged["heldout_retrieval"]["english"][field] = value
    with pytest.raises(ValueError):
        _validate_run(
            damaged, seed=17,
            expected=expected(), previous_audio=None,
        )


def test_rank_arithmetic_recomputed_from_true_category_order():
    good = {
        "category_candidates": 5,
        "rank_per_category": [1, 2, 3, 4, 5],
        "rank1_matches": 1, "rank5_matches": 5,
        "recall_at_1": 0.2, "recall_at_5": 1.0,
        "mean_reciprocal_rank": sum(1 / i for i in range(1, 6)) / 5,
        "mean_matched_minus_other_cosine": 0.1,
        "all_candidates_from_synthetic_dsp_only": True,
        "english_hindi_same_recording_not_independent": True,
        "open_vocabulary_or_audio_quality_established": False,
    }
    observed = _validate_retrieval(good, expected_categories=5)
    assert observed["recall_at_1"] == 0.2
    altered = copy.deepcopy(good)
    altered["rank_per_category"][0] = 5
    with pytest.raises(ValueError, match="rank"):
        _validate_retrieval(altered, expected_categories=5)


def test_source_group_index_follows_exact_family_seed_language_order():
    family = SYNTHETIC_COMBINATIONS[3]
    category = "-".join(str(x) for x in family)
    idx = _expected_source_group(category, 1, "hi", (0, 1))
    assert idx == 3 * 4 + 2 + 1
    assert _expected_source_group(category, 0, "en", (0, 1)) == 3 * 4
    for args in (
        ("invented", 0, "en", (0, 1)),
        (category, 100, "en", (0, 1)),
        (category, 0, "es", (0, 1)),
    ):
        with pytest.raises(ValueError):
            _expected_source_group(*args)


def test_duplicate_semantic_group_or_wrong_audio_index_refused(first):
    actual = first["training"]["category_disjoint_minibatches"][0]
    damaged = copy.deepcopy(first)
    row = damaged["training"]["category_disjoint_minibatches"][0]
    row["source_category_groups"][1] = row["source_category_groups"][0]
    with pytest.raises(ValueError, match="negatives"):
        _validate_run(
            damaged, seed=17,
            expected=expected(), previous_audio=None,
        )
    wrong = copy.deepcopy(first)
    wrong["training"]["category_disjoint_minibatches"][0][
        "selected_unique_recordings"][0]["recording_index"] += 1
    with pytest.raises(ValueError, match="index"):
        _validate_run(
            wrong, seed=17,
            expected=expected(), previous_audio=None,
        )
    assert len(set(actual["source_category_groups"])) == 4


def test_duplicate_caption_or_incorrect_language_source_rejected(first):
    for change in ("language", "seed", "category"):
        damaged = copy.deepcopy(first)
        chosen = damaged["training"][
            "category_disjoint_minibatches"][0][
                "selected_unique_recordings"][0
            ]
        if change == "language":
            chosen["caption_language"] = "fr"
        elif change == "seed":
            chosen["recording_seed"] = 999
        else:
            chosen["source_category"] = "invented"
        with pytest.raises(ValueError):
            _validate_run(
                damaged, seed=17,
                expected=expected(), previous_audio=None,
            )


def test_altered_heldout_candidate_audio_or_source_hash_rejected(first):
    damaged = copy.deepcopy(first)
    damaged["heldout_retrieval"]["heldout_category_audio_sha256"][0] = "0" * 64
    with pytest.raises(ValueError, match="Source audio"):
        _validate_run(
            damaged, seed=17, expected=expected(),
            previous_audio=(
                first["data"]["train_fingerprint"],
                first["data"]["validation_fingerprint"],
                tuple(first["heldout_retrieval"]["heldout_category_audio_sha256"]),
            ),
        )
    damaged = copy.deepcopy(first)
    damaged["heldout_retrieval"]["heldout_category_audio_sha256"][1] = (
        damaged["heldout_retrieval"]["heldout_category_audio_sha256"][0]
    )
    with pytest.raises(ValueError):
        _validate_run(
            damaged, seed=17,
            expected=expected(), previous_audio=None,
        )


def test_real_two_model_seed_retrieval_replication_descriptive_only(actual_runs):
    def fixture_runner(*, seed, **kwargs):
        assert kwargs["updates"] == 1
        return copy.deepcopy(actual_runs[seed])

    r = replicate_studies(**options(), runner=fixture_runner)
    assert r["protocol"] == PROTOCOL
    assert len(r["configuration_sha256"]) == 64
    assert r["independent_model_initializations"] == 2
    assert r["configuration"]["model_seeds"] == [17, 29]
    assert len(r["real_independent_model_training_runs"]) == 2
    assert r["source_sha256_and_heldout_candidate_panel_matched_across_runs"] is True
    assert r["unique_initial_and_final_model_sha256_validated"] is True
    assert r["category_disjoint_info_nce_false_negatives_rejected"] is True
    assert r["one_heldout_recording_per_category_and_language_verified"] is True
    assert r["same_audio_used_for_english_and_hindi_retrieval"] is True
    assert r["distinct_caption_tokens_not_extra_audio_recordings"] is True
    assert r["statistical_significance_or_open_vocabulary_semantics_established"] is False
    assert r["human_listening_or_audio_generation_quality_established"] is False
    assert r["no_audio_embeddings_or_model_weights_written"] is True
    assert r["model_publication_authorized"] is False
    runs = r["real_independent_model_training_runs"]
    assert len({x["initial_model_sha256"] for x in runs}) == 2
    assert len({x["trained_model_sha256"] for x in runs}) == 2
    for language in ALIGNED_LANGUAGES:
        stats = r["descriptive_between_initialization_retrieval"][language]
        assert set(stats) == set(METRICS)
        for metric in stats.values():
            assert metric["count"] == 2
            assert math.isfinite(metric["mean"])
            assert math.isfinite(metric["sample_stddev"])
            assert metric["min"] <= metric["mean"] <= metric["max"]
    assert r["descriptive_between_initialization_retrieval"][
        "cross_language_same_recording_cosine"]["count"] == 2
    assert len(json.dumps(r, allow_nan=False, ensure_ascii=False)) < 25_000


def test_repeated_model_weights_cannot_count_as_independent(first):
    def repeated_seed(*, seed, **kwargs):
        fake = copy.deepcopy(first)
        fake["configuration"]["seed"] = seed
        fake["configuration_sha256"] = hashlib.sha256(json.dumps(
            fake["configuration"], ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        return fake

    with pytest.raises(ValueError, match="initialization"):
        replicate_studies(**options(), runner=repeated_seed)


@pytest.mark.parametrize("change", [
    {"model_seeds": (17,)},
    {"model_seeds": (17, 17)},
    {"model_seeds": (True, 29)},
    {"updates": 0}, {"updates": 65},
    {"batch_categories": 0}, {"batch_categories": 17},
    {"temperature": float("nan")},
    {"learning_rate": -1},
    {"training_seeds": (0, 0)},
    {"validation_seeds": (0,)},
    {"runner": False},
])
def test_invalid_replication_settings_fail_before_training(change):
    with pytest.raises(ValueError):
        replicate_studies(**(options() | change))
