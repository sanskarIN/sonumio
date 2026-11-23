"""Real first-party bilingual audio/text research training and split controls."""
from __future__ import annotations

import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.audio_text_alignment_study import (
    PROTOCOL, CATEGORIES, _settings, _training_indices, run_study,
)
from research.audio_text_alignment import PROTOCOL as MODEL_PROTOCOL
from research.latent_audio import synthetic_bilingual_waves
from research.text_audio import SYNTHETIC_COMBINATIONS


def tiny() -> dict:
    return {
        "seed": 17, "updates": 1,
        "batch_categories": 4,
        "learning_rate": 0.0005, "temperature": 0.1,
        "training_seeds": (0,), "validation_seeds": (101,),
    }


@pytest.fixture(scope="module")
def study():
    return run_study(**tiny())


@pytest.mark.parametrize("bad", [
    {"seed": -1}, {"seed": True}, {"seed": 2**32},
    {"updates": 0}, {"updates": 65}, {"updates": True},
    {"batch_categories": 1}, {"batch_categories": 17},
    {"batch_categories": True},
    {"learning_rate": 0}, {"learning_rate": -0.01},
    {"learning_rate": 10}, {"learning_rate": True},
    {"learning_rate": float("nan")},
    {"temperature": 0}, {"temperature": 0.01},
    {"temperature": 1.5}, {"temperature": True},
    {"temperature": float("inf")},
    {"training_seeds": ()}, {"validation_seeds": ()},
    {"training_seeds": (0, 0)},
    {"validation_seeds": (101, 101)},
    {"training_seeds": (0,), "validation_seeds": (0,)},
    {"training_seeds": [0]},
    {"validation_seeds": (True,)},
    {"validation_seeds": (2**32,)},
])
def test_bad_bilingual_training_settings_fail_before_work(bad):
    with pytest.raises(ValueError):
        _settings(**(tiny() | bad))


def test_category_disjoint_batches_are_reproducible_and_never_duplicate_labels():
    for batch_count in (2, 4, 8, 16):
        a = torch.Generator(device="cpu").manual_seed(66)
        b = torch.Generator(device="cpu").manual_seed(66)
        first = _training_indices(
            batch_categories=batch_count,
            recording_seeds=(0, 1),
            generator=a,
        )
        repeated = _training_indices(
            batch_categories=batch_count,
            recording_seeds=(0, 1),
            generator=b,
        )
        assert torch.equal(first[0], repeated[0])
        assert first[1:] == repeated[1:]
        indices, keys, metadata = first
        assert len(indices) == len(keys) == len(metadata) == batch_count
        assert len(set(keys)) == len(keys)
        assert len(set(indices.tolist())) == batch_count
        for index, key, row in zip(indices.tolist(), keys, metadata):
            assert row["source_category"] == key
            assert row["recording_seed"] in (0, 1)
            assert row["caption_language"] in ("en", "hi")
            assert row["recording_index"] == index
            assert 0 <= index < 21 * 2 * 2


def test_real_training_model_states_source_rights_and_nonpublication(study):
    report = study
    assert report["protocol"] == PROTOCOL
    assert report["original_model_protocol"] == MODEL_PROTOCOL
    assert len(report["configuration_sha256"]) == 64
    settings = report["configuration"]
    assert settings["seed"] == 17
    assert settings["updates"] == 1
    assert settings["batch_categories"] == 4
    assert settings["training_seeds"] == [0]
    assert settings["validation_seeds"] == [101]
    assert settings["category_combinations"] == len(SYNTHETIC_COMBINATIONS)
    assert settings["audio_sample_rate"] == 8000
    assert settings["audio_frames"] == 8192
    data = report["data"]
    assert data["unique_training_recordings"] == 21
    assert data["unique_heldout_recordings"] == 21
    assert data["training_bilingual_caption_examples"] == 42
    assert data["heldout_bilingual_caption_examples"] == 42
    assert data["first_party_procedural_sounds_only"] is True
    assert data["train_heldout_exact_audio_overlap"] is False
    assert data["independent_real_world_categories_proven"] is False
    assert data["train_fingerprint"] != data["validation_fingerprint"]
    state = report["model"]
    assert len(state["initial_sha256"]) == 64
    assert len(state["trained_sha256"]) == 64
    assert state["initial_sha256"] != state["trained_sha256"]
    assert state["unchanged_during_evaluation"] is True
    assert state["weights_published"] is False
    assert report["global_torch_rng_preserved"] is True
    assert report["no_model_weights_or_audio_saved"] is True
    assert report["model_publication_authorized"] is False
    assert report["semantic_generalization_or_perceptual_quality_claim"] is False
    serialized = json.dumps(report, allow_nan=False, ensure_ascii=False)
    assert len(serialized) < 35_000
    assert "base64" not in serialized


def test_real_training_loss_and_no_same_category_negatives(study):
    train = study["training"]
    assert train["actual_optimizer_steps"] == 1
    assert math.isfinite(train["loss_initial"])
    assert math.isfinite(train["loss_final"])
    assert train["loss_initial"] > 0
    assert train["loss_final"] > 0
    assert train["loss_min"] <= train["loss_initial"] <= train["loss_max"]
    assert train["same_category_false_negatives_used"] is False
    assert train["bilingual_audio_recordings_counted_once"] is True
    assert len(train["category_disjoint_minibatches"]) == 1
    entry = train["category_disjoint_minibatches"][0]
    assert len(entry["source_category_groups"]) == 4
    assert len(set(entry["source_category_groups"])) == 4
    assert len(entry["selected_unique_recordings"]) == 4
    assert entry["same_category_negative_pairs"] == 0
    assert math.isfinite(entry["positive_score"])
    assert math.isfinite(entry["negative_score"])


def test_heldout_bilingual_retrieval_does_not_count_two_audios_per_source(study):
    evals = study["heldout_retrieval"]
    assert evals["unique_source_categories"] == CATEGORIES == 21
    assert len(evals["heldout_category_audio_sha256"]) == 21
    assert len(set(evals["heldout_category_audio_sha256"])) == 21
    assert evals["same_audio_candidates_for_both_languages"] is True
    assert evals["category_labels_not_unique_recording_ids"] is True
    assert evals["language_templates_share_sound_categories"] is True
    assert evals["heldout_recordings_used_in_training"] is False
    assert evals["independent_real_world_prompt_semantics_proven"] is False
    assert math.isfinite(evals["english_hindi_same_recording_cosine_mean"])
    for name in ("english", "hindi", "wrong_label_cyclic_shift"):
        metric = evals[name]
        assert metric["category_candidates"] == 21
        assert 0 <= metric["rank1_matches"] <= 21
        assert 0 <= metric["rank5_matches"] <= 21
        assert 0 <= metric["recall_at_1"] <= 1
        assert 0 <= metric["recall_at_5"] <= 1
        assert metric["recall_at_5"] >= metric["recall_at_1"]
        assert 0 < metric["mean_reciprocal_rank"] <= 1
        assert math.isfinite(metric["mean_matched_minus_other_cosine"])
        assert len(metric["rank_per_category"]) == 21
        assert all(1 <= rank <= 21 for rank in metric["rank_per_category"])
        assert metric["open_vocabulary_or_audio_quality_established"] is False


def test_source_bilingual_pairing_is_exact_for_first_party_data():
    observed = synthetic_bilingual_waves((101,))
    assert observed.examples == 42
    assert torch.equal(observed.waves[::2], observed.waves[1::2])
    assert not torch.equal(observed.tokens[::2], observed.tokens[1::2])


def test_run_study_does_not_advance_global_torch_rng():
    before = torch.random.get_rng_state().clone()
    run_study(**tiny())
    assert torch.equal(before, torch.random.get_rng_state())
