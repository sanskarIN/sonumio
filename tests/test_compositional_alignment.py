"""Compositional InfoNCE, source-category lattice and bilingual probe tests."""
from __future__ import annotations

import copy
import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.compositional_alignment import (
    PROTOCOL, LOSS_PROTOCOL, RELATIONSHIP_PROTOCOL,
    SOURCE_GROUPS, METRICS, relation,
    category_relation_matrix, overlap_masked_info_nce,
    compositional_retrieval, evaluate_bilingual_composition,
)
from research.audio_text_alignment import OriginalAudioTextAligner
from research.latent_audio import FRAMES
from research.text_audio import ByteTextTokenizer
from research.weight_ema import model_state_sha256


def _original_audio(freq=220, *, batch=3):
    t = torch.arange(FRAMES, dtype=torch.float32) / 8000
    x = 0.13 * torch.sin(2 * math.pi * freq * t)
    return x.reshape(1, 1, FRAMES).repeat(batch, 1, 1)


@pytest.fixture(scope="module")
def evaluation():
    torch.manual_seed(101)
    model = OriginalAudioTextAligner(channels=16, embedding_dim=32)
    rng_before = torch.random.get_rng_state().clone()
    fingerprint_before = model_state_sha256(model)
    report = evaluate_bilingual_composition(
        model, validation_seeds=(101,),
    )
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert model_state_sha256(model) == fingerprint_before
    return report


def test_original_21_category_relation_lattice_is_exact_symmetric_and_complete():
    matrix, report = category_relation_matrix(SOURCE_GROUPS)
    assert matrix.shape == (21, 21)
    assert torch.equal(matrix, matrix.T)
    assert torch.equal(torch.diag(matrix), torch.full((21,), 2, dtype=torch.int8))
    assert report["protocol"] == RELATIONSHIP_PROTOCOL
    assert report["number_of_groups"] == 21
    assert report["ordered_exact_relations"] == 21
    assert report["ordered_partially_shared_relations"] == 180
    assert report["ordered_disjoint_relations"] == 240
    assert (matrix == 2).sum() == 21
    assert (matrix == 1).sum() == 180
    assert (matrix == 0).sum() == 240
    assert report["each_group_exactly_matches_only_itself"] is True
    assert report["partially_shared_categories_not_identical_positive_recordings"] is True
    assert report["structural_dsp_group_metadata_only_not_human_semantics"] is True


@pytest.mark.parametrize(("left", "right", "expected"), [
    ((0,), (0,), "exact"),
    ((0, 1), (0, 1), "exact"),
    ((0,), (0, 2), "partially_shared"),
    ((0, 1), (1, 2), "partially_shared"),
    ((0,), (1,), "disjoint"),
    ((0, 1), (2, 3), "disjoint"),
])
def test_exact_related_and_disjoint_sound_relationships(left, right, expected):
    assert relation(left, right) == expected
    assert relation(right, left) == expected


def test_masked_infonce_does_not_push_partial_source_categories_as_negatives():
    groups = ((0,), (0, 1), (2,))
    vector = torch.eye(3)
    masked, report = overlap_masked_info_nce(
        vector, vector, groups, temperature=0.2,
    )
    assert masked.ndim == 0
    assert torch.isfinite(masked)
    assert masked >= 0
    assert report["protocol"] == LOSS_PROTOCOL
    assert report["groups"] == 3
    assert report["exact_positive_edges"] == 3
    assert report["ignored_partially_shared_negative_edges"] == 2
    assert report["true_disjoint_negative_edges"] == 4
    assert report["mean_disjoint_negative_count_per_anchor"] == pytest.approx(4 / 3)
    assert report["partial_overlap_groups_used_as_negatives"] is False
    assert report["same_group_english_hindi_duplicates_used_as_negatives"] is False
    assert report["no_external_pretrained_semantic_model"] is True
    assert report["audio_quality_or_open_vocabulary_understanding_proven"] is False
    assert report["masked_symmetric_info_nce"] <= report[
        "ordinary_symmetric_info_nce_with_partials"]


def test_masked_information_loss_has_real_grads_through_both_original_towers():
    torch.manual_seed(12)
    network = OriginalAudioTextAligner(
        channels=16, embedding_dim=32,
    )
    waves = torch.cat((
        _original_audio(220, batch=1),
        _original_audio(313, batch=1),
        _original_audio(440, batch=1),
    ), dim=0)
    tokens = ByteTextTokenizer().encode((
        "rain", "rain and wind", "forest birds",
    ))
    features_a, features_t = network(waves, tokens)
    criterion, report = overlap_masked_info_nce(
        features_a, features_t,
        ((0,), (0, 1), (3,)),
        temperature=0.15,
    )
    assert criterion.item() > 0
    criterion.backward()
    for parameters in (
        network.audio_tower.parameters(),
        network.text_tower.parameters(),
        network.text_projection.parameters(),
    ):
        assert any(
            parameter.grad is not None
            and torch.isfinite(parameter.grad).all()
            and parameter.grad.abs().sum() > 0
            for parameter in parameters
        )
    assert report["ignored_partially_shared_negative_edges"] == 2
    assert report["true_disjoint_negative_edges"] == 4


def test_masked_info_nce_rejects_unusable_related_only_panel():
    embeddings = torch.eye(2)
    with pytest.raises(ValueError, match="nonoverlapping negative"):
        overlap_masked_info_nce(
            embeddings, embeddings, ((0,), (0, 1)),
        )


def test_mathematical_perfect_embeddings_report_exact_top1_for_21_categories():
    perfect = torch.eye(21)
    report = compositional_retrieval(
        perfect, perfect, SOURCE_GROUPS,
    )
    assert report["protocol"] == PROTOCOL
    assert report["candidate_categories"] == 21
    assert report["exact_rank_one_count"] == 21
    assert report["exact_rank_one_fraction"] == 1.0
    assert report["groups_with_partial_relations"] == 21
    assert report["every_category_has_disjoint_negative"] is True
    assert report["no_external_pretrained_sound_embeddings"] is True
    assert report["semantic_accuracy_or_perceptual_quality_proven"] is False
    for record in report["per_acoustic_category"]:
        assert record["exact_rank_in_complete_category_panel"] == 1
        assert record["exact_similarity"] == 1.0
        assert record["partially_shared_similarity"] == 0.0
        assert record["disjoint_similarity"] == 0.0
        assert record["exact_minus_disjoint"] == 1.0
        assert record["partial_minus_disjoint"] == 0.0
        assert record["exact_minus_partial"] == 1.0
        assert record["exact_candidate_count"] == 1
        assert record["partially_shared_candidate_count"] >= 1
        assert record["disjoint_candidate_count"] >= 1
    assert report["relationship_descriptive"]["exact_minus_disjoint"]["mean"] == 1


def test_wrong_category_caption_order_reduces_exact_top1_without_discarding_related():
    vectors = torch.eye(21, dtype=torch.float32)
    wrong = compositional_retrieval(
        vectors, vectors.roll(shifts=1, dims=0), SOURCE_GROUPS,
    )
    assert wrong["exact_rank_one_count"] < 21
    assert wrong["exact_rank_one_fraction"] < 1
    assert set(wrong["relationship_descriptive"]) == set(METRICS)
    assert len(wrong["per_acoustic_category"]) == 21


def test_minimal_disjoint_panel_does_not_invent_partial_relationships():
    groups = ((0,), (1,))
    score = compositional_retrieval(
        torch.eye(2), torch.eye(2), groups,
    )
    for row in score["per_acoustic_category"]:
        assert row["partially_shared_similarity"] is None
        assert row["partial_minus_disjoint"] is None
        assert row["exact_minus_partial"] is None
        assert row["disjoint_candidate_count"] == 1
    assert score["relationship_descriptive"][
        "partially_shared_similarity"]["groups_available"] == 0


def test_compositional_reference_uses_same_recording_for_two_languages(evaluation):
    assert evaluation["protocol"] == PROTOCOL
    assert len(evaluation["configuration_sha256"]) == 64
    config = evaluation["configuration"]
    assert config["validation_seeds"] == [101]
    assert config["source_categories"] == 21
    assert len(config["recording_sha256"]) == 21
    assert len(set(config["recording_sha256"])) == 21
    assert config["same_audio_recordings_for_both_languages"] is True
    assert evaluation["independent_source_recordings_in_candidate_panel"] == 21
    assert evaluation["two_languages_not_two_independent_recordings"] is True
    assert evaluation["model_state_unchanged"] is True
    assert evaluation["global_torch_rng_unchanged"] is True
    assert evaluation["existing_model_weights_not_saved_or_released"] is True
    assert evaluation["perceptual_or_open_vocabulary_prompt_alignment_claim"] is False
    assert math.isfinite(evaluation["same_category_english_hindi_caption_cosine_mean"])
    for lang in ("english", "hindi", "wrong_caption_category_control"):
        results = evaluation[lang]
        assert results["candidate_categories"] == 21
        assert len(results["per_acoustic_category"]) == 21
        assert 0 <= results["exact_rank_one_fraction"] <= 1
        assert results["no_external_pretrained_sound_embeddings"] is True
        assert results["semantic_accuracy_or_perceptual_quality_proven"] is False
        for group in results["per_acoustic_category"]:
            assert 1 <= group["exact_rank_in_complete_category_panel"] <= 21
            assert group["partially_shared_candidate_count"] >= 1
            assert group["disjoint_candidate_count"] >= 1
            assert all(math.isfinite(v) for k, v in group.items()
                       if k in METRICS and v is not None)
    output = json.dumps(evaluation, allow_nan=False, ensure_ascii=False)
    assert len(output) < 72_000
    assert "base64" not in output


@pytest.mark.parametrize("bad_group", [
    (),
    ((0,),),
    ((0,), (0,)),
    ((0, 0), (1,)),
    ((1, 0), (2,)),
    ((0, 1, 2), (3,)),
    ((True,), (1,)),
    ((0,), ("rain",)),
    ((6,), (1,)),
    [(0,), (1,)],
])
def test_bad_source_group_topology_rejected(bad_group):
    with pytest.raises(ValueError):
        category_relation_matrix(bad_group)


@pytest.mark.parametrize("temperature", [
    -1, 0, 0.019, 1.001, True, float("nan"), "0.1",
])
def test_masked_loss_rejects_invalid_temperature(temperature):
    source = torch.eye(3)
    with pytest.raises(ValueError):
        overlap_masked_info_nce(
            source, source, ((0,), (0, 1), (2,)),
            temperature=temperature,
        )


@pytest.mark.parametrize("bad_vectors", [
    torch.zeros(2, 3, dtype=torch.float64),
    torch.full((2, 3), float("nan")),
    torch.full((2, 3), float("inf")),
    torch.zeros(1, 3),
    torch.zeros(22, 3),
    torch.zeros(2, 2, 2),
])
def test_nonfinite_wrong_dtype_or_mismatched_embedding_panel_rejected(bad_vectors):
    with pytest.raises(ValueError):
        compositional_retrieval(
            bad_vectors, torch.zeros(2, 3),
            ((0,), (1,)),
        )
    with pytest.raises(ValueError):
        overlap_masked_info_nce(
            bad_vectors, torch.zeros(2, 3),
            ((0,), (1,)),
        )


def test_compositional_model_probe_is_deterministic_under_same_model_and_audio():
    torch.manual_seed(31)
    network = OriginalAudioTextAligner(
        channels=16, embedding_dim=32,
    )
    prior = model_state_sha256(network)
    rng = torch.random.get_rng_state().clone()
    a = evaluate_bilingual_composition(network, validation_seeds=(101,))
    b = evaluate_bilingual_composition(network, validation_seeds=(101,))
    assert a["configuration_sha256"] == b["configuration_sha256"]
    assert a["english"] == b["english"]
    assert a["hindi"] == b["hindi"]
    assert a["wrong_caption_category_control"] == b["wrong_caption_category_control"]
    assert model_state_sha256(network) == prior
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_invalid_bilingual_model_and_source_seeds_rejected_before_evaluation():
    model = OriginalAudioTextAligner(channels=16, embedding_dim=32)
    for seeds in (
        (), (101, 101), (True,), (-1,), (2**32,),
        [101], ("101",), (101, 102, 103, 104, 105),
    ):
        with pytest.raises(ValueError):
            evaluate_bilingual_composition(model, validation_seeds=seeds)
    with pytest.raises(ValueError):
        evaluate_bilingual_composition(None, validation_seeds=(101,))


def test_overlap_masked_loss_exact_duplicates_refused():
    encoded = torch.eye(3)
    groups = ((0,), (0,), (2,))
    with pytest.raises(ValueError):
        overlap_masked_info_nce(encoded, encoded, groups)
