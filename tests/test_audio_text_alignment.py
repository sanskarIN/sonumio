"""Original bilingual audio/text contrastive learning numerical and safety tests."""
from __future__ import annotations

import math

import pytest
torch = pytest.importorskip("torch")

from research.audio_text_alignment import (
    PROTOCOL, CONTRASTIVE_PROTOCOL,
    SourceAudioEncoder, OriginalAudioTextAligner,
    category_disjoint_info_nce, retrieval_metrics,
)
from research.text_audio import ByteTextTokenizer
from research.latent_audio import FRAMES


def _sound(frequency=220, batch=3):
    time = torch.arange(FRAMES).float() / 8000
    signal = 0.15 * torch.sin(2 * math.pi * frequency * time)
    return signal.reshape(1, 1, FRAMES).repeat(batch, 1, 1)


@pytest.mark.parametrize(("channels", "embedding_dim"), [
    (16, 16), (16, 32), (24, 64), (32, 32),
])
def test_original_audio_tower_shapes_norms_and_trainable_grads(channels, embedding_dim):
    torch.manual_seed(17)
    model = SourceAudioEncoder(
        channels=channels, embedding_dim=embedding_dim,
    )
    audio = _sound(batch=3)
    output = model(audio)
    assert output.shape == (3, embedding_dim)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.norm(dim=1), torch.ones(3), atol=1e-4)
    loss = output.square().mean() + output[:, 0].mean()
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in model.parameters()
    )


def test_original_audio_text_embedding_paired_training_gradient():
    torch.manual_seed(18)
    model = OriginalAudioTextAligner(channels=16, embedding_dim=32)
    audio = torch.cat((_sound(220, batch=1), _sound(440, batch=1)))
    tokens = ByteTextTokenizer().encode(["rain", "बारिश"])
    feature_audio, feature_text = model(audio, tokens)
    assert feature_audio.shape == feature_text.shape == (2, 32)
    assert torch.isfinite(feature_audio).all()
    assert torch.isfinite(feature_text).all()
    loss, diagnostic = category_disjoint_info_nce(
        feature_audio, feature_text, ("rain", "wind"),
    )
    assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    assert diagnostic["protocol"] == CONTRASTIVE_PROTOCOL
    assert diagnostic["unique_categories"] == 2
    assert diagnostic["duplicates_or_same_category_false_negatives_allowed"] is False
    assert diagnostic["external_model_or_artifacts_used"] is False
    loss.backward()
    for params in (
        model.audio_tower.parameters(),
        model.text_tower.parameters(),
        model.text_projection.parameters(),
    ):
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0 for p in params
        )


def test_info_nce_paired_alignment_beats_explicitly_shuffled_labels():
    pair = torch.eye(4, dtype=torch.float32)
    keys = ("rain", "wind", "forest", "ocean")
    positive, report = category_disjoint_info_nce(
        pair, pair, keys, temperature=0.1,
    )
    negative, _ = category_disjoint_info_nce(
        pair, pair.roll(1, dims=0), keys, temperature=0.1,
    )
    assert positive >= 0
    assert positive.item() < 0.01
    assert negative > positive + 1
    assert report["positive_similarity_mean"] > report["off_diagonal_similarity_mean"]
    assert report["positive_cosine_mean"] == pytest.approx(1)


def test_retrieval_math_returns_exact_ranks_for_perfect_category_pairs():
    embeddings = torch.eye(21, dtype=torch.float32)
    keys = tuple(f"source-{i}" for i in range(21))
    perfect = retrieval_metrics(embeddings, embeddings, keys)
    assert perfect["category_candidates"] == 21
    assert perfect["rank1_matches"] == 21
    assert perfect["rank5_matches"] == 21
    assert perfect["recall_at_1"] == 1
    assert perfect["recall_at_5"] == 1
    assert perfect["mean_reciprocal_rank"] == 1
    assert perfect["mean_matched_minus_other_cosine"] == pytest.approx(1)
    assert perfect["rank_per_category"] == [1] * 21
    assert perfect["open_vocabulary_or_audio_quality_established"] is False
    broken = retrieval_metrics(
        embeddings, embeddings.roll(1, 0), keys,
    )
    assert broken["rank1_matches"] == 0
    assert broken["recall_at_1"] == 0
    assert broken["mean_reciprocal_rank"] < 1


@pytest.mark.parametrize("bad", [
    (16, 8), (16, 17), (16, 128), (16, True),
    (8, 32), (17, 32), (48, 32), (True, 32),
    ("16", 32), (16.0, 32),
])
def test_bad_audio_text_architecture_refused(bad):
    with pytest.raises(ValueError):
        SourceAudioEncoder(channels=bad[0], embedding_dim=bad[1])
    with pytest.raises(ValueError):
        OriginalAudioTextAligner(channels=bad[0], embedding_dim=bad[1])


@pytest.mark.parametrize("invalid_audio", [
    torch.zeros(2, 1, FRAMES - 1),
    torch.zeros(2, 1, FRAMES + 1),
    torch.zeros(2, 1, FRAMES, dtype=torch.float64),
    torch.full((2, 1, FRAMES), float("nan")),
    torch.full((2, 1, FRAMES), float("inf")),
    torch.ones(2, 1, FRAMES) * 2,
    torch.zeros(17, 1, FRAMES),
    torch.zeros(1, 2, FRAMES),
])
def test_bad_source_audio_rejected_before_encoding(invalid_audio):
    model = OriginalAudioTextAligner(channels=16)
    tokens = ByteTextTokenizer().encode(["rain", "wind"])
    with pytest.raises(ValueError):
        model(invalid_audio, tokens)


@pytest.mark.parametrize("invalid_tokens", [
    torch.zeros(2, 129, dtype=torch.long),
    torch.zeros(2, 130, dtype=torch.float32),
    torch.ones(2, 130, dtype=torch.long) * 999,
    torch.ones(2, 130, dtype=torch.long) * -1,
    torch.zeros(1, 130, dtype=torch.long),
])
def test_invalid_text_tokens_refused(invalid_tokens):
    model = OriginalAudioTextAligner(channels=16)
    with pytest.raises(ValueError):
        model(_sound(batch=2), invalid_tokens)


@pytest.mark.parametrize("bad_temperature", [
    0, 0.019, 1.01, -1, True, "0.1", float("nan"), float("inf"),
])
def test_info_nce_temperature_must_be_finite_and_bounded(bad_temperature):
    embeddings = torch.eye(3)
    with pytest.raises(ValueError):
        category_disjoint_info_nce(
            embeddings, embeddings, ("a", "b", "c"),
            temperature=bad_temperature,
        )


@pytest.mark.parametrize("keys", [
    ("rain", "rain", "wind"),
    ("rain", "wind"),
    ("rain", 42, "forest"),
    ("rain", "", "wind"),
    ["rain", "forest", "wind"],
])
def test_duplicate_category_false_negatives_and_malformed_keys_refused(keys):
    embeddings = torch.eye(3)
    with pytest.raises(ValueError):
        category_disjoint_info_nce(embeddings, embeddings, keys)


def test_nonfinite_or_mismatched_contrastive_embeddings_rejected():
    one = torch.eye(3)
    broken = one.clone()
    broken[0, 0] = float("nan")
    with pytest.raises(ValueError):
        category_disjoint_info_nce(
            one, broken, ("rain", "wind", "forest")
        )
    with pytest.raises(ValueError):
        category_disjoint_info_nce(
            one, torch.eye(2), ("rain", "wind", "forest")
        )
    with pytest.raises(ValueError):
        retrieval_metrics(one, broken, ("rain", "wind", "forest"))


def test_candidate_count_must_be_at_least_two():
    source = torch.eye(1)
    with pytest.raises(ValueError):
        category_disjoint_info_nce(source, source, ("rain",))
    with pytest.raises(ValueError):
        retrieval_metrics(source, source, ("rain",))


def test_random_bilingual_audios_without_pretrained_embeddings_still_finite():
    torch.manual_seed(23)
    model = OriginalAudioTextAligner(channels=16, embedding_dim=32)
    audio = torch.cat((_sound(220, batch=1), _sound(310, batch=1)))
    tokens = ByteTextTokenizer().encode(["wind", "हवा"])
    with torch.no_grad():
        a, b = model(audio, tokens)
        metrics = retrieval_metrics(a, b, ("wind", "forest"))
    assert 0 <= metrics["recall_at_1"] <= 1
    assert 0 <= metrics["recall_at_5"] <= 1
    assert all(1 <= x <= 2 for x in metrics["rank_per_category"])
    assert metrics["all_candidates_from_synthetic_dsp_only"] is True
    assert metrics["english_hindi_same_recording_not_independent"] is True
