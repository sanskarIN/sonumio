"""Real original two-codec matched training and held-out spectral metrics."""
from __future__ import annotations

import hashlib
import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.codec_loss_study import (
    PROTOCOL, ARMS, _settings, _source_unique, run_study,
)
from research.codec_objectives import COMPONENT_NAMES
from research.latent_audio import (
    FRAMES, synthetic_bilingual_waves, TinyAudioAutoencoder,
)
from research.text_audio import SYNTHETIC_COMBINATIONS


def tiny() -> dict:
    return {
        "model_seed": 17,
        "updates": 1,
        "training_batch": 2,
        "learning_rate": 0.001,
        "training_seeds": (0,),
        "heldout_seeds": (101,),
    }


@pytest.fixture(scope="module")
def report():
    return run_study(**tiny())


@pytest.mark.parametrize("bad", [
    {"model_seed": -1}, {"model_seed": True},
    {"model_seed": 2**32}, {"updates": 0},
    {"updates": 65}, {"updates": True},
    {"training_batch": 0}, {"training_batch": 9},
    {"training_batch": True},
    {"learning_rate": 0}, {"learning_rate": -0.5},
    {"learning_rate": 20}, {"learning_rate": True},
    {"learning_rate": float("nan")},
    {"learning_rate": float("inf")},
    {"training_seeds": ()},
    {"heldout_seeds": ()},
    {"training_seeds": (0, 0)},
    {"heldout_seeds": (101, 101)},
    {"training_seeds": (0,), "heldout_seeds": (0,)},
    {"heldout_seeds": (True,)},
    {"training_seeds": [0]},
    {"training_seeds": "0"},
    {"heldout_seeds": (2**32,)},
])
def test_invalid_codec_pilot_settings_are_rejected_before_training(bad):
    with pytest.raises(ValueError):
        _settings(**(tiny() | bad))


def test_actual_codec_training_uses_matched_initial_weights_and_source_splits(report):
    assert report["protocol"] == PROTOCOL
    assert len(report["configuration_sha256"]) == 64
    assert report["configuration"]["model_seed"] == 17
    assert report["configuration"]["updates"] == 1
    assert report["configuration"]["training_seeds"] == [0]
    assert report["configuration"]["heldout_seeds"] == [101]
    data = report["data"]
    assert data["training_recordings"] == 21
    assert data["heldout_recordings"] == 21
    assert data["english_hindi_captions_per_split"] == 42
    assert data["exact_recording_overlap"] is False
    assert data["heldout_audio_used_in_training"] is False
    assert len(data["source_recording_sha256_heldout"]) == 21
    assert len(set(data["source_recording_sha256_heldout"])) == 21
    assert data["train_fingerprint"] != data["heldout_fingerprint"]
    models = report["models"]
    assert len(models["identical_initial_sha256"]) == 64
    assert models["trained_states_distinct"] is True
    assert models["unchanged_during_evaluation"] is True
    assert set(models["trained_model_sha256"]) == set(ARMS)
    assert all(
        models["trained_model_sha256"][arm] != models["identical_initial_sha256"]
        for arm in ARMS
    )
    assert models["trained_model_sha256"]["original_loss"] != models[
        "trained_model_sha256"]["multiresolution_loss"]
    controls = report["paired_controls"]
    assert controls["same_initial_parameters"] is True
    assert controls["same_training_minibatches"] is True
    assert controls["same_optimizer_type_learning_rate_and_gradient_clipping"] is True
    assert controls["only_reconstruction_training_loss_differs"] is True
    assert controls["heldout_objective_components_identical_for_comparison"] is True
    assert controls["english_hindi_caption_duplicates_counted_once"] is True


def test_all_heldout_waveforms_scored_without_language_duplication(report):
    measurements = report["heldout_reconstruction_diagnostics"]
    assert measurements["independent_acoustic_recordings_scored"] == 21
    assert measurements["bilingual_caption_duplicates_skipped"] == 21
    assert measurements["source_combination_families"] == 21
    assert measurements["matched_heldout_waveform_for_both_codecs"] is True
    assert measurements["no_perceptual_quality_or_statistical_significance_claim"] is True
    assert len(measurements["per_recording"]) == 21
    assert len(set(row["heldout_sha256"] for row in measurements["per_recording"])) == 21
    for name in ARMS:
        components = measurements["components_mean_per_independent_recording"][name]
        assert set(components) == set(COMPONENT_NAMES)
        assert all(math.isfinite(v) and v >= 0 for v in components.values())
        for key in ("sample_aligned_mae", "sample_aligned_rmse", "snr_db"):
            assert math.isfinite(measurements["aggregate_waveform_metrics"][name][key])
    for record in measurements["per_recording"]:
        assert len(record["heldout_sha256"]) == 64
        assert record["acoustic_group"]
        assert record["recording_seed"] == 101
        assert record["independent_acoustic_group_within_this_seed"] is True
        assert all(math.isfinite(v) for v in record[
            "multiresolution_minus_original_components"
        ].values())
    family = measurements["source_family_deltas_multires_minus_original"]
    assert set(family) == set(COMPONENT_NAMES)
    for metric in COMPONENT_NAMES:
        values = family[metric]
        assert values["source_combination_families"] == 21
        assert values["caption_duplicates_counted_once"] is True
        assert math.isfinite(values["mean_over_acoustic_families"])
        assert 0 <= values["positive_family_deltas"] <= 21
        assert 0 <= values["negative_family_deltas"] <= 21


def test_paired_training_results_do_not_claim_quality_or_publish_assets(report):
    losses = report["training_losses_not_directly_comparable"]
    for arm in ARMS:
        assert losses[arm]["count"] == 1
        assert math.isfinite(losses[arm]["first"])
        assert math.isfinite(losses[arm]["last"])
        assert losses[arm]["first"] > 0 and losses[arm]["last"] > 0
    assert report["global_torch_rng_isolated"] is True
    assert report["no_model_weights_or_audio_saved"] is True
    assert report["model_publication_authorized"] is False
    assert report["perceptual_fidelity_or_music_speech_quality_established"] is False
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False)
    assert len(serialized) < 60_000
    assert "base64" not in serialized
    assert '"samples"' not in serialized


def test_one_audio_source_is_counted_once_despite_two_languages():
    source = synthetic_bilingual_waves((101,))
    unique, fingerprints = _source_unique(source, (101,))
    assert unique.shape == (21, 1, FRAMES)
    assert len(fingerprints) == 21
    assert len(set(fingerprints)) == 21
    assert source.examples == 42
    assert torch.equal(source.waves[::2], source.waves[1::2])
    assert not torch.equal(source.tokens[::2], source.tokens[1::2])


def test_forged_bilingual_source_pair_is_rejected():
    from research.latent_audio import BilingualWaves
    source = synthetic_bilingual_waves((101,))
    damaged = source.waves.clone()
    damaged[1, 0, 100] += 0.1
    forged = BilingualWaves(
        damaged, source.tokens, source.fingerprint, source.examples,
    )
    with pytest.raises(ValueError, match="bilingual"):
        _source_unique(forged, (101,))


def test_invalid_explicit_objective_config_types_rejected():
    from research.codec_loss_study import run_study
    with pytest.raises(ValueError):
        run_study(**(tiny() | {"weights": {"waveform": 1}}))
    with pytest.raises(ValueError):
        run_study(**(tiny() | {"spectral": {"fft_sizes": (256,)}}))


def test_train_loss_isolation_preserves_global_random_generator():
    before = torch.random.get_rng_state().clone()
    run_study(**tiny())
    assert torch.equal(before, torch.random.get_rng_state())
