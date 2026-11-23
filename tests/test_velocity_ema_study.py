"""End-to-end control tests for original synthetic v-model EMA experiments."""
from __future__ import annotations

import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.velocity_ema_study import (
    PROTOCOL, _settings, _validation_mse, run_study,
)


def _tiny() -> dict:
    return dict(
        seed=17, steps=1, codec_steps=1, diffusion_steps=8,
        channels=16, decay=0.8, warmup=False, sampling_seeds=(301,),
    )


@pytest.mark.parametrize("bad", [
    {"seed": -1}, {"seed": True}, {"seed": 2**32},
    {"steps": 0}, {"steps": 65}, {"steps": True},
    {"codec_steps": 0}, {"codec_steps": 17},
    {"diffusion_steps": 7}, {"diffusion_steps": 17},
    {"channels": 32}, {"channels": True},
    {"decay": True}, {"decay": float("nan")}, {"decay": 1},
    {"decay": -0.1}, {"warmup": 1},
    {"sampling_seeds": ()}, {"sampling_seeds": (1, 1)},
    {"sampling_seeds": (True,)}, {"sampling_seeds": (2**32,)},
    {"sampling_seeds": (1, 2, 3)},
])
def test_invalid_settings_rejected_before_training(bad):
    with pytest.raises(ValueError):
        _settings(**(_tiny() | bad))


def test_real_online_ema_training_sampling_report_and_no_weight_publication():
    prior = torch.random.get_rng_state().clone()
    report = run_study(**_tiny())
    assert torch.equal(prior, torch.random.get_rng_state())
    assert report["protocol"] == PROTOCOL
    data = report["training"]
    assert data["independent_training_recordings"] == 42
    assert data["independent_heldout_recordings"] == 42
    assert data["bilingual_caption_examples_per_split"] == 84
    assert data["train_heldout_exact_overlap"] is False
    assert data["train_fingerprint"] != data["validation_fingerprint"]
    assert report["online_and_ema_same_training_trajectory"] is True
    assert report["heldout_audio_used_for_training"] is False
    assert report["no_model_weights_or_audio_saved"] is True
    assert report["model_publication_authorized"] is False
    assert report["listening_or_semantic_audio_quality_evaluated"] is False
    assert len(report["configuration_sha256"]) == 64
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False)
    assert len(encoded) < 50_000
    for arm in ("raw", "latent"):
        mse = report["heldout_velocity_objective"][arm]
        assert mse["bilingual_caption_examples"] == 84
        assert mse["distinct_audio_recordings"] == 42
        assert mse["shared_noise_and_timesteps"] is True
        assert mse["online"] > 0 and mse["ema"] > 0
        assert mse["ema_minus_online"] == pytest.approx(
            mse["ema"] - mse["online"])
        shadow = report["model_state_integrity"][arm]
        assert shadow["ema_updates"] == 1
        assert shadow["same_parameters_through_evaluation"]
        assert len(shadow["online_model_sha256"]) == 64
        assert len(shadow["ema_model_sha256"]) == 64
        rows = report["in_memory_generated_numerical_comparisons"][arm]
        assert len(rows) == 4
        for row in rows:
            assert row["seed"] == 301
            assert row["category"] in ("rain", "wind")
            assert row["language"] in ("en", "hi")
            assert row["prompt_adherence_or_human_quality_measured"] is False
            for variant in ("online", "ema"):
                data = row[variant]
                assert 0 <= data["postclip"]["clipping_fraction"] <= 1
                assert math.isfinite(data["postclip"]["rms"])
                assert len(data["postclip"]["sha256"]) == 64
                assert data["prelimiter_signal_source"] == (
                    "latent_codec_decoder_before_tanh" if arm == "latent"
                    else "raw_ddpm_before_final_clamp"
                )
                assert math.isfinite(data["preclip_peak"])
            difference = row["online_ema_waveform_difference"]
            assert 0 <= difference["mean_absolute_difference"] <= 2
        assert report["by_arm"][arm]["generated_pairs_compared"] == 4


def test_nonfinite_heldout_prediction_is_rejected():
    from torch import nn
    from research.latent_audio import synthetic_bilingual_waves
    from research.wave_diffusion import DiffusionSchedule

    class InvalidModel(nn.Module):
        def forward(self, x, timestep, token_ids):
            return torch.full_like(x, float("nan"))

    validation = synthetic_bilingual_waves((101,))
    with pytest.raises(ValueError, match="Nonfinite"):
        _validation_mse(validation, InvalidModel(), InvalidModel(),
                        DiffusionSchedule(8), seed=17)


def test_online_and_ema_share_same_noise_not_only_same_random_seed():
    from torch import nn
    from research.latent_audio import synthetic_bilingual_waves
    from research.wave_diffusion import DiffusionSchedule

    class ConstantPredictor(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.register_buffer("value", torch.tensor(value))

        def forward(self, x, t, tokens):
            return torch.zeros_like(x) + self.value

    source = synthetic_bilingual_waves((101,))
    online = ConstantPredictor(0)
    averaged = ConstantPredictor(0)
    result = _validation_mse(
        source, online, averaged, DiffusionSchedule(8), seed=900,
    )
    assert result["online"] == result["ema"]
    assert result["ema_minus_online"] == 0


def test_decay_zero_control_matches_online_model_and_generated_waveforms_exactly():
    """A null hypothesis control: EMA with decay=0 must equal online weights."""
    report = run_study(**(_tiny() | {"decay": 0.0}))
    for arm in ("raw", "latent"):
        states = report["model_state_integrity"][arm]
        assert states["online_model_sha256"] == states["ema_model_sha256"]
        metrics = report["heldout_velocity_objective"][arm]
        assert metrics["online"] == metrics["ema"]
        assert metrics["ema_minus_online"] == 0.0
        for row in report["in_memory_generated_numerical_comparisons"][arm]:
            assert row["online"]["postclip"]["sha256"] == row["ema"]["postclip"]["sha256"]
            delta = row["online_ema_waveform_difference"]
            assert delta["exact_same_waveform"] is True
            assert delta["mean_absolute_difference"] == 0.0
