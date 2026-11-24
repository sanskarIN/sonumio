"""Frozen transform bridge and fail-closed latent study tests."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research import frozen_transform_study as study
from research.frozen_transform_latent import (
    METHODS, FrozenTransformCodec, latent_statistics, state_sha256,
)
from research.latent_audio import FRAMES, TextLatentDenoiser
from research.text_audio import ByteTextTokenizer
from research.transform_audio_codec import TransformFit
from research.velocity_ddim import sample_velocity_ddim
from research.wave_diffusion import DiffusionSchedule


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(31)
    waves = rng.normal(0, 0.1, (12, FRAMES)).astype(np.float32)
    labels = np.repeat(np.arange(6), 2)
    return TransformFit.fit(waves, labels), torch.from_numpy(waves[:, None].copy())


@pytest.mark.parametrize("method", METHODS)
def test_continuous_bridge_matches_numpy_and_preserves_frozen_state(fitted, method):
    transform, waves = fitted
    native = FrozenTransformCodec(transform, method, waves, calibrate=False)
    normalized = FrozenTransformCodec(transform, method, waves, calibrate=True)
    before = state_sha256(normalized)
    expected = transform.decode(transform.encode(waves[:, 0].numpy(), method, 16), method, 16)
    native_latent = native.encode(waves)
    assert native_latent.shape == (12, 16, 128)
    assert not native_latent.requires_grad
    np.testing.assert_allclose(native.decode(native_latent)[:, 0].numpy(), expected,
                               atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(normalized.decode(normalized.encode(waves)), native.decode(native_latent),
                               atol=2e-6, rtol=1e-5)
    assert normalized.describe()["trainable_parameters"] == 0
    assert normalized.describe()["quantized"] is False
    moments = latent_statistics(normalized, waves)
    assert moments["maximum_absolute_channel_mean"] < 1e-5
    assert moments["mean_absolute_channel_std_minus_one"] < 1e-5
    assert state_sha256(normalized) == before


def test_calibration_floor_and_heldout_shift(fitted):
    transform, waves = fitted
    silence = torch.zeros_like(waves)
    codec = FrozenTransformCodec(transform, "dct_low", silence, calibrate=True)
    assert codec.describe()["floored_channels"] == 16
    assert torch.equal(codec.decode(codec.encode(silence)), silence)
    shifted = silence + 0.1
    assert codec.encode(shifted).abs().max() > 1
    assert codec.describe()["training_channel_std"] == [0.0] * 16


@pytest.mark.parametrize("bad", [
    torch.zeros(1, 2, FRAMES), torch.zeros(1, 1, 8191),
    torch.zeros(1, 1, FRAMES, dtype=torch.float64),
    torch.full((1, 1, FRAMES), float("nan")),
])
def test_invalid_waveforms(fitted, bad):
    transform, waves = fitted
    codec = FrozenTransformCodec(transform, "pca", waves, calibrate=True)
    with pytest.raises(ValueError, match="transform batch"):
        codec.encode(bad)


def test_invalid_latents_and_methods(fitted):
    transform, waves = fitted
    with pytest.raises(ValueError, match="Unsupported"):
        FrozenTransformCodec(transform, "mdct_ranked", waves, calibrate=True)
    codec = FrozenTransformCodec(transform, "dct_ranked", waves, calibrate=False)
    with pytest.raises(ValueError, match="transform batch"):
        codec.decode(torch.zeros(1, 16, 129))


def test_actual_optimizer_and_ddim_smoke(fitted):
    """Small numerical smoke, not a bypass of the study readiness gate."""
    transform, waves = fitted
    tokens = ByteTextTokenizer().encode(["rain"] * len(waves))
    schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
    results = []
    with torch.random.fork_rng(devices=[]):
        for calibrate in (False, True):
            codec = FrozenTransformCodec(transform, "dct_ranked", waves, calibrate=calibrate)
            before = state_sha256(codec)
            torch.manual_seed(17)
            model = TextLatentDenoiser(16)
            training = study.train_denoiser(model, codec.encode(waves), tokens, schedule,
                                            seed=18, steps=2)
            results.append(training)
            wave = sample_velocity_ddim(model, schedule, "rain", evaluations=2, seed=301,
                                        guidance=1, latent=True, codec=codec)
            assert wave.shape == (FRAMES,)
            assert torch.isfinite(wave).all()
            assert state_sha256(codec) == before
            assert state_sha256(model) == training["trained_state_sha256"]
    assert results[0]["initial_state_sha256"] == results[1]["initial_state_sha256"]
    assert results[0]["input_trace_sha256"] == results[1]["input_trace_sha256"]


def test_real_preparation_and_study_gate(monkeypatch):
    preparation, codecs, train, validation, probe = study.prepare(17)
    assert preparation["selected_transform"] in METHODS
    assert not preparation["confirmation_used_for_selection"]
    assert not preparation["exact_cross_domain_overlap"]
    assert preparation["confirmation"]["recordings"] == 24
    assert sum(row["recordings"] for row in preparation["noise_confirmation"]) == 28
    assert len({preparation[key] for key in ("fit_fingerprint", "selection_fingerprint",
                                             "confirmation_fingerprint", "noise_confirmation_fingerprint")}) == 4
    gate = preparation["gate"]
    assert gate["passed"] == (
        gate["reference_probe_passed"] and gate["original_correct_accepted"] >= 16
        and gate["reconstruction_correct_accepted"] >= 16 and gate["all_categories_preserved"]
        and gate["original_noise_accepted"] == gate["reconstructed_noise_accepted"] == 0
    )
    monkeypatch.setattr(study, "prepare", lambda seed: (preparation, codecs, train, validation, probe))
    if not gate["passed"]:
        monkeypatch.setattr(study, "train_denoiser",
                            lambda *args, **kwargs: pytest.fail("Training ran after failed gate"))
    rng = torch.random.get_rng_state().clone()
    report = study.run_study(train_steps=1, evaluations=2)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not report["weights_or_audio_saved"]
    assert not report["model_readiness_established"]
    if gate["passed"]:
        assert report["status"] == "trained"
        assert set(report["arms"]) == {"native", "calibrated"}
        for arm in report["arms"].values():
            assert len(arm["generated"]["cases"]) == 12
            assert arm["heldout"]["unique_recordings"] == 42
    else:
        assert report["status"] == "reconstruction_gate_failed"
        assert report["arms"] == {}
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"train_steps": 0}, {"evaluations": 9}, {"sampling_seeds": (1, 1)},
])
def test_invalid_study_config(kwargs):
    with pytest.raises(ValueError, match="bounded frozen-transform"):
        study.run_study(**kwargs)
