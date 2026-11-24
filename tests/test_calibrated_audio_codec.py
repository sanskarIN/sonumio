import json

import pytest

torch = pytest.importorskip("torch")

from research import calibrated_audio_codec as codec
from research import codec_calibration_study as study
from research.latent_audio import FRAMES, LATENT_CHANNELS, LATENT_FRAMES, TinyAudioAutoencoder


def test_residual_codec_shape_gradient_and_bounded_decode():
    model = codec.ResidualAudioCodec()
    wave = torch.randn((2, 1, FRAMES)) * 0.1
    latent = model.encode(wave)
    assert latent.shape == (2, LATENT_CHANNELS, LATENT_FRAMES)
    pre, decoded = model.decode_with_preactivation(latent)
    assert decoded.shape == wave.shape == pre.shape
    assert torch.equal(decoded, pre.tanh())
    assert torch.max(decoded.abs()) <= 1
    (decoded - wave).abs().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_train_only_calibration_and_round_trip():
    rng = torch.Generator().manual_seed(18)
    value = torch.randn((4, LATENT_CHANNELS, LATENT_FRAMES), generator=rng)
    value = value * torch.arange(1, 17).view(1, 16, 1) + 7
    calibration = codec.LatentCalibration.fit(value)
    standardized = calibration.normalize(value)
    torch.testing.assert_close(standardized.mean((0, 2)), torch.zeros(16), atol=1e-6, rtol=0)
    torch.testing.assert_close(standardized.std((0, 2), correction=0), torch.ones(16), atol=1e-6, rtol=0)
    torch.testing.assert_close(calibration.restore(standardized), value)
    # Distribution shift must remain visible; no refitting on validation.
    shifted = calibration.normalize(value + calibration.scale * 2)
    torch.testing.assert_close(shifted.mean((0, 2)), torch.full((16,), 2.0), atol=2e-6, rtol=0)
    assert calibration.describe()["training_recordings"] == 4


def test_constant_channel_floor_and_round_trip():
    value = torch.full((3, 16, 128), 0.5)
    calibration = codec.LatentCalibration.fit(value)
    assert calibration.describe()["floored_channels"] == 16
    assert torch.equal(calibration.normalize(value), torch.zeros_like(value))
    assert torch.equal(calibration.restore(torch.zeros_like(value)), value)
    assert torch.isfinite(calibration.normalize(value + 0.01)).all()


@pytest.mark.parametrize("value", [
    torch.zeros(2, 15, 128), torch.zeros(2, 16, 127),
    torch.full((2, 16, 128), float("nan")), torch.zeros(1, 16, 128),
    torch.zeros(2, 16, 128, dtype=torch.float64),
    torch.zeros(2, 16, 128, requires_grad=True),
])
def test_calibration_rejects_invalid_training_inputs(value):
    with pytest.raises(ValueError):
        codec.LatentCalibration.fit(value)


@pytest.mark.parametrize("codec_cls", [TinyAudioAutoencoder, codec.ResidualAudioCodec])
def test_adapter_reconstruction_is_same_as_native(codec_cls):
    native = codec_cls().eval()
    wave = torch.randn(3, 1, FRAMES) * 0.1
    with torch.no_grad():
        latent = native.encode(wave)
        calibration = codec.LatentCalibration.fit(latent)
        expected = native.decode(latent)
        adapter = codec.CalibratedCodec(native, calibration)
        normalized = adapter.encode(wave)
        restored = adapter.native_latent(normalized)
        torch.testing.assert_close(restored, latent, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(adapter.decode(normalized), expected, atol=1e-6, rtol=1e-5)
        assert all(not p.requires_grad for p in adapter.parameters())
        assert set(adapter.state_dict()) >= {"offset", "scale"}


def test_native_adapter_is_identity_in_latent_space():
    native = TinyAudioAutoencoder().eval()
    wave = torch.randn(2, 1, FRAMES)
    with torch.no_grad():
        expected = native.encode(wave)
        adapter = codec.CalibratedCodec(native, None)
        assert torch.equal(adapter.encode(wave), expected)
        assert torch.equal(adapter.native_latent(expected), expected)


def test_reconstruction_diagnostics_identity_silence_and_inversion():
    wave = torch.linspace(-0.5, 0.5, FRAMES).view(1, 1, FRAMES)
    same = codec.reconstruction_diagnostics(wave, wave)[0]
    assert same["mae"] == 0
    assert same["relative_squared_error"] == 0
    assert same["waveform_cosine"] == pytest.approx(1)
    assert same["multiresolution_log_magnitude_mae"] == 0
    inverted = codec.reconstruction_diagnostics(wave, -wave)[0]
    assert inverted["relative_squared_error"] == pytest.approx(4)
    assert inverted["waveform_cosine"] == pytest.approx(-1)
    silent = codec.reconstruction_diagnostics(torch.zeros_like(wave), wave)[0]
    assert silent["relative_squared_error"] is None
    assert silent["waveform_cosine"] is None
    assert silent["rms_ratio"] is None
    json.dumps(silent, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"codec_steps": 0}, {"train_steps": 129},
    {"evaluations": 9}, {"sampling_seeds": (301, 301)},
])
def test_invalid_study_config(kwargs):
    with pytest.raises(ValueError, match="bounded codec study"):
        study.run_study(**kwargs)


def test_paired_study_real_training_and_generation():
    rng = torch.random.get_rng_state().clone()
    result = study.run_study(codec_steps=1, train_steps=1, evaluations=2)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert result["unique_training_recordings"] == result["unique_validation_recordings"] == 42
    assert result["same_denoiser_initialization_and_noise"]
    assert result["configuration"]["scalar_compression_ratio"] == 4
    assert not result["exact_split_overlap"]
    assert not result["weights_or_audio_saved"]
    assert not result["model_readiness_established"]
    arms = result["arms"]
    assert arms["legacy_native"]["native_codec_sha256"] == arms["legacy_calibrated"]["native_codec_sha256"]
    assert arms["legacy_native"]["calibration"] is None
    for name in study.ARMS:
        arm = arms[name]
        assert arm["reconstruction"]["validation_unique_recordings"] == 42
        assert arm["reconstruction"]["validation_by_source_count"]["1"]["recordings"] == 12
        assert arm["reconstruction"]["validation_by_source_count"]["2"]["recordings"] == 30
        assert len(arm["reconstruction"]["source_holdout_oracle"]["cases"]) == 18
        assert arm["heldout_velocity"]["caption_cases_per_timestep"] == 84
        if result["reference_probe"]["gate"]["passed"]:
            assert len(arm["generated"]["cases"]) == 12
            assert len(arm["generated"]["null_checks"]) == 1
        else:
            assert arm["generated"] is None
        if name != "legacy_native":
            assert arm["calibration"]["training_recordings"] == 42
            assert arm["training_latent_statistics"]["maximum_absolute_channel_mean"] < 1e-3
    assert len({arm["denoising"]["initial_state_sha256"] for arm in arms.values()}) == 1
    assert len({arm["denoising"]["training_input_trace_sha256"] for arm in arms.values()}) == 1
    json.dumps(result, allow_nan=False)
