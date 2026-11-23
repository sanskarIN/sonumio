"""Differentiable CPU spectral, transient, and codec-signal controls."""
from __future__ import annotations

import json
import math

import pytest
torch = pytest.importorskip("torch")

from research.codec_objectives import (
    PROTOCOL, FFT_SIZES, COMPONENT_NAMES, WEIGHT_NAMES,
    CodecObjectiveWeights, CodecSpectralConfig,
    _spectrogram, _rms_envelope, _highband_ratio,
    codec_components, weighted_codec_loss, describe_codec_error,
)
from research.latent_audio import FRAMES, TinyAudioAutoencoder


def _sine(hz: float, *, phase=0.0, amplitude=0.3, batch=2):
    axis = torch.arange(FRAMES).float() / 8000
    signal = amplitude * torch.sin(
        2 * math.pi * hz * axis + phase
    )
    return signal.view(1, 1, FRAMES).expand(batch, 1, -1).clone()


def test_identity_reconstruction_gives_zero_components_and_neutral_snr():
    original = _sine(440)
    metrics = codec_components(original, original)
    assert set(metrics) == set(COMPONENT_NAMES)
    assert all(value.shape == () for value in metrics.values())
    assert all(torch.isfinite(x) and x >= 0 for x in metrics.values())
    assert all(value.item() < 1e-7 for value in metrics.values())
    total, named = weighted_codec_loss(
        original, original, return_components=True,
    )
    assert total.item() < 1e-6
    assert named.keys() == metrics.keys()
    d = describe_codec_error(original, original)
    assert d["protocol"] == PROTOCOL
    assert d["clips"] == 2 and d["frames"] == FRAMES
    assert d["sample_rate"] == 8000
    assert d["sample_aligned_mae"] == 0
    assert d["sample_aligned_rmse"] == 0
    assert d["audio_or_model_weights_written"] is False
    assert d["external_perceptual_model_used"] is False
    assert d["listenability_or_semantics_measured"] is False
    assert d["zero_error_uses_finite_epsilon"] is True
    assert math.isfinite(d["snr_db"])
    assert len(json.dumps(d, allow_nan=False)) < 3000


def test_different_audio_has_positive_finite_reconstruction_errors():
    original = _sine(440)
    altered = _sine(440, phase=0.5)
    outputs = codec_components(original, altered)
    assert outputs["waveform_l1"] > 0
    assert outputs["preemphasis_l1"] > 0
    assert outputs["envelope_rms_l1"] >= 0
    assert outputs["spectral_convergence"] > 0
    assert torch.isfinite(weighted_codec_loss(original, altered))
    assert weighted_codec_loss(original, altered).item() > 0


def test_untouched_silence_does_not_break_spectral_convergence():
    silence = torch.zeros((1, 1, FRAMES), dtype=torch.float32)
    metrics = codec_components(silence, silence)
    assert all(float(v) == 0 for v in metrics.values())
    result = describe_codec_error(silence, silence)
    assert result["sample_aligned_mae"] == 0
    assert math.isfinite(result["snr_db"])
    assert result["components"]["spectral_convergence"] == 0


def test_silence_reference_and_small_nonzero_prediction_finite():
    silence = torch.zeros((1, 1, FRAMES), dtype=torch.float32)
    candidate = _sine(400, amplitude=0.01, batch=1)
    loss = weighted_codec_loss(silence, candidate)
    assert loss.item() > 0
    assert math.isfinite(loss.item())
    metrics = describe_codec_error(silence, candidate)
    assert math.isfinite(metrics["snr_db"])
    assert all(math.isfinite(v) for v in metrics["components"].values())


def test_loss_gradients_reach_actual_original_neural_codec():
    torch.manual_seed(17)
    codec = TinyAudioAutoencoder()
    original = _sine(330, batch=2)
    reconstructed = codec(original)
    assert reconstructed.shape == original.shape
    loss, components = weighted_codec_loss(
        original, reconstructed, return_components=True,
    )
    assert math.isfinite(loss.item()) and loss.item() > 0
    assert all(math.isfinite(float(x)) for x in components.values())
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in codec.encoder.parameters()
    )
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        and p.grad.abs().sum() > 0
        for p in codec.decoder.parameters()
    )


def test_neural_codec_objective_reproducible_with_matching_waveforms():
    torch.manual_seed(21)
    codec = TinyAudioAutoencoder().eval()
    audio = _sine(291, batch=1)
    with torch.no_grad():
        decoded = codec(audio)
        first, components = weighted_codec_loss(
            audio, decoded, return_components=True,
        )
        second = weighted_codec_loss(audio, decoded)
    assert torch.equal(first, second)
    assert first == sum(
        components[k] * getattr(CodecObjectiveWeights(), weight)
        for k, weight in zip(COMPONENT_NAMES, WEIGHT_NAMES)
    )


def test_fft_and_rms_envelopes_have_expected_dimensions():
    source = _sine(411)
    for n_fft in FFT_SIZES:
        mag = _spectrogram(source[:, 0], n_fft)
        assert mag.shape[0] == 2
        assert mag.shape[1] == n_fft // 2 + 1
        assert mag.shape[2] > 1
        assert torch.isfinite(mag).all()
    for width in (64, 256, 1024):
        envelope = _rms_envelope(source[:, 0], width)
        assert envelope.shape == (2, FRAMES // width)
        assert torch.isfinite(envelope).all()
        assert torch.all(envelope >= 0)


def test_high_frequency_ratio_separates_low_and_high_tones():
    low = _sine(100, batch=1)
    high = _sine(2500, batch=1)
    low_mag = _spectrogram(low[:, 0], 1024)
    high_mag = _spectrogram(high[:, 0], 1024)
    low_ratio = float(_highband_ratio(low_mag, 1024, 1500))
    high_ratio = float(_highband_ratio(high_mag, 1024, 1500))
    assert 0 <= low_ratio < 0.1
    assert 0.8 < high_ratio <= 1
    assert high_ratio > low_ratio
    assert codec_components(low, high)["highband_energy_ratio_l1"] > 0.1


def test_transient_difference_is_visible_in_flux_and_waveform():
    reference = torch.zeros((1, 1, FRAMES))
    changed = torch.zeros_like(reference)
    reference[0, 0, 1024] = 0.8
    changed[0, 0, 1060] = 0.8
    metrics = codec_components(reference, changed)
    assert metrics["waveform_l1"] > 0
    assert metrics["preemphasis_l1"] > 0
    assert metrics["spectral_flux_l1"] > 0
    assert metrics["linear_magnitude_l1"] >= 0


def test_custom_fft_plan_and_explicit_weight_changes_are_respected():
    reference = _sine(500, batch=1)
    predicted = _sine(500, phase=0.3, batch=1)
    minimal = CodecSpectralConfig(
        fft_sizes=(256,), highband_hz=1200,
        high_frequency_boost=0,
        preemphasis_coefficient=0,
    )
    dense = CodecSpectralConfig(
        fft_sizes=(256, 512, 1024, 2048),
        highband_hz=1200,
        high_frequency_boost=2,
        preemphasis_coefficient=1,
    )
    loss_a = weighted_codec_loss(reference, predicted, config=minimal)
    loss_b = weighted_codec_loss(reference, predicted, config=dense)
    assert torch.isfinite(loss_a) and torch.isfinite(loss_b)
    assert loss_a != loss_b
    custom = CodecObjectiveWeights(
        waveform=1, preemphasis=0, envelope=0,
        log_magnitude=0, linear_magnitude=0,
        spectral_convergence=0, spectral_flux=0,
        highband_energy_ratio=0,
    )
    only_wave = weighted_codec_loss(
        reference, predicted, weights=custom,
    )
    assert only_wave == torch.nn.functional.l1_loss(reference, predicted)


@pytest.mark.parametrize("bad", [
    {"waveform": True}, {"waveform": float("nan")},
    {"preemphasis": -1}, {"envelope": float("inf")},
    {"log_magnitude": 20}, {"linear_magnitude": "0.5"},
    {"spectral_convergence": None},
    {"highband_energy_ratio": 100},
])
def test_bad_component_weight_rejected(bad):
    with pytest.raises(ValueError):
        CodecObjectiveWeights(**bad)


def test_all_zero_objective_weights_refused():
    with pytest.raises(ValueError, match="nonzero"):
        CodecObjectiveWeights(**{name: 0.0 for name in WEIGHT_NAMES})


@pytest.mark.parametrize("bad", [
    {"fft_sizes": ()}, {"fft_sizes": (512, 256)},
    {"fft_sizes": (256, 256)}, {"fft_sizes": (17,)},
    {"fft_sizes": [256]}, {"fft_sizes": (True,)},
    {"highband_hz": 0}, {"highband_hz": 4000},
    {"highband_hz": True},
    {"high_frequency_boost": float("nan")},
    {"high_frequency_boost": -1},
    {"high_frequency_boost": 3},
    {"preemphasis_coefficient": float("nan")},
    {"preemphasis_coefficient": -1},
    {"preemphasis_coefficient": 2},
    {"preemphasis_coefficient": True},
])
def test_bad_spectral_configuration_rejected(bad):
    with pytest.raises(ValueError):
        CodecSpectralConfig(**bad)


@pytest.mark.parametrize("bad", [
    torch.zeros(1, 1, FRAMES - 1),
    torch.zeros(1, 1, FRAMES + 1),
    torch.zeros(1, 1, FRAMES, dtype=torch.float64),
    torch.full((1, 1, FRAMES), float("nan")),
    torch.full((1, 1, FRAMES), float("inf")),
    torch.ones(1, 1, FRAMES) * 1.01,
    torch.zeros(0, 1, FRAMES),
    torch.zeros(9, 1, FRAMES),
    torch.zeros(1, 2, FRAMES),
])
def test_invalid_waveforms_rejected_before_stft(bad):
    original = _sine(440, batch=1)
    with pytest.raises(ValueError):
        codec_components(original, bad)


def test_wrong_weights_or_config_types_rejected():
    source = _sine(440, batch=1)
    with pytest.raises(ValueError):
        weighted_codec_loss(source, source, weights={"waveform": 1})
    with pytest.raises(ValueError):
        codec_components(source, source, config={"fft_sizes": [256]})
    with pytest.raises(ValueError):
        weighted_codec_loss(source, source, return_components=1)
