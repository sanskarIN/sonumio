"""Fixed seed modulation and reverb transformations with explicit dry/wet."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.effects import (
    DelayConfig, ModulationConfig, chorus, convolution_reverb,
    delay, effect_report, flanger, ring_modulate, tremolo,
)
from sonumio.audio_core.errors import AudioContractError
from sonumio.audio_core.numerics import stable_peak


@pytest.fixture
def impulse():
    signal = np.zeros((2, 2400), dtype=np.float32)
    signal[:, 0] = 1
    return make_buffer(signal, 24000)


@pytest.fixture
def tone():
    time = np.arange(4800, dtype=np.float32) / 24000
    sample = .5 * np.sin(2 * np.pi * 440 * time)
    return make_buffer(np.stack((sample, sample)), 24000)


def test_single_tap_delay_has_exact_integer_echo_position(impulse):
    delayed = delay(
        impulse, DelayConfig(delay_seconds=.01, feedback=.5, wet=1, taps=1)
    )
    assert delayed.samples.shape == impulse.samples.shape
    assert delayed.samples[0, 240] == pytest.approx(1)
    assert np.all(delayed.samples[:, :240] == 0)
    assert delayed.samples[0, 241] == 0


def test_delay_dry_wet_and_pingpong_routing():
    signal = np.zeros((2, 2400), dtype=np.float32)
    signal[0, 0] = 1
    buffer = make_buffer(signal, 24000)
    dry = delay(buffer, DelayConfig(wet=0))
    np.testing.assert_array_equal(dry.samples, buffer.samples)
    ping = delay(buffer, DelayConfig(delay_seconds=.01,
                                    wet=1, ping_pong=True, taps=1))
    assert ping.samples[1, 240] == pytest.approx(1)
    assert ping.samples[0, 240] == pytest.approx(0)


@pytest.mark.parametrize("waveform", ["sine", "triangle", "square", "saw"])
def test_tremolo_preserves_finite_frames_and_is_repeatable(tone, waveform):
    settings = ModulationConfig(rate_hz=4, depth=.6, wet=.7,
                                waveform=waveform)
    first = tremolo(tone, settings)
    second = tremolo(tone, settings)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.frames == tone.frames
    assert first.channels == tone.channels
    assert np.isfinite(first.samples).all()
    assert first.source_note == "tremolo"


def test_ring_modulation_changes_tone_but_does_not_modify_input(tone):
    original = tone.samples.copy()
    settings = ModulationConfig(rate_hz=10, depth=1, wet=1)
    output = ring_modulate(tone, settings)
    assert not np.array_equal(output.samples, tone.samples)
    np.testing.assert_array_equal(tone.samples, original)
    assert np.isfinite(output.samples).all()


def test_chorus_and_flanger_have_deterministic_bounded_outputs(tone):
    a = chorus(tone, rate_hz=.6, voices=3)
    b = chorus(tone, rate_hz=.6, voices=3)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.samples.shape == tone.samples.shape
    x = flanger(tone, feedback=.25, wet=.5)
    y = flanger(tone, feedback=.25, wet=.5)
    np.testing.assert_array_equal(x.samples, y.samples)
    assert np.isfinite(x.samples).all()
    assert x.frames == tone.frames


def test_fir_convolution_reverb_with_tail_preserves_sample_clock(impulse):
    ir = np.array([.5, .25, .125], dtype=np.float32)
    wet = convolution_reverb(impulse, ir, wet=1,
                             normalization="none", truncate_tail=False)
    assert wet.frames == impulse.frames + 2
    assert wet.sample_rate == impulse.sample_rate
    np.testing.assert_allclose(wet.samples[0, :3], ir, atol=1e-6)
    no_tail = convolution_reverb(impulse, ir, wet=1,
                                normalization="none", truncate_tail=True)
    assert no_tail.frames == impulse.frames
    np.testing.assert_allclose(no_tail.samples[0, :3], ir, atol=1e-6)


def test_dry_convolution_and_error_report_are_explicit(impulse):
    kernel = np.array([1, 0, 0], dtype=np.float32)
    output = convolution_reverb(impulse, kernel, wet=0)
    np.testing.assert_array_equal(output.samples, impulse.samples)
    report = effect_report(impulse, output, "identity")
    assert report["new_tail_frames"] == 0
    assert report["input_peak"] == report["output_peak"]
    assert report["original_generator_unchanged"] is True
    assert report["model_inference"] is False


def test_invalid_delay_and_modulation_configuration_fail_closed():
    with pytest.raises(AudioContractError):
        DelayConfig(feedback=1)
    with pytest.raises(AudioContractError):
        DelayConfig(taps=0)
    with pytest.raises(AudioContractError):
        ModulationConfig(rate_hz=-1)
    with pytest.raises(AudioContractError):
        ModulationConfig(waveform="script")
    with pytest.raises(AudioContractError):
        ModulationConfig(depth=float("nan"))


def test_unapproved_reverb_impulse_shape_and_nonfinite_rejected(tone):
    for bad in (
        np.ones((2, 4), dtype=np.float32),
        np.array([1, np.nan], dtype=np.float32),
        np.ones(4096, dtype=np.float32),
        np.array([1, 2], dtype=np.int32),
    ):
        with pytest.raises(AudioContractError):
            convolution_reverb(tone, bad)
    with pytest.raises(AudioContractError):
        chorus(tone, voices=0)
    with pytest.raises(AudioContractError):
        flanger(tone, feedback=1)
