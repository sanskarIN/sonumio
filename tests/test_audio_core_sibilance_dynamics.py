"""Sibilance-band dynamics tests: selectivity, causality, limits and bypass."""
import json

import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.sibilance_dynamics import (
    DeEsserSettings, isolate_sibilance, deesser_gain, deess,
)


RATE = 16000


def audio_with_burst(frames=12000, channels=1, high=True):
    n = np.arange(frames, dtype=np.float64)
    tone = .07 * np.sin(2 * np.pi * 500 * n / RATE)
    burst = np.zeros(frames, dtype=np.float64)
    if high:
        a, b = 3500, min(8000, frames)
        burst[a:b] = .35 * np.sin(2 * np.pi * 6000 * n[a:b] / RATE)
    return make_buffer(
        np.tile((tone + burst).astype(np.float32), (channels, 1)), RATE,
    )


def options(**changes):
    base = dict(
        n_fft=512, hop_frames=128,
        band_low_hz=4500., band_high_hz=7200.,
        threshold_dbfs=-30., ratio=8.,
        attack_seconds=0., release_seconds=.002,
    )
    base.update(changes)
    return DeEsserSettings(**base)


def amplitude(audio, hz, start, stop):
    n = np.arange(start, stop)
    sine = np.sin(2 * np.pi * hz * n / RATE)
    return abs(float(np.dot(
        audio.samples[0, start:stop].astype(np.float64), sine,
    ))) / len(sine)


@pytest.mark.parametrize("channels", [1, 2])
def test_high_frequency_burst_reduced_without_erasing_low_tone(channels):
    original = audio_with_burst(channels=channels)
    result = deess(original, options())
    high_before = amplitude(original, 6000, 4200, 7100)
    high_after = amplitude(result.audio, 6000, 4200, 7100)
    low_before = amplitude(original, 500, 4200, 7100)
    low_after = amplitude(result.audio, 500, 4200, 7100)
    assert high_before > .1
    assert high_after < high_before * .65
    assert low_after > low_before * .95
    assert result.report.affected_frames > 100
    assert result.report.minimum_gain < .75
    assert result.report.output_band_rms_estimate < result.report.input_band_rms
    assert result.audio.spec == original.spec


def test_low_band_unchanged_at_normal_numerical_precision():
    source = audio_with_burst(high=False)
    output = deess(source, options())
    np.testing.assert_allclose(output.audio.samples, source.samples, atol=1e-4)
    assert output.report.output_peak > 0


def test_isolated_band_respects_declared_frequency_bounds():
    source = audio_with_burst()
    with pytest.raises(AudioContractError):
        isolate_sibilance(source, options(band_high_hz=9000.))
    band = isolate_sibilance(source, options())
    assert band.spec == source.spec
    assert band.samples.dtype == np.float32
    assert np.isfinite(band.samples).all()
    assert not band.samples.flags.writeable
    assert np.max(np.abs(band.samples)) > .01


@pytest.mark.parametrize("wet", [0., .2, .5, .8, 1.])
def test_dry_wet_output_geometry_and_exact_bypass(wet):
    source = audio_with_burst(channels=2)
    result = deess(source, options(wet=wet))
    assert result.audio.spec == source.spec
    assert result.report.wet == wet
    assert result.report.as_dict()["model_inference"] is False
    assert result.report.as_dict()["speech_recognition_used"] is False
    assert result.report.as_dict()["automatic_clipping_prevention"] is False
    if wet == 0:
        assert result.audio is source
        np.testing.assert_array_equal(result.audio.samples, source.samples)


def test_gain_is_finite_bounded_and_deterministic():
    band = isolate_sibilance(audio_with_burst(channels=2), options())
    a = deesser_gain(band, options())
    b = deesser_gain(band, options())
    np.testing.assert_array_equal(a, b)
    assert a.shape == band.samples.shape
    assert np.isfinite(a).all()
    assert 0 < float(a.min()) < .7
    assert float(a.max()) <= 1


def test_linked_channels_apply_same_envelope():
    active = audio_with_burst().samples[0]
    quiet = audio_with_burst(high=False).samples[0]
    stereo = make_buffer(np.stack([active, quiet]), RATE)
    band = isolate_sibilance(stereo, options())
    linked = deesser_gain(band, options(link_channels=True))
    individual = deesser_gain(band, options(link_channels=False))
    np.testing.assert_array_equal(linked[0], linked[1])
    assert float(np.mean(individual[1, 4200:7100])) > (
        float(np.mean(linked[1, 4200:7100]))
    )


@pytest.mark.parametrize("ratio", [1., 2., 4., 8., 30.])
def test_ratio_changes_gain_monotonically(ratio):
    band = isolate_sibilance(audio_with_burst(), options())
    gain = deesser_gain(band, options(ratio=ratio))
    target = gain[0, 6000]
    assert target >= 0
    assert target <= 1
    if ratio == 1:
        assert target == pytest.approx(1.)


def test_zero_maximum_reduction_and_ratio_one_bypass():
    source = audio_with_burst()
    for config in (
        options(maximum_reduction_db=0),
        options(ratio=1),
    ):
        out = deess(source, config)
        np.testing.assert_array_equal(out.audio.samples, source.samples)
        assert out.report.minimum_gain == pytest.approx(1.)


def test_attack_release_detectors_are_causal_and_bounded():
    source = audio_with_burst()
    band = isolate_sibilance(source, options())
    fast = deesser_gain(
        band, options(attack_seconds=0, release_seconds=0),
    )
    slow = deesser_gain(
        band, options(attack_seconds=.03, release_seconds=.08),
    )
    assert fast[0, 3555] < 1
    assert slow[0, 3555] >= fast[0, 3555]
    assert np.isfinite(slow).all()
    assert np.min(slow) > 0


def test_stereo_custom_layout_preserved():
    data = audio_with_burst(channels=2).samples
    original = make_buffer(
        data, RATE,
        layout=AudioLayout(2, "discrete", ("speech-left", "speech-right")),
    )
    out = deess(original, options())
    assert out.audio.spec.layout == original.spec.layout


def test_source_immutability_and_repeated_replay():
    source = audio_with_burst(channels=2)
    original = source.samples.copy()
    a = deess(source, options())
    b = deess(source, options())
    np.testing.assert_array_equal(a.audio.samples, b.audio.samples)
    assert a.report == b.report
    np.testing.assert_array_equal(source.samples, original)
    assert not a.audio.samples.flags.writeable
    json.dumps(a.report.as_dict(), allow_nan=False)


def test_silence_avoids_false_reduction():
    source = make_buffer(np.zeros((2, 4096), dtype=np.float32), RATE)
    output = deess(source, options())
    np.testing.assert_array_equal(output.audio.samples, 0)
    assert output.report.minimum_gain == 1
    assert output.report.affected_frames == 0


@pytest.mark.parametrize("bad", [
    dict(n_fft=255), dict(n_fft=8193), dict(n_fft=True),
    dict(hop_frames=1), dict(hop_frames=512, n_fft=512),
    dict(band_low_hz=-1), dict(band_high_hz=1),
    dict(band_low_hz=5000, band_high_hz=4000),
    dict(threshold_dbfs=-110), dict(threshold_dbfs=1),
    dict(ratio=.5), dict(ratio=31),
    dict(attack_seconds=-1), dict(attack_seconds=3),
    dict(release_seconds=-1), dict(release_seconds=5),
    dict(maximum_reduction_db=-1), dict(maximum_reduction_db=50),
    dict(wet=-1), dict(wet=2), dict(wet=float("nan")),
    dict(link_channels=0),
])
def test_invalid_options_rejected_early(bad):
    with pytest.raises(AudioContractError):
        DeEsserSettings(**bad)


def test_bad_audio_or_settings_rejected():
    with pytest.raises(AudioContractError):
        deess(None)
    with pytest.raises(AudioContractError):
        deess(audio_with_burst(), None)
    with pytest.raises(AudioContractError):
        deesser_gain(None)


def test_detector_sample_work_limit_inside_audio_buffer_duration():
    # Valid 43.75-second AudioBuffer on 48 kHz, yet intentionally above
    # the more restrictive 4-million-control-sample DSP budget.
    huge = make_buffer(
        np.zeros((2, 2_100_000), dtype=np.float32), 48000,
    )
    with pytest.raises(AudioLimitError):
        deess(huge, options())


def test_input_below_spectral_nyquist_fails_before_fft():
    # Default 7800 Hz high band cannot be represented at an 8 kHz clock.
    audio = make_buffer(np.zeros(500, dtype=np.float32), 8000)
    with pytest.raises(AudioContractError):
        deess(audio)


def test_channel_linkage_flag_does_not_change_audio_clock():
    original = audio_with_burst(channels=2)
    linked = deess(original, options(link_channels=True)).audio
    unlinked = deess(original, options(link_channels=False)).audio
    assert linked.sample_rate == unlinked.sample_rate == RATE
    assert linked.frames == unlinked.frames == original.frames
    assert linked.channels == unlinked.channels == 2
