"""Reference-aligned NLMS echo model: convergence, frozen states and chunk parity."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.nlms_echo import (
    NlmsSettings, NlmsGeometry, NlmsEchoStream,
    suppress_reference_echo,
)


def known_echo(frames=10000, rate=8000, seed=30, channels=1):
    rng = np.random.default_rng(seed)
    reference = rng.normal(0, .2, frames)
    impulse = np.zeros((channels, 12))
    for c in range(channels):
        impulse[c, 0] = .6 - .1 * c
        impulse[c, 4] = .25 + .05 * c
        impulse[c, 11] = -.12
    microphone = np.stack([
        np.convolve(reference, path, mode="full")[:frames]
        for path in impulse
    ])
    return (
        make_buffer(reference.astype(np.float32), rate),
        make_buffer(microphone.astype(np.float32), rate),
        impulse,
    )


@pytest.mark.parametrize("channels", [1, 2, 3])
@pytest.mark.parametrize("chunk_frames", [1, 37, 4096])
def test_synthetic_echo_path_is_recovered(channels, chunk_frames):
    reference, microphone, truth = known_echo(channels=channels, frames=4096)
    settings = NlmsSettings(taps=12, step_size=.7, epsilon=1e-6)
    result = suppress_reference_echo(
        reference, microphone, settings, chunk_frames=chunk_frames,
    )
    assert result.residual.spec == microphone.spec
    assert result.prediction.spec == microphone.spec
    assert result.weights.shape == truth.shape
    np.testing.assert_allclose(result.weights, truth, atol=.02, rtol=.04)
    assert result.summary.echo_reduction_db > 10
    assert result.summary.adapted_frames >= 4000
    assert result.summary.frames == 4096
    assert result.summary.microphone_channels == channels
    assert not result.weights.flags.writeable
    with pytest.raises(ValueError):
        result.weights.setflags(write=True)


def test_stream_boundaries_are_bitwise_invariant():
    reference, microphone, _ = known_echo(frames=1313, channels=2)
    settings = NlmsSettings(taps=16, step_size=.4)
    a = NlmsEchoStream(8000, 2, settings)
    whole_error, whole_echo = a.push(reference.samples, microphone.samples)
    whole_weights, whole_report = a.finish()
    b = NlmsEchoStream(8000, 2, settings)
    errors = []
    echoes = []
    for i in range(0, 1313, 7):
        end = min(i + 7, 1313)
        error, echo = b.push(
            reference.samples[:, i:end],
            microphone.samples[:, i:end],
        )
        errors.append(error)
        echoes.append(echo)
    chunk_weights, chunk_report = b.finish()
    np.testing.assert_array_equal(np.concatenate(errors, axis=1), whole_error)
    np.testing.assert_array_equal(np.concatenate(echoes, axis=1), whole_echo)
    np.testing.assert_array_equal(chunk_weights, whole_weights)
    assert chunk_report == whole_report


def test_silent_far_end_produces_exact_microphone_residual():
    rng = np.random.default_rng(91)
    reference = make_buffer(np.zeros(1000, dtype=np.float32), 8000)
    microphone = make_buffer(rng.normal(0, .1, 1000).astype(np.float32), 8000)
    result = suppress_reference_echo(
        reference, microphone, NlmsSettings(taps=32),
    )
    np.testing.assert_array_equal(result.residual.samples, microphone.samples)
    np.testing.assert_array_equal(result.prediction.samples, 0)
    np.testing.assert_array_equal(result.weights, 0)
    assert result.summary.adapted_frames == 0
    assert result.summary.frozen_frames == 1000


def test_explicit_freeze_mask_disables_learning():
    reference, microphone, _ = known_echo(frames=2000)
    mask = np.ones(2000, dtype=np.bool_)
    frozen = suppress_reference_echo(reference, microphone, freeze_mask=mask)
    np.testing.assert_array_equal(frozen.residual.samples, microphone.samples)
    np.testing.assert_array_equal(frozen.weights, 0)
    assert frozen.summary.frozen_frames == 2000


def test_partial_freeze_does_not_disable_every_sample():
    reference, microphone, _ = known_echo(frames=2500)
    mask = np.zeros(2500, dtype=np.bool_)
    mask[1000:1400] = True
    result = suppress_reference_echo(
        reference, microphone,
        NlmsSettings(taps=16, step_size=.5), freeze_mask=mask,
    )
    assert result.summary.frozen_frames == 400
    assert result.summary.adapted_frames == 2100
    assert result.summary.echo_reduction_db > 5


def test_disabled_adaptation_is_exact_bypass():
    reference, microphone, _ = known_echo(frames=777)
    result = suppress_reference_echo(
        reference, microphone, NlmsSettings(enabled=False),
    )
    np.testing.assert_array_equal(result.residual.samples, microphone.samples)
    assert result.summary.adapted_frames == 0


def test_no_untrusted_source_mutation():
    reference, microphone, _ = known_echo()
    ref_copy = reference.samples.copy()
    mic_copy = microphone.samples.copy()
    suppress_reference_echo(reference, microphone, NlmsSettings(taps=16))
    np.testing.assert_array_equal(reference.samples, ref_copy)
    np.testing.assert_array_equal(microphone.samples, mic_copy)


def test_no_automatic_double_talk_claim_in_report():
    reference, microphone, _ = known_echo(frames=500)
    result = suppress_reference_echo(reference, microphone,
                                     NlmsSettings(taps=12))
    report = result.summary.as_dict()
    assert report["protocol"] == "sonumio-explicit-reference-nlms-v1"
    assert report["model_inference"] is False
    assert report["double_talk_handling_automatic"] is False
    assert report["clock_alignment_automatic"] is False
    assert report["echo_cancellation_guaranteed"] is False


@pytest.mark.parametrize("bad", [
    dict(taps=0), dict(taps=4097), dict(taps=True),
    dict(step_size=0), dict(step_size=1.5),
    dict(epsilon=0), dict(epsilon=2), dict(epsilon=float("nan")),
    dict(leakage=-1), dict(leakage=.1),
    dict(maximum_weight_l2=0), dict(maximum_weight_l2=10001),
    dict(enabled=0),
])
def test_invalid_settings_rejected(bad):
    with pytest.raises(AudioContractError):
        NlmsSettings(**bad)


def test_unequal_pcm_clocks_and_channels_rejected():
    reference, mic, _ = known_echo(frames=1000)
    with pytest.raises(AudioContractError):
        suppress_reference_echo(make_buffer(reference.samples, 16000), mic)
    with pytest.raises(AudioContractError):
        suppress_reference_echo(reference, make_buffer(mic.samples[:, :999], 8000))
    with pytest.raises(AudioContractError):
        suppress_reference_echo(make_buffer(np.zeros((2, 1000)), 8000), mic)


def test_invalid_freeze_masks_rejected_before_processing():
    reference, mic, _ = known_echo(frames=1000)
    for mask in (
        np.zeros((1, 1000), dtype=bool),
        np.ones(1000, dtype=np.float64),
        np.ones(999, dtype=bool),
    ):
        with pytest.raises(AudioContractError):
            suppress_reference_echo(reference, mic, freeze_mask=mask)


def test_untrusted_stream_push_does_not_advance_clock():
    stream = NlmsEchoStream(8000, 1, NlmsSettings(taps=8))
    with pytest.raises(AudioContractError):
        stream.finish()
    for ref, mic in (
        (np.zeros((2, 20)), np.zeros(20)),
        (np.zeros(20), np.zeros(19)),
        (np.array([np.nan]), np.zeros(1)),
        (np.ones(20), np.ones((2, 20))),
    ):
        with pytest.raises(AudioContractError):
            stream.push(ref, mic)
        assert stream.input_frames == 0
    assert stream.push(np.ones(20), np.ones(20))[0].shape == (1, 20)
    stream.finish()
    assert stream.finished
    with pytest.raises(AudioContractError):
        stream.push(np.ones(1), np.ones(1))
    with pytest.raises(AudioContractError):
        stream.finish()


def test_adaptation_budget_preflights_extreme_geometry():
    with pytest.raises(AudioLimitError):
        NlmsGeometry(8, 2_000_000, 4096, 8000)


def test_float32_microphone_output_is_immutable():
    ref, mic, _ = known_echo(frames=500, channels=2)
    out = suppress_reference_echo(ref, mic, NlmsSettings(taps=12))
    assert out.residual.samples.dtype == np.float32
    assert not out.residual.samples.flags.writeable
    assert out.prediction.channels == 2
