import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.matrix_convolution import (
    ConvolutionGeometry, ImpulseMatrix, MatrixConvolver, convolve_matrix,
    matrix_reverb,
)


def reference(source, kernel):
    out = np.zeros((kernel.shape[0], source.shape[1] + kernel.shape[2] - 1))
    for receiver in range(kernel.shape[0]):
        for channel in range(source.shape[0]):
            out[receiver] += np.convolve(source[channel], kernel[receiver, channel])
    return out


@pytest.mark.parametrize("inputs,outputs", [(1, 1), (2, 2), (1, 2), (3, 1)])
@pytest.mark.parametrize("taps", [1, 3, 63, 64, 65, 149, 4097])
def test_matrix_matches_direct_linear_convolution(inputs, outputs, taps):
    rng = np.random.default_rng(414)
    source = rng.normal(0, .1, (inputs, 175)).astype(np.float32)
    kernel = rng.normal(0, .01, (outputs, inputs, taps))
    impulse = ImpulseMatrix(kernel, 16000, block_frames=64)
    audio = make_buffer(source, 16000)
    actual = convolve_matrix(audio, impulse)
    assert actual.frames == source.shape[1] + taps - 1
    assert actual.channels == outputs
    np.testing.assert_allclose(actual.samples, reference(source, kernel),
                               atol=1e-7, rtol=2e-6)


@pytest.mark.parametrize("size", [1, 63, 64, 65, 128, 129, 343])
@pytest.mark.parametrize("tail", [False, True])
def test_chunk_boundaries_do_not_change_fft_clock(size, tail):
    rng = np.random.default_rng(88)
    source = rng.uniform(-.5, .5, (2, size))
    kernel = rng.uniform(-.05, .05, (2, 2, 173))
    impulse = ImpulseMatrix(kernel, 8000, block_frames=64)
    a = MatrixConvolver(impulse)
    expected = np.concatenate([a.push(source), a.finish(include_tail=tail)], axis=1)
    b = MatrixConvolver(impulse)
    chunks = [b.push(source[:, i:i + 7]) for i in range(0, size, 7)]
    chunks.append(b.finish(include_tail=tail))
    actual = np.concatenate(chunks, axis=1)
    np.testing.assert_array_equal(actual, expected)
    assert b.input_frames == size
    assert b.emitted_frames == size + (172 if tail else 0)
    assert b.finished
    direct = reference(source, kernel)
    np.testing.assert_allclose(actual, direct[:, :actual.shape[1]], atol=1e-12)


def test_channel_routing_and_no_wrap():
    source = np.zeros((2, 70), dtype=np.float32)
    source[0, -1] = 1
    kernel = np.zeros((2, 2, 100))
    kernel[1, 0, 99] = .5
    result = convolve_matrix(make_buffer(source, 8000),
                             ImpulseMatrix(kernel, 8000, block_frames=64))
    expected = np.zeros((2, 169))
    expected[1, 168] = .5
    np.testing.assert_allclose(result.samples, expected, atol=1e-12)


def test_immutable_kernel_and_diagonal_layout():
    source = np.array([1., .25])
    matrix = ImpulseMatrix.diagonal(source, 8000, 2, block_frames=64)
    source[:] = 0
    assert matrix.coefficients[0, 0, 0] == 1
    assert matrix.coefficients[1, 0, 0] == 0
    with pytest.raises(ValueError):
        matrix.coefficients.setflags(write=True)


@pytest.mark.parametrize("mode,expected", [
    ("receiver_peak", .25), ("receiver_energy", .5), ("none", 1.),
])
def test_normalization_preserves_receiver_ratios(mode, expected):
    kernel = np.ones((2, 2, 2))
    kernel[1] = 0
    actual = ImpulseMatrix(kernel, 8000, normalization=mode).coefficients
    np.testing.assert_allclose(actual[0], expected)
    np.testing.assert_array_equal(actual[1], 0)


def test_wet_dry_and_explicit_layout():
    audio = make_buffer(np.array([[1., .5], [0., .2]], dtype=np.float32), 8000)
    impulse = ImpulseMatrix.diagonal(np.array([0., .5]), 8000, 2)
    dry = matrix_reverb(audio, impulse, wet=0)
    np.testing.assert_array_equal(dry.samples, audio.samples)
    wet = matrix_reverb(audio, impulse, wet=1, include_tail=True)
    np.testing.assert_allclose(wet.samples, [[0, .5, .25], [0, 0, .1]], atol=1e-12)
    layout = AudioLayout(2, "discrete", ("A", "B"))
    out = convolve_matrix(audio, impulse, output_layout=layout)
    assert out.spec.layout == layout


@pytest.mark.parametrize("value", [
    np.zeros((2, 2)), np.zeros((0, 1, 2)), np.zeros((1, 1, 0)),
    np.zeros((1, 1, 2), dtype=np.int32),
    np.array([[[np.nan]]]), np.array([[[np.inf]]]), np.array([[[1e7]]]),
])
def test_invalid_impulse(value):
    with pytest.raises(AudioContractError):
        ImpulseMatrix(value, 8000)


@pytest.mark.parametrize("block", [True, 32, 65, 32768])
def test_invalid_block(block):
    with pytest.raises(AudioContractError):
        ConvolutionGeometry(1, 1, 1, block)


def test_resource_bound_before_fft():
    with pytest.raises(AudioLimitError):
        ConvolutionGeometry(8, 8, 262144, 64)
    g = ConvolutionGeometry(1, 1, 32768, 64)
    with pytest.raises(AudioLimitError):
        g.validate_work(10_000_000)


def test_invalid_push_does_not_advance_clock():
    engine = MatrixConvolver(ImpulseMatrix(np.ones((1, 1, 5)), 8000,
                                           block_frames=64))
    with pytest.raises(AudioContractError):
        engine.finish()
    with pytest.raises(AudioContractError):
        engine.push(np.ones((2, 4)))
    with pytest.raises(AudioContractError):
        engine.push(np.array([np.nan]))
    with pytest.raises(AudioContractError):
        engine.push(np.array([1e7]))
    assert engine.input_frames == engine.emitted_frames == 0
    assert engine.push(np.ones(5)).shape == (1, 0)
    with pytest.raises(AudioContractError):
        engine.finish(include_tail=1)
    assert engine.finish().shape == (1, 9)
    with pytest.raises(AudioContractError):
        engine.finish()
    with pytest.raises(AudioContractError):
        engine.push(np.ones(1))


def test_offline_clock_and_layout_validation():
    impulse = ImpulseMatrix(np.ones((2, 1, 3)), 8000)
    with pytest.raises(AudioContractError):
        convolve_matrix(make_buffer(np.ones(3), 16000), impulse)
    audio = make_buffer(np.ones(3), 8000)
    with pytest.raises(AudioContractError):
        convolve_matrix(audio, impulse, output_layout=AudioLayout.for_channels(1))
    with pytest.raises(AudioContractError):
        matrix_reverb(audio, impulse)
    with pytest.raises(AudioContractError):
        convolve_matrix(audio, impulse, include_tail=1)
