"""Numerical and lifecycle regression tests for partitioned convolution."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.matrix_convolution import ImpulseMatrix, convolve_matrix
from sonumio.audio_core.partitioned_convolution import (
    PartitionGeometry, PartitionedConvolver, convolve_partitioned,
)


def direct(source, kernel):
    output = np.zeros((kernel.shape[0], source.shape[1] + kernel.shape[2] - 1))
    for receiver in range(kernel.shape[0]):
        for channel in range(kernel.shape[1]):
            output[receiver] += np.convolve(source[channel], kernel[receiver, channel])
    return output


def stream(engine, source, sizes, *, tail=True):
    chunks = []
    offset = 0
    for size in sizes:
        if offset == source.shape[1]:
            break
        end = min(source.shape[1], offset + size)
        chunks.append(engine.push(source[:, offset:end]))
        offset = end
    if offset < source.shape[1]:
        chunks.append(engine.push(source[:, offset:]))
    chunks.append(engine.finish(include_tail=tail))
    return np.concatenate(chunks, axis=1)


@pytest.mark.parametrize("inputs,outputs", [(1, 1), (1, 2), (2, 1), (3, 2)])
@pytest.mark.parametrize("taps", [1, 2, 63, 64, 65, 128, 129, 511, 1025])
@pytest.mark.parametrize("tail", [False, True])
def test_partitioned_matches_linear_convolution(inputs, outputs, taps, tail):
    rng = np.random.default_rng(942)
    source = rng.normal(0, .15, (inputs, 173)).astype(np.float32)
    kernel = rng.normal(0, .025, (outputs, inputs, taps))
    impulse = ImpulseMatrix(kernel, 16000, block_frames=64)
    audio = make_buffer(source, 16000)
    result = convolve_partitioned(audio, impulse, include_tail=tail)
    expected = direct(source, kernel)
    frames = 173 + (taps - 1 if tail else 0)
    assert result.frames == frames
    assert result.channels == outputs
    np.testing.assert_allclose(result.samples, expected[:, :frames], atol=1e-7, rtol=2e-6)
    ordinary = convolve_matrix(audio, impulse, include_tail=tail)
    np.testing.assert_allclose(result.samples, ordinary.samples, atol=1e-7, rtol=2e-6)


@pytest.mark.parametrize("frames", [1, 63, 64, 65, 127, 128, 129, 997])
@pytest.mark.parametrize("taps", [1, 64, 65, 389])
@pytest.mark.parametrize("tail", [False, True])
def test_stream_exact_invariance(frames, taps, tail):
    rng = np.random.default_rng(345)
    source = rng.normal(0, .1, (2, frames))
    kernel = rng.normal(0, .1, (2, 2, taps))
    impulse = ImpulseMatrix(kernel, 8000, block_frames=64)
    a = PartitionedConvolver(impulse)
    expected = stream(a, source, [frames], tail=tail)
    b = PartitionedConvolver(impulse)
    actual = stream(b, source, [1, 8, 55, 2, 93, 17, 64] * 12, tail=tail)
    np.testing.assert_array_equal(actual, expected)
    reference = direct(source, kernel)
    n = frames + (taps - 1 if tail else 0)
    np.testing.assert_allclose(actual, reference[:, :n], atol=1e-11)
    assert a.input_frames == b.input_frames == frames
    assert a.emitted_frames == b.emitted_frames == n
    assert a.finished and b.finished


@pytest.mark.parametrize("block", [64, 128, 256, 1024])
def test_delayed_route_survives_ring_wrap(block):
    source = np.zeros((2, 7 * block + 3))
    source[1, 5 * block + 2] = .75
    kernel = np.zeros((3, 2, 3 * block + 5))
    kernel[2, 1, -1] = -.5
    impulse = ImpulseMatrix(kernel, 24000, block_frames=block)
    engine = PartitionedConvolver(impulse)
    actual = stream(engine, source, [block - 1] * 10)
    expected = np.zeros((3, source.shape[1] + kernel.shape[-1] - 1))
    expected[2, 8 * block + 6] = -.375
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_ring_reset_on_distinct_streams():
    impulse = ImpulseMatrix(np.ones((1, 1, 130)), 8000, block_frames=64)
    nonzero = PartitionedConvolver(impulse)
    stream(nonzero, np.ones((1, 160)), [160])
    fresh = PartitionedConvolver(impulse)
    out = stream(fresh, np.zeros((1, 160)), [160])
    np.testing.assert_array_equal(out, 0)


def test_full_block_and_single_tap_finish_is_empty():
    impulse = ImpulseMatrix(np.array([[[.5]]]), 8000, block_frames=64)
    engine = PartitionedConvolver(impulse)
    np.testing.assert_allclose(engine.push(np.ones(64)), .5, atol=1e-12)
    assert engine.finish().shape == (1, 0)
    assert engine.emitted_frames == 64


def test_push_rejection_is_nonmutating():
    impulse = ImpulseMatrix(np.ones((1, 1, 9)), 8000, block_frames=64)
    engine = PartitionedConvolver(impulse)
    with pytest.raises(AudioContractError):
        engine.finish()
    engine.push(np.ones(7))
    for bad in (np.zeros((2, 7)), np.array([np.nan]), np.array([np.inf]),
                np.array([1e8]), np.array([], dtype=float), np.array([1])):
        with pytest.raises(AudioContractError):
            engine.push(bad)
        assert engine.input_frames == 7
        assert engine.emitted_frames == 0
    with pytest.raises(AudioContractError):
        engine.finish(include_tail=1)
    expected = direct(np.ones((1, 7)), impulse.coefficients)
    np.testing.assert_allclose(engine.finish(), expected, atol=1e-12)
    with pytest.raises(AudioContractError):
        engine.push(np.ones(1))
    with pytest.raises(AudioContractError):
        engine.finish()


@pytest.mark.parametrize("block", [0, 63, 65, True, 32768])
def test_invalid_partition_block(block):
    with pytest.raises(AudioContractError):
        PartitionGeometry(1, 1, 5, block)


def test_geometry_budget_and_fixed_fft_size():
    g = PartitionGeometry(1, 1, 4097, 64)
    assert g.partitions == 65
    assert g.bins == 65
    assert g.fft_size == 128
    assert g.blocks_for(64, False) == 1
    assert g.blocks_for(64, True) == 65
    with pytest.raises(AudioLimitError):
        g.validate_work(10_000_000, include_tail=True)
    with pytest.raises(AudioLimitError):
        PartitionGeometry(8, 8, 262144, 64)


def test_tail_budget_failure_can_retry_without_tail(monkeypatch):
    import sonumio.audio_core.partitioned_convolution as module
    impulse = ImpulseMatrix(np.ones((1, 1, 129)), 8000, block_frames=64)
    engine = PartitionedConvolver(impulse)
    engine.push(np.ones(64))
    monkeypatch.setattr(module, "MAX_PARTITION_PRODUCTS", 200)
    with pytest.raises(AudioLimitError):
        engine.finish()
    assert not engine.finished
    assert engine.input_frames == engine.emitted_frames == 64
    assert engine.finish(include_tail=False).shape == (1, 0)


def test_layout_and_clock_checks():
    impulse = ImpulseMatrix(np.ones((2, 1, 5)), 8000, block_frames=64)
    audio = make_buffer(np.ones(12), 8000)
    layout = AudioLayout(2, "discrete", ("front", "rear"))
    result = convolve_partitioned(audio, impulse, output_layout=layout)
    assert result.spec.layout == layout
    assert result.samples.dtype == np.float32
    assert not result.samples.flags.writeable
    with pytest.raises(AudioContractError):
        convolve_partitioned(make_buffer(np.ones(12), 16000), impulse)
    with pytest.raises(AudioContractError):
        convolve_partitioned(audio, impulse, output_layout=AudioLayout.for_channels(1))
    with pytest.raises(AudioContractError):
        convolve_partitioned(audio, impulse, include_tail=1)
