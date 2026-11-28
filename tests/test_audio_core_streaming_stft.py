"""Fixed-clock incremental STFT/ISTFT parity, lifecycle and limits."""
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.stft import StftConfig, stft, istft
from sonumio.audio_core.streaming_stft import (
    StreamingStft, StreamingIstft, streaming_stft, streaming_istft,
)


def audio(frames, channels=1, seed=47):
    rng = np.random.default_rng(seed)
    samples = rng.normal(0, .12, (channels, frames)).astype(np.float32)
    return make_buffer(samples, 8000)


@pytest.mark.parametrize("frames", [1, 13, 63, 64, 65, 127, 128, 129, 351, 512])
@pytest.mark.parametrize("channels", [1, 2, 3])
@pytest.mark.parametrize("hop", [16, 32])
def test_streaming_fft_matches_offline_for_arbitrary_short_pcm(frames, channels, hop):
    pcm = audio(frames, channels)
    cfg = StftConfig(n_fft=64, hop_length=hop, center=True, pad_end=True)
    reference = stft(pcm, cfg)
    actual = streaming_stft(pcm, cfg, push_frames=7)
    assert actual.original_frames == reference.original_frames
    assert actual.frames == reference.frames
    assert actual.sample_rate == reference.sample_rate
    assert actual.config == reference.config
    np.testing.assert_allclose(actual.spectrum, reference.spectrum,
                               rtol=1e-12, atol=1e-11)
    assert not actual.spectrum.flags.writeable


@pytest.mark.parametrize("frames", [64, 65, 127, 128, 131, 256, 257, 351])
@pytest.mark.parametrize("pad_end", [False, True])
def test_unpadded_analysis_uses_identical_tail_policy(frames, pad_end):
    pcm = audio(frames)
    cfg = StftConfig(n_fft=64, hop_length=16,
                     center=False, pad_end=pad_end)
    expected = stft(pcm, cfg)
    actual = streaming_stft(pcm, cfg, push_frames=9)
    np.testing.assert_allclose(actual.spectrum, expected.spectrum, atol=1e-12)
    assert actual.frames == expected.frames


@pytest.mark.parametrize("frames", [1, 64, 65, 127, 128, 129, 200, 1000])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("chunk", [1, 7, 37])
def test_streaming_istft_reconstructs_offline_waveform(frames, channels, chunk):
    original = audio(frames, channels)
    config = StftConfig(n_fft=64, hop_length=16)
    spectrum = stft(original, config)
    actual = streaming_istft(spectrum, push_frames=chunk)
    offline = istft(spectrum)
    assert actual.spec == offline.spec
    np.testing.assert_allclose(actual.samples, offline.samples, atol=1e-7)
    np.testing.assert_allclose(actual.samples, original.samples, atol=1e-6)


def test_no_centered_padding_disallows_missing_unpadded_data():
    cfg = StftConfig(n_fft=64, hop_length=16, center=False, pad_end=False)
    signal = audio(40)
    engine = StreamingStft(1, 8000, cfg)
    assert engine.push(signal.samples).shape == (1, 33, 0)
    with pytest.raises(AudioContractError):
        engine.finish()
    assert not engine.finished
    # A partial prefix remains valid; additional samples can complete it.
    more = np.zeros((1, 24), dtype=np.float32)
    frames = engine.push(more)
    assert frames.shape == (1, 33, 1)
    assert engine.finish().shape == (1, 33, 0)
    assert engine.finished


def test_chunk_boundaries_do_not_affect_fft_frame_values():
    pcm = audio(1403, 2)
    cfg = StftConfig(n_fft=256, hop_length=64)
    a = streaming_stft(pcm, cfg, push_frames=1403)
    b = streaming_stft(pcm, cfg, push_frames=1)
    c = streaming_stft(pcm, cfg, push_frames=19)
    np.testing.assert_array_equal(a.spectrum, b.spectrum)
    np.testing.assert_array_equal(a.spectrum, c.spectrum)


def test_streaming_inverse_chunk_boundaries_are_invariant():
    pcm = audio(2021, 2)
    cfg = StftConfig(n_fft=256, hop_length=64)
    spec = stft(pcm, cfg)
    a = streaming_istft(spec, push_frames=1)
    b = streaming_istft(spec, push_frames=5)
    c = streaming_istft(spec, push_frames=spec.frames)
    np.testing.assert_array_equal(a.samples, b.samples)
    np.testing.assert_array_equal(a.samples, c.samples)


def test_stream_analysis_rejects_invalid_push_without_advancing_state():
    cfg = StftConfig(n_fft=64, hop_length=16)
    engine = StreamingStft(1, 8000, cfg)
    for bad in (
        np.zeros((2, 10), dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf]),
        np.array([1, 2, 3], dtype=np.int32),
        np.zeros(0, dtype=np.float32),
    ):
        with pytest.raises(AudioContractError):
            engine.push(bad)
        assert engine.input_frames == 0
        assert engine.output_frames == 0
    with pytest.raises(AudioContractError):
        engine.finish()
    assert engine.push(np.ones(64, dtype=np.float32)).shape[2] > 0
    engine.finish()
    assert engine.finished
    with pytest.raises(AudioContractError):
        engine.push(np.ones(1, dtype=np.float32))
    with pytest.raises(AudioContractError):
        engine.finish()


def test_inverse_rejects_unexpected_shape_and_dtype_without_mutation():
    cfg = StftConfig(n_fft=64, hop_length=16)
    engine = StreamingIstft(1, 8000, 100, cfg)
    for bad in (
        np.zeros((1, 33, 1)),
        np.zeros((2, 33, 1), dtype=np.complex128),
        np.zeros((1, 32, 1), dtype=np.complex128),
        np.zeros((1, 33, 0), dtype=np.complex128),
        np.full((1, 33, 1), complex(np.nan, 0)),
    ):
        with pytest.raises(AudioContractError):
            engine.push(bad)
        assert engine.spectral_frames == 0
        assert engine.output_frames == 0
    with pytest.raises(AudioContractError):
        engine.finish()
    assert not engine.finished


def test_inverse_requires_exact_number_of_frames():
    signal = audio(267)
    spec = stft(signal, StftConfig(n_fft=64, hop_length=16))
    engine = StreamingIstft(1, 8000, signal.frames, spec.config)
    first = engine.push(spec.spectrum[:, :, :2])
    with pytest.raises(AudioContractError):
        engine.finish()
    assert not engine.finished
    rest = engine.push(spec.spectrum[:, :, 2:])
    end = engine.finish()
    result = np.concatenate((first, rest, end), axis=1)
    np.testing.assert_allclose(result, signal.samples, atol=1e-6)


def test_inverse_rejects_more_frames_than_geometry_allows():
    signal = audio(200)
    spec = stft(signal, StftConfig(n_fft=64, hop_length=16))
    engine = StreamingIstft(1, 8000, signal.frames, spec.config)
    with pytest.raises(AudioContractError):
        engine.push(np.repeat(spec.spectrum, 2, axis=2))
    assert engine.spectral_frames == 0


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman", "sqrt_hann"])
@pytest.mark.parametrize("hop", [8, 16, 32])
def test_stft_window_options_match_exact_reference(window, hop):
    signal = audio(416, 2)
    cfg = StftConfig(n_fft=64, hop_length=hop, window=window)
    ref = stft(signal, cfg)
    sample = streaming_stft(signal, cfg, push_frames=3)
    np.testing.assert_allclose(sample.spectrum, ref.spectrum, atol=1e-12)
    inverted = streaming_istft(sample, push_frames=2)
    np.testing.assert_allclose(inverted.samples, istft(ref).samples, atol=1e-6)


def test_invalid_streaming_constructor_bounds():
    with pytest.raises(AudioContractError):
        StreamingStft(9, 8000)
    with pytest.raises(AudioContractError):
        StreamingStft(True, 8000)
    with pytest.raises(AudioContractError):
        StreamingStft(1, 8000, None)
    with pytest.raises(AudioContractError):
        StreamingIstft(1, 8000, 0)
    with pytest.raises(AudioContractError):
        StreamingIstft(1, 8000, 64, epsilon=.5)


def test_streaming_adapter_rejects_invalid_chunk_shapes():
    signal = audio(200)
    spec = stft(signal)
    with pytest.raises(AudioContractError):
        streaming_stft(signal, push_frames=0)
    with pytest.raises(AudioContractError):
        streaming_stft(signal, push_frames=True)
    with pytest.raises(AudioContractError):
        streaming_istft(spec, push_frames=0)
    with pytest.raises(AudioContractError):
        streaming_istft(None)


def test_source_and_inverse_are_immutable_audio_buffers():
    signal = audio(333, 2)
    before = signal.samples.copy()
    transformed = streaming_stft(signal, StftConfig(n_fft=128, hop_length=32))
    inverse = streaming_istft(transformed)
    np.testing.assert_array_equal(signal.samples, before)
    assert inverse.samples.dtype == np.float32
    assert not inverse.samples.flags.writeable
    with pytest.raises(ValueError):
        inverse.samples.setflags(write=True)
