"""Streaming sinc reference equivalence, lifecycle and resource regressions."""
import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError, AudioLimitError
from sonumio.audio_core.resample import ResampleSettings, sinc_resample, output_frames
from sonumio.audio_core.stream_resample import StreamingSincResampler, MAX_RESAMPLE_INPUT_FRAMES
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import MAX_STREAM_PACKET_FRAMES


def convert(samples, rate, settings, sizes):
    processor = StreamingSincResampler(rate, samples.shape[0], samples.shape[1], settings)
    packets = []
    start = 0
    for count in sizes:
        end = min(samples.shape[1], start + count)
        if end == start:
            continue
        packets.extend(processor.push(FramePacket(rate, start, samples[:, start:end])))
        assert processor.retained_frames <= 2 * processor._width + 2
        start = end
    assert start == samples.shape[1]
    packets.extend(processor.finish())
    position = 0
    for packet in packets:
        assert packet.start_frame == position
        assert packet.sample_rate == settings.target_rate
        assert packet.frames <= MAX_STREAM_PACKET_FRAMES
        assert not packet.samples.flags.writeable
        position = packet.stop_frame
    assert position == processor.expected_output_frames
    assert processor.received_frames == samples.shape[1]
    assert processor.emitted_frames == position
    assert processor.retained_frames == 0
    assert processor.finished
    return np.concatenate([packet.samples for packet in packets], axis=1)


@pytest.mark.parametrize("rate,target", [(8000, 192000), (192000, 8000),
                                          (44100, 16000), (48000, 44100), (24000, 24000)])
@pytest.mark.parametrize("window", ["kaiser", "hann", "blackman"])
def test_streaming_matches_full_reference_across_packet_boundaries(rate, target, window):
    samples = np.random.default_rng(13).uniform(-.25, .25, (2, 337)).astype(np.float32)
    settings = ResampleSettings(target, lobes=8, window=window, chunk_frames=64)
    reference = sinc_resample(samples, rate, target, settings=settings)
    streamed = convert(samples, rate, settings, [1, 19, 3, 64, 7, 243])
    np.testing.assert_array_equal(streamed, reference)


@pytest.mark.parametrize("rate,target,n", [(48000, 8000, 4), (44100, 16000, 2),
                                            (8000, 192000, 1), (8000, 8000, 1)])
def test_short_clips_use_same_boundary_convention(rate, target, n):
    data = np.full((1, n), .5, dtype=np.float32)
    settings = ResampleSettings(target)
    np.testing.assert_allclose(convert(data, rate, settings, [1]*n), .5, atol=1e-6)


def test_zero_output_length_rejected():
    with pytest.raises(AudioContractError, match="zero frames"):
        StreamingSincResampler(192000, 1, 1, ResampleSettings(8000))


def test_lookahead_and_incomplete_finish_are_explicit():
    r = StreamingSincResampler(8000, 1, 100, ResampleSettings(16000))
    assert r.push(FramePacket(8000, 0, np.ones(10, dtype=np.float32))) == ()
    assert r.emitted_frames == 0
    with pytest.raises(AudioIntegrityError, match="Incomplete"):
        r.finish()
    assert not r.finished
    packets = r.push(FramePacket(8000, 10, np.ones(90, dtype=np.float32)))
    assert packets
    assert r.emitted_frames < r.expected_output_frames
    tail = r.finish()
    assert tail[-1].stop_frame == 200
    with pytest.raises(AudioContractError):
        r.finish()
    with pytest.raises(AudioContractError):
        r.push(FramePacket(8000, 100, np.ones(1, dtype=np.float32)))


def test_rejected_packets_do_not_advance_or_discard_context():
    r = StreamingSincResampler(8000, 2, 100, ResampleSettings(16000))
    r.push(FramePacket(8000, 0, np.zeros((2, 50), dtype=np.float32)))
    state = (r.received_frames, r.emitted_frames, r.retained_frames)
    for packet in (
        FramePacket(16000, 50, np.zeros((2, 5), dtype=np.float32)),
        FramePacket(8000, 49, np.zeros((2, 5), dtype=np.float32)),
        FramePacket(8000, 51, np.zeros((2, 5), dtype=np.float32)),
        FramePacket(8000, 50, np.zeros((1, 5), dtype=np.float32)),
        FramePacket(8000, 50, np.zeros((2, 51), dtype=np.float32)),
    ):
        with pytest.raises(AudioContractError):
            r.push(packet)
        assert (r.received_frames, r.emitted_frames, r.retained_frames) == state
    r.push(FramePacket(8000, 50, np.zeros((2, 50), dtype=np.float32)))
    r.finish()
    assert r.emitted_frames == 200


def test_packet_cap_rejected_before_state_change():
    n = MAX_RESAMPLE_INPUT_FRAMES + 1
    r = StreamingSincResampler(8000, 1, n, ResampleSettings(16000))
    with pytest.raises(AudioLimitError, match="input cap"):
        r.push(FramePacket(8000, 0, np.zeros(n, dtype=np.float32)))
    assert r.received_frames == r.emitted_frames == r.retained_frames == 0


@pytest.mark.parametrize("rate,target", [(8000, 192000), (192000, 8000)])
def test_large_ratio_retention_and_output_packet_bounds(rate, target):
    n = 10000
    samples = np.full((1, n), .125, dtype=np.float32)
    settings = ResampleSettings(target, lobes=4, chunk_frames=65536)
    result = convert(samples, rate, settings, [8192, 1808])
    assert result.shape == (1, output_frames(n, rate, target))
    np.testing.assert_allclose(result, .125, atol=1e-6)


def test_identity_is_exact_and_has_no_lookahead():
    r = StreamingSincResampler(16000, 1, 5, ResampleSettings(16000))
    packet = FramePacket(16000, 0, np.arange(5, dtype=np.float32))
    assert r.push(packet)[0] is packet
    assert r.retained_frames == 0
    assert r.finish() == ()


def test_long_stream_history_does_not_scale_with_total_duration():
    r = StreamingSincResampler(48000, 1, 100000, ResampleSettings(8000, lobes=4))
    for start in range(0, 100000, 1000):
        r.push(FramePacket(48000, start, np.full(1000, .25, dtype=np.float32)))
        assert r.retained_frames <= 2 * r._width + 2
    r.finish()
    assert r.emitted_frames == output_frames(100000, 48000, 8000)


@pytest.mark.parametrize("channels,frames,settings", [
    (0, 100, ResampleSettings(8000)),
    (9, 100, ResampleSettings(8000)),
    (True, 100, ResampleSettings(8000)),
    (1, 0, ResampleSettings(8000)),
    (1, 8000*3600+1, ResampleSettings(8000)),
    (1, 100, ResampleSettings(8000, method="linear")),
    (1, 100, None),
])
def test_invalid_configuration(channels, frames, settings):
    with pytest.raises(AudioContractError):
        StreamingSincResampler(8000, channels, frames, settings)
