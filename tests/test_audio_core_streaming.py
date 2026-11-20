"""Exact PCM packet ordering, windowing and bounded stream state tests."""
from __future__ import annotations

import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.streaming import (
    FramePacket, FramerConfig, PacketCollector, StreamingFramer,
    WindowPacket, frame_stream_report, offline_window_reference, split_packets,
)
from sonumio.audio_core.errors import AudioContractError, AudioLimitError


@pytest.fixture
def stereo():
    rng = np.random.default_rng(100)
    return make_buffer(
        rng.uniform(-.9, .9, (2, 2100)).astype(np.float32), 24000
    )


@pytest.mark.parametrize("packet_frames", [1, 7, 64, 127, 500, 2048, 3000])
def test_split_reassembly_preserves_exact_pcm_for_every_chunk_size(stereo, packet_frames):
    packets = split_packets(stereo, packet_frames=packet_frames)
    assert packets[0].start_frame == 0
    assert packets[-1].stop_frame == stereo.frames
    assert all(p.sample_rate == stereo.sample_rate for p in packets)
    assert all(p.channels == stereo.channels for p in packets)
    collector = PacketCollector(
        sample_rate=stereo.sample_rate,
        layout=stereo.spec.layout,
        maximum_frames=stereo.frames,
    )
    for packet in packets:
        collector.push(packet)
    assert collector.frames_seen == stereo.frames
    restored = collector.finish()
    np.testing.assert_array_equal(restored.samples, stereo.samples)
    assert restored.samples.flags.writeable is False
    assert restored.sample_rate == stereo.sample_rate
    assert collector.finished is True


def test_packet_readonly_ownership_and_absolute_slicing(stereo):
    raw = stereo.mutable_copy()
    packet = FramePacket(24000, 100, raw)
    raw[:] = 0
    np.testing.assert_array_equal(packet.samples, stereo.samples)
    with pytest.raises(ValueError):
        packet.samples[0, 0] = 55
    sliced = packet.sliced(300, 500)
    assert sliced.start_frame == 400
    assert sliced.stop_frame == 600
    assert sliced.frames == 200
    np.testing.assert_array_equal(sliced.samples, stereo.samples[:, 300:500])
    assert packet.as_buffer().channels == 2


def test_stream_packet_backing_cannot_reenable_numpy_write_flag(stereo):
    packet = FramePacket(24000, 0, stereo.samples)
    assert packet.samples.flags.writeable is False
    with pytest.raises(ValueError):
        packet.samples.setflags(write=True)
    packet_slice = packet.sliced(100, 150)
    assert packet_slice.samples.flags.writeable is False
    with pytest.raises(ValueError):
        packet_slice.samples.setflags(write=True)


def test_packet_sequence_rejects_gaps_overlaps_and_wrong_clocks(stereo):
    collector = PacketCollector(
        sample_rate=24000, layout=AudioLayout.for_channels(2),
        maximum_frames=2100
    )
    collector.push(FramePacket(24000, 0, stereo.samples[:, :100]))
    with pytest.raises(AudioContractError, match="gap|overlap|reordering"):
        collector.push(FramePacket(24000, 101, stereo.samples[:, 100:200]))
    with pytest.raises(AudioContractError, match="gap|overlap|reordering"):
        collector.push(FramePacket(24000, 99, stereo.samples[:, 100:200]))
    with pytest.raises(AudioContractError, match="clock"):
        collector.push(FramePacket(44100, 100, stereo.samples[:, 100:200]))
    with pytest.raises(AudioContractError):
        collector.push(FramePacket(24000, 100, stereo.samples[0, 100:200]))


def test_packet_collector_will_not_exceed_declared_maximum(stereo):
    collector = PacketCollector(
        sample_rate=24000, layout=AudioLayout.for_channels(2),
        maximum_frames=100,
    )
    with pytest.raises(AudioLimitError):
        collector.push(FramePacket(24000, 0, stereo.samples[:, :101]))


def test_collector_can_finalize_only_once(stereo):
    packets = split_packets(stereo, packet_frames=3000)
    collector = PacketCollector(
        sample_rate=24000, layout=AudioLayout.for_channels(2),
        maximum_frames=3000,
    )
    with pytest.raises(AudioContractError):
        collector.finish()
    collector.push(packets[0])
    collector.finish()
    with pytest.raises(AudioContractError):
        collector.finish()
    with pytest.raises(AudioContractError):
        collector.push(FramePacket(24000, stereo.frames,
                                   stereo.samples[:, :1]))


@pytest.mark.parametrize("hop,window", [
    (64, 128), (64, 256), (256, 256), (256, 512), (700, 700)
])
@pytest.mark.parametrize("pad_last", [False, True])
def test_chunk_boundaries_do_not_change_extracted_windows(
    stereo, hop, window, pad_last
):
    cfg = FramerConfig(window_frames=window, hop_frames=hop,
                       pad_last=pad_last)
    expected = offline_window_reference(stereo, cfg)
    outputs = []
    framer = StreamingFramer(sample_rate=24000, channels=2, config=cfg)
    for packet in split_packets(stereo, packet_frames=137):
        outputs.extend(framer.push(packet))
    outputs.extend(framer.finish())
    assert [x.start_frame for x in outputs] == [
        x.start_frame for x in expected
    ]
    assert [x.original_frames for x in outputs] == [
        x.original_frames for x in expected
    ]
    assert len(outputs) == len(expected)
    for actual, oracle in zip(outputs, expected):
        np.testing.assert_array_equal(actual.samples, oracle.samples)
    assert framer.received == stereo.frames


def test_short_packet_with_padding_only_emits_one_final_window():
    audio = make_buffer(np.ones(40, dtype=np.float32), 24000)
    cfg = FramerConfig(window_frames=128, hop_frames=32, pad_last=True)
    framer = StreamingFramer(sample_rate=24000, channels=1, config=cfg)
    output = framer.push(FramePacket(24000, 0, audio.samples))
    assert output == ()
    final = framer.finish()
    assert len(final) == 1
    assert final[0].start_frame == 0
    assert final[0].original_frames == 40
    np.testing.assert_array_equal(final[0].samples[0, :40], 1)
    np.testing.assert_array_equal(final[0].samples[0, 40:], 0)


def test_short_packet_without_padding_emits_no_frames():
    audio = make_buffer(np.ones(40, dtype=np.float32), 24000)
    cfg = FramerConfig(window_frames=128, hop_frames=32, pad_last=False)
    framer = StreamingFramer(sample_rate=24000, channels=1, config=cfg)
    assert framer.push(FramePacket(24000, 0, audio.samples)) == ()
    assert framer.finish() == ()


def test_framer_refuses_bad_windows_channels_and_post_close_writes(stereo):
    with pytest.raises(AudioContractError):
        FramerConfig(window_frames=10, hop_frames=5)
    with pytest.raises(AudioContractError):
        FramerConfig(window_frames=128, hop_frames=129)
    with pytest.raises(AudioContractError):
        StreamingFramer(sample_rate=24000, channels=9)
    cfg = FramerConfig(window_frames=128, hop_frames=32)
    framer = StreamingFramer(sample_rate=24000, channels=2, config=cfg)
    framer.finish()
    with pytest.raises(AudioContractError):
        framer.push(FramePacket(24000, 0, stereo.samples[:, :64]))


def test_window_report_is_numerical_not_neural_or_network_transport(stereo):
    cfg = FramerConfig(window_frames=256, hop_frames=128, pad_last=True)
    packets = split_packets(stereo, packet_frames=128)
    windows = offline_window_reference(stereo, cfg)
    report = frame_stream_report(packets, windows)
    assert report["stream_packets"] == len(packets)
    assert report["window_packets"] == len(windows)
    assert report["source_frames"] == 2100
    assert report["last_packet_stop"] == 2100
    assert report["model_inference"] is False
    assert report["network_streaming_implemented"] is False
    assert report["real_time_latency_guaranteed"] is False


def test_framer_rejects_noncontiguous_source_positions(stereo):
    framer = StreamingFramer(
        sample_rate=24000, channels=2,
        config=FramerConfig(256, 64),
    )
    framer.push(FramePacket(24000, 0, stereo.samples[:, :100]))
    with pytest.raises(AudioContractError):
        framer.push(FramePacket(24000, 200, stereo.samples[:, 200:300]))
