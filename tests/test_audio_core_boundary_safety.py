"""Resource-exhaustion and filesystem-race regressions for local PCM transport."""
from __future__ import annotations

import os

import numpy as np
import pytest

from sonumio.audio_core.contracts import AudioLayout, make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioFormatError, AudioLimitError
from sonumio.audio_core.streaming import (
    FramePacket, FramerConfig, PacketCollector, StreamingFramer,
)
from sonumio.audio_core.waveio import read_wav_file, write_wav_file
import sonumio.audio_core.streaming as streaming
import sonumio.audio_core.waveio as waveio


def _mono(frames=256):
    return make_buffer(
        np.sin(np.arange(frames, dtype=np.float32) / 13).astype(np.float32),
        8000,
    )


def test_packet_collector_rejects_excessive_tiny_packets_without_advancing(monkeypatch):
    monkeypatch.setattr(streaming, "MAX_COLLECTOR_PACKETS", 3)
    collector = PacketCollector(
        sample_rate=8000, layout=AudioLayout.for_channels(1), maximum_frames=20,
    )
    for offset in range(3):
        collector.push(FramePacket(8000, offset, np.array([0.1], dtype=np.float32)))
    with pytest.raises(AudioLimitError, match="packet count"):
        collector.push(FramePacket(8000, 3, np.array([0.2], dtype=np.float32)))
    assert collector.frames_seen == 3
    assert not collector.finished
    out = collector.finish()
    assert out.frames == 3
    np.testing.assert_allclose(out.samples, [[0.1, 0.1, 0.1]])


def test_streaming_dense_overlap_is_rejected_without_mutating_framer():
    config = FramerConfig(window_frames=16, hop_frames=1)
    framer = StreamingFramer(sample_rate=8000, channels=1, config=config)
    monster = FramePacket(8000, 0, np.zeros(50000, dtype=np.float32))
    with pytest.raises(AudioLimitError, match="smaller FramePacket"):
        framer.push(monster)
    assert framer.received == 0
    assert framer._start == 0
    assert framer._buffer.shape == (1, 0)
    accepted = framer.push(FramePacket(
        8000, 0, np.arange(32, dtype=np.float32) / 64
    ))
    assert len(accepted) == 17
    assert accepted[0].start_frame == 0
    assert accepted[-1].start_frame == 16


def test_streaming_large_windows_budget_independent_of_output_count():
    # 1,436 overlapping windows x 8 x 8,192 x 4B exceeds 64 MiB,
    # despite the number of windows being less than 16,384.
    framer = StreamingFramer(
        sample_rate=8000, channels=8,
        config=FramerConfig(window_frames=8192, hop_frames=64),
    )
    source = FramePacket(8000, 0, np.zeros((8, 100000), dtype=np.float32))
    with pytest.raises(AudioLimitError, match="materialization budget"):
        framer.push(source)
    assert framer.received == 0
    assert framer.finish() == ()


def test_streaming_budget_accepts_partitioned_equivalent_data():
    data = np.sin(np.arange(1000, dtype=np.float32) / 7)
    config = FramerConfig(window_frames=32, hop_frames=2, pad_last=True)
    framer = StreamingFramer(sample_rate=8000, channels=1, config=config)
    windows = []
    for start in range(0, 1000, 100):
        windows.extend(framer.push(FramePacket(8000, start, data[start:start+100])))
    windows.extend(framer.finish())
    assert windows[0].start_frame == 0
    assert windows[-1].start_frame < 1000
    for item in windows:
        np.testing.assert_array_equal(
            item.samples[0, :item.original_frames],
            data[item.start_frame:item.start_frame + item.original_frames],
        )


def test_atomic_wav_no_clobber_when_destination_appears_after_preflight(
    tmp_path, monkeypatch
):
    target = tmp_path / "race.wav"
    real_link = os.link
    contenders_bytes = b"written-by-another-process"

    def simulate_competitor(source, destination, *args, **kwargs):
        target.write_bytes(contenders_bytes)
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(waveio.os, "link", simulate_competitor)
    with pytest.raises(AudioFormatError, match="overwrite"):
        write_wav_file(_mono(), target)
    assert target.read_bytes() == contenders_bytes
    assert not list(tmp_path.glob(".sonumio-*.partial"))


def test_explicit_wav_overwrite_remains_atomic(tmp_path):
    path = tmp_path / "replacement.wav"
    first = _mono(128)
    second = _mono(192)
    write_wav_file(first, path)
    write_wav_file(second, path, overwrite=True)
    assert read_wav_file(path).frames == second.frames
    assert not list(tmp_path.glob(".sonumio-*.partial"))


def test_wav_read_refuses_directory_and_bounded_oversize(tmp_path, monkeypatch):
    with pytest.raises(AudioFormatError, match="ordinary"):
        read_wav_file(tmp_path)
    path = tmp_path / "small.wav"
    write_wav_file(_mono(), path)
    monkeypatch.setattr(waveio, "MAX_WAV_BYTES", 44)
    with pytest.raises(AudioLimitError, match="size cap"):
        read_wav_file(path)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX O_NOFOLLOW required")
def test_wav_read_rejects_symlink_swapped_during_open(tmp_path, monkeypatch):
    target = tmp_path / "destination.wav"
    decoy = tmp_path / "private.wav"
    write_wav_file(_mono(), target)
    write_wav_file(_mono(128), decoy)
    original_open = os.open

    def swap_before_open(path, flags, *args, **kwargs):
        if str(path) == str(target):
            target.unlink()
            target.symlink_to(decoy)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(waveio.os, "open", swap_before_open)
    with pytest.raises(AudioFormatError, match="ordinary"):
        read_wav_file(target)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFOs required")
def test_wav_read_refuses_fifo_without_waiting_for_writer(tmp_path):
    path = tmp_path / "named-pipe.wav"
    os.mkfifo(path)
    with pytest.raises(AudioFormatError, match="ordinary"):
        read_wav_file(path)


@pytest.mark.skipif(
    not (hasattr(os, "mkfifo") and hasattr(os, "O_NONBLOCK")),
    reason="POSIX nonblocking open required",
)
def test_wav_read_path_swap_to_fifo_does_not_hang(tmp_path, monkeypatch):
    target = tmp_path / "replaced-by-fifo.wav"
    write_wav_file(_mono(), target)
    original_open = os.open

    def swap_to_fifo(path, flags, *args, **kwargs):
        if str(path) == str(target):
            target.unlink()
            os.mkfifo(target)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(waveio.os, "open", swap_to_fifo)
    with pytest.raises(AudioFormatError, match="ordinary"):
        read_wav_file(target)


def test_wav_write_failure_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "no-links.wav"

    def hardlink_not_supported(*args, **kwargs):
        raise OSError("this file system forbids hard links")

    monkeypatch.setattr(waveio.os, "link", hardlink_not_supported)
    with pytest.raises(AudioFormatError, match="unsupported"):
        write_wav_file(_mono(), path)
    assert not path.exists()
    assert not list(tmp_path.glob(".sonumio-*.partial"))
