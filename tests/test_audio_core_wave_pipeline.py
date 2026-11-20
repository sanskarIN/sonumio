"""Validated long WAV streaming gain and two-pass integrity workflow."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

import sonumio.audio_core.wave_pipeline as wave_pipeline
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_pipeline import (
    WavProcessingConfig, WavProcessingResult, process_wav_file,
)
from sonumio.audio_core.wave_reader import WavPacketReader
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.audio_core.waveio import read_wav_file


def _wav(tmp_path, *, channels=1, bits=16, frames=500, level=0.6):
    source = tmp_path / "source.wav"
    values = (level * np.sin(
        np.arange(frames, dtype=np.float32) * 2 * np.pi / 63
    )).astype(np.float32)
    with WavPacketWriter(
        source, WavPacketSpec(8000, channels, frames,
                              QuantizationSettings(bits=bits)),
    ) as writer:
        for start in range(0, frames, 83):
            raw = np.stack([
                values[start:start + 83] * (1.0 - channel / 10.0)
                for channel in range(channels)
            ])
            writer.push(FramePacket(8000, start, raw))
        result = writer.finish()
    return source, result


@pytest.mark.parametrize("bits", [8, 16, 24])
@pytest.mark.parametrize("channels", [1, 2, 4])
@pytest.mark.parametrize("packet_frames", [1, 13, 257])
def test_identity_transcode_is_byte_for_byte_and_chunk_invariant(
    tmp_path, bits, channels, packet_frames
):
    source, original = _wav(tmp_path, channels=channels, bits=bits)
    result = process_wav_file(
        source, tmp_path / "copied.wav",
        config=WavProcessingConfig(packet_frames=packet_frames),
        expected_source_sha256=original.wav_sha256,
    )
    assert isinstance(result, WavProcessingResult)
    assert result.source_sha256 == original.wav_sha256
    assert result.output_sha256 == original.wav_sha256
    assert result.frames == 500
    assert result.source_bits == bits and result.output_bits == bits
    assert result.clipped_samples == 0
    assert result.original_model_inference is False
    assert (tmp_path / "copied.wav").read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_explicit_peak_reduction_applies_global_gain(tmp_path):
    source, result = _wav(tmp_path, level=0.85, frames=2048)
    processed = process_wav_file(
        source, tmp_path / "reduced.wav",
        config=WavProcessingConfig(
            gain_db=9, peak_ceiling_dbfs=-6, packet_frames=81,
        ),
        expected_source_sha256=result.wav_sha256,
    )
    expected_ceiling = 10**(-6 / 20)
    assert processed.source_peak > 0.8
    assert processed.output_peak_before_quantization <= expected_ceiling + 1e-6
    assert processed.output_rms_before_quantization < processed.source_rms
    assert processed.effective_gain_db < 0
    assert read_wav_file(processed.destination).frames == 2048


def test_peak_boost_only_with_explicit_consent(tmp_path):
    source, _ = _wav(tmp_path, level=.05)
    no_boost = process_wav_file(
        source, tmp_path / "unchanged.wav",
        config=WavProcessingConfig(peak_ceiling_dbfs=-1,
                                   allow_peak_boost=False),
    )
    boosted = process_wav_file(
        source, tmp_path / "boosted.wav",
        config=WavProcessingConfig(peak_ceiling_dbfs=-1,
                                   allow_peak_boost=True),
    )
    assert no_boost.effective_gain_db == pytest.approx(0)
    assert boosted.effective_gain_db == pytest.approx(12)
    assert boosted.output_peak_before_quantization > (
        no_boost.output_peak_before_quantization
    )


def test_default_clipping_policy_rejects_without_publishing(tmp_path):
    source, _ = _wav(tmp_path, level=0.7)
    output = tmp_path / "should-not-exist.wav"
    with pytest.raises(AudioLimitError, match="would clip"):
        process_wav_file(
            source, output,
            config=WavProcessingConfig(gain_db=18, packet_frames=43),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_explicit_hard_clipping_reports_number_of_clipped_samples(tmp_path):
    source, _ = _wav(tmp_path, level=0.7)
    result = process_wav_file(
        source, tmp_path / "hard.wav",
        config=WavProcessingConfig(gain_db=18, clip_policy="hard"),
    )
    assert result.clipped_samples > 0
    assert result.output_peak_before_quantization == 1.0
    assert result.output_rms_before_quantization <= 1.0
    assert read_wav_file(result.destination).frames == 500


def test_rejected_input_sha_does_not_create_output_or_parent(tmp_path):
    source, _ = _wav(tmp_path)
    destination = tmp_path / "new" / "blocked.wav"
    with pytest.raises(AudioIntegrityError, match="SHA256"):
        process_wav_file(
            source, destination, expected_source_sha256="0" * 64,
            create_parent=True,
        )
    assert not destination.parent.exists()


def test_source_mutation_between_passes_aborts_unpublished_wav(tmp_path, monkeypatch):
    source, _ = _wav(tmp_path, frames=205)
    output = tmp_path / "raced.wav"
    factory = wave_pipeline.WavPacketReader
    calls = 0

    def mutate_before_second_pass(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            with open(path, "r+b") as fp:
                fp.seek(44 + 18)
                old = fp.read(1)
                fp.seek(44 + 18)
                fp.write(bytes([old[0] ^ 1]))
        return factory(path, *args, **kwargs)

    monkeypatch.setattr(wave_pipeline, "WavPacketReader",
                        mutate_before_second_pass)
    with pytest.raises(AudioIntegrityError, match="SHA256"):
        process_wav_file(source, output, config=WavProcessingConfig(packet_frames=17))
    assert calls == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))


def test_same_file_and_hardlinked_alias_rejected_even_with_overwrite(tmp_path):
    source, _ = _wav(tmp_path)
    with pytest.raises(AudioContractError, match="differ"):
        process_wav_file(source, source, overwrite=True)
    alias = tmp_path / "alias.wav"
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("Host does not support hardlinks")
    with pytest.raises(AudioContractError, match="differ"):
        process_wav_file(source, alias, overwrite=True)
    assert source.exists()


def test_existing_destination_requires_explicit_overwrite(tmp_path):
    source, _ = _wav(tmp_path)
    dest = tmp_path / "other.wav"
    dest.write_bytes(b"occupied")
    with pytest.raises(AudioContractError, match="overwrite"):
        process_wav_file(source, dest)
    assert dest.read_bytes() == b"occupied"
    result = process_wav_file(source, dest, overwrite=True)
    assert result.destination == dest
    assert dest.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("bad_config", [
    {"gain_db": 40}, {"peak_ceiling_dbfs": 2},
    {"peak_ceiling_dbfs": -100}, {"clip_policy": "smooth"},
    {"allow_peak_boost": 1}, {"packet_frames": 0},
    {"packet_frames": 65537}, {"output_bits": 20}, {"output_bits": True},
])
def test_invalid_processing_settings_are_rejected(bad_config):
    with pytest.raises(AudioContractError):
        WavProcessingConfig(**bad_config)


def test_long_form_wav_processes_without_materializing_entire_recording(tmp_path):
    rate = 8000
    frames = 121 * rate
    source = tmp_path / "long.wav"
    with WavPacketWriter(source, WavPacketSpec(rate, 1, frames)) as writer:
        for start in range(0, frames, 8000):
            writer.push(FramePacket(
                rate, start, np.full(8000, 0.125, dtype=np.float32)
            ))
        original = writer.finish()
    processed = process_wav_file(
        source, tmp_path / "processed-long.wav",
        config=WavProcessingConfig(gain_db=-6, packet_frames=4096),
        expected_source_sha256=original.wav_sha256,
    )
    assert processed.frames == frames
    assert processed.channels == 1
    assert processed.output_bytes == 44 + frames * 2
    with WavPacketReader(processed.destination,
                         expected_sha256=processed.output_sha256) as reader:
        assert sum(packet.frames for packet in reader) == frames
        assert reader.verified_sha256 == processed.output_sha256
    assert not list(tmp_path.glob(".sonumio-stream-*.partial"))
