"""Atomic verified WAV resampling, PCM output and integrity controls."""
import hashlib
import json
import os

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError, AudioLimitError
from sonumio.audio_core.resample import ResampleSettings, sinc_resample
from sonumio.audio_core.wave_reader import WavPacketReader
from sonumio.audio_core.wave_resample import resample_wav_file
from sonumio.audio_core.waveio import write_wav_file, read_wav_file


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_file(tmp_path, frames=1001, rate=24000):
    path = tmp_path / "source.wav"
    t = np.arange(frames) / rate
    samples = np.stack([.25*np.sin(2*np.pi*250*t), .2*np.cos(2*np.pi*730*t)])
    write_wav_file(make_buffer(samples.astype(np.float32), rate), path)
    return path


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_verified_wav_reference_matches_offline(tmp_path, bits):
    src = source_file(tmp_path)
    dst = tmp_path / "target.wav"
    settings = ResampleSettings(16000, lobes=8)
    result = resample_wav_file(src, dst, settings=settings, output_bits=bits,
                               expected_source_sha256=digest(src), packet_frames=31)
    with WavPacketReader(dst) as reader:
        data = np.concatenate([p.samples for p in reader], axis=1)
        assert reader.verified_sha256 == result.output_sha256
        assert reader.info.bits_per_sample == bits
    reference = sinc_resample(read_wav_file(src).samples, 24000, 16000, settings=settings)
    np.testing.assert_allclose(data, reference, atol=max(2**(1-bits), 1e-7))
    assert result.source_sha256 == digest(src)
    assert result.output_sha256 == digest(dst)
    assert result.output_bytes == dst.stat().st_size
    assert result.output_frames == reference.shape[1]
    assert result.source_frames == 1001
    assert result.clipped_samples == 0
    assert json.loads(json.dumps(result.as_dict()))["target_rate"] == 16000


@pytest.mark.parametrize("seed", [None, 42])
def test_output_bytes_are_packet_partition_invariant(tmp_path, seed):
    src = source_file(tmp_path)
    settings = ResampleSettings(44100, lobes=8)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    first = resample_wav_file(src, a, settings=settings, dither_seed=seed, packet_frames=17)
    second = resample_wav_file(src, b, settings=settings, dither_seed=seed, packet_frames=701,
                               expected_output_sha256=first.output_sha256)
    assert first.output_sha256 == second.output_sha256
    assert a.read_bytes() == b.read_bytes()


def test_wrong_source_hash_never_publishes_or_leaves_scratch(tmp_path):
    src = source_file(tmp_path)
    dst = tmp_path / "target.wav"
    with pytest.raises(AudioIntegrityError, match="source SHA256"):
        resample_wav_file(src, dst, settings=ResampleSettings(8000),
                          expected_source_sha256="0"*64, packet_frames=19)
    assert not dst.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob(".sonumio-stream-*"))


def test_wrong_output_hash_preserves_existing_destination(tmp_path):
    src = source_file(tmp_path)
    dst = tmp_path / "target.wav"
    dst.write_bytes(b"previous contents")
    with pytest.raises(AudioIntegrityError, match="output SHA256"):
        resample_wav_file(src, dst, settings=ResampleSettings(8000),
                          expected_output_sha256="0"*64, overwrite=True)
    assert dst.read_bytes() == b"previous contents"
    assert not list(tmp_path.glob(".sonumio-stream-*"))


def test_aliases_rejected_before_publication(tmp_path):
    src = source_file(tmp_path)
    with pytest.raises(AudioContractError, match="paths must differ"):
        resample_wav_file(src, src, settings=ResampleSettings(8000), overwrite=True)
    dst = tmp_path / "linked.wav"
    try:
        os.link(src, dst)
    except OSError:
        pytest.skip("hard links unavailable")
    with pytest.raises(AudioContractError, match="files must differ"):
        resample_wav_file(src, dst, settings=ResampleSettings(8000), overwrite=True)
    assert digest(src) == digest(dst)


def test_sinc_overshoot_requires_explicit_clip_policy(tmp_path):
    src, dst = tmp_path / "source.wav", tmp_path / "target.wav"
    write_wav_file(make_buffer(np.array([-1, .999, .999, -1], dtype=np.float32), 8000), src)
    settings = ResampleSettings(32000)
    with pytest.raises(AudioLimitError, match="normalized PCM"):
        resample_wav_file(src, dst, settings=settings)
    assert not dst.exists()
    result = resample_wav_file(src, dst, settings=settings, clip_policy="hard")
    assert result.clipped_samples > 0
    assert result.peak_before_clipping > 1
    assert np.max(np.abs(read_wav_file(dst).samples)) <= 1
    assert not list(tmp_path.glob(".sonumio-stream-*"))


def test_default_no_clobber_and_explicit_overwrite(tmp_path):
    src = source_file(tmp_path)
    dst = tmp_path / "target.wav"
    dst.write_bytes(b"old")
    with pytest.raises(AudioContractError, match="overwrite"):
        resample_wav_file(src, dst, settings=ResampleSettings(8000))
    assert dst.read_bytes() == b"old"
    result = resample_wav_file(src, dst, settings=ResampleSettings(8000), overwrite=True)
    assert digest(dst) == result.output_sha256
