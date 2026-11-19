"""Frame-exact opt-in edge fades for hash-verified local PCM processing."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.errors import AudioContractError, AudioLimitError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_pipeline import WavProcessingConfig, process_wav_file
from sonumio.audio_core.wave_reader import WavPacketReader
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.cli import main


def _source(tmp_path, samples, bits=16):
    pcm = np.asarray(samples, dtype=np.float32)
    path = tmp_path / "input.wav"
    with WavPacketWriter(path, WavPacketSpec(
        24000, pcm.shape[0], pcm.shape[1], QuantizationSettings(bits=bits),
    )) as writer:
        writer.push(FramePacket(24000, 0, pcm))
        record = writer.finish()
    return path, record.wav_sha256


def _read(path):
    with WavPacketReader(path, packet_frames=13) as reader:
        chunks = [packet.samples for packet in reader]
        assert reader.verified_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    return np.concatenate(chunks, axis=1)


def test_opt_in_linear_fades_are_packet_invariant_and_preserve_length(tmp_path):
    source, source_sha = _source(tmp_path, np.full((2, 61), 0.5, dtype=np.float32))
    digests = set()
    for size in (1, 3, 7, 61):
        out = tmp_path / f"faded-{size}.wav"
        result = process_wav_file(
            source, out, expected_source_sha256=source_sha,
            config=WavProcessingConfig(
                fade_in_frames=7, fade_out_frames=11, packet_frames=size,
            ),
        )
        assert result.frames == result.source_frames == 61
        assert result.fade_in_frames == 7
        assert result.fade_out_frames == 11
        assert result.source_sha256 == source_sha
        assert result.output_sha256 == hashlib.sha256(out.read_bytes()).hexdigest()
        values = _read(out)
        frame = np.arange(61, dtype=np.float64)
        expected = 0.5 * np.minimum(frame / 6, 1.0)
        expected *= np.minimum((60-frame) / 10, 1.0)
        np.testing.assert_allclose(values, expected[None, :], rtol=0, atol=1/32768)
        assert np.all(values[:, 0] == 0) and np.all(values[:, -1] == 0)
        assert np.all(values[:, 7:51] == 0.5)
        digests.add(result.output_sha256)
    assert len(digests) == 1


def test_trim_dc_dither_and_overlapping_fades_have_stable_hashes(tmp_path):
    samples = np.zeros((2, 77), dtype=np.float32)
    samples[0, 7:60] = 0.15
    samples[1, 9:69] = -0.25
    source, sha = _source(tmp_path, samples)
    outputs = []
    for size in (1, 4, 13, 77):
        dest = tmp_path / f"mix-{size}.wav"
        result = process_wav_file(
            source, dest, expected_source_sha256=sha,
            config=WavProcessingConfig(
                trim_silence_dbfs=-35, remove_dc=True,
                fade_in_frames=40, fade_out_frames=40,
                dither_seed=923, output_bits=24, packet_frames=size,
            ),
        )
        assert (result.retained_start_frame, result.retained_end_frame) == (7, 69)
        assert result.frames == 62 and result.dc_removed and result.dither_applied
        outputs.append(dest.read_bytes())
    assert len(set(outputs)) == 1


@pytest.mark.parametrize("bad", [-1, True, 1.5, "12", None, 1_920_001])
def test_invalid_fade_lengths_rejected_before_io(bad):
    with pytest.raises(AudioContractError, match="fade_in_frames"):
        WavProcessingConfig(fade_in_frames=bad)
    with pytest.raises(AudioContractError, match="fade_out_frames"):
        WavProcessingConfig(fade_out_frames=bad)


@pytest.mark.parametrize("fade", ["in", "out"])
def test_too_long_fade_does_not_publish_or_create_parent(tmp_path, fade):
    source, _ = _source(tmp_path, np.full((1, 17), 0.25, dtype=np.float32))
    target = tmp_path / "not-created" / "no.wav"
    config = WavProcessingConfig(**{f"fade_{fade}_frames": 18})
    with pytest.raises(AudioLimitError, match="fade exceeds"):
        process_wav_file(source, target, config=config, create_parent=True)
    assert not target.parent.exists()
    assert not list(tmp_path.rglob(".sonumio-stream-*.partial"))


def test_overlong_fade_preserves_existing_overwrite_destination(tmp_path):
    source, _ = _source(tmp_path, np.full((1, 6), 0.25, dtype=np.float32))
    target = tmp_path / "existing.wav"
    target.write_bytes(b"existing user data")
    with pytest.raises(AudioLimitError, match="fade exceeds"):
        process_wav_file(
            source, target, overwrite=True,
            config=WavProcessingConfig(fade_out_frames=7),
        )
    assert target.read_bytes() == b"existing user data"


def test_one_frame_edge_fades_and_defaults(tmp_path):
    source, sha = _source(tmp_path, np.full((1, 11), 0.5, dtype=np.float32))
    untouched = process_wav_file(source, tmp_path / "untouched.wav")
    assert untouched.fade_in_frames == untouched.fade_out_frames == 0
    assert untouched.output_sha256 == sha
    faded = process_wav_file(
        source, tmp_path / "endpoints.wav",
        config=WavProcessingConfig(fade_in_frames=1, fade_out_frames=1),
    )
    values = _read(faded.destination)
    assert values[0, 0] == 0 and values[0, -1] == 0
    np.testing.assert_array_equal(values[0, 1:-1],
                                  np.full(9, 0.5, dtype=np.float32))


def test_cli_exposes_exact_frame_controls_and_report(tmp_path, capsys):
    source, sha = _source(tmp_path, np.full((1, 31), 0.5, dtype=np.float32))
    out = tmp_path / "cli.wav"
    assert main([
        "wav-process", str(source), "--output", str(out),
        "--expected-sha256", sha, "--packet-frames", "3",
        "--fade-in-frames", "4", "--fade-out-frames", "6",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["fade_in_frames"] == 4
    assert report["fade_out_frames"] == 6
    assert report["source_frames"] == report["frames"] == 31
    assert report["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    samples = _read(out)
    assert samples[0, 0] == samples[0, -1] == 0
    assert samples[0, 5] == 0.5
