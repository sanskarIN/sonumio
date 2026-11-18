"""Bounded two-pass WAV pipeline keeps seeded TPDF chunk invariant."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.audio_core.wave_pipeline import WavProcessingConfig, process_wav_file
from sonumio.audio_core.stream_dither import STREAM_TPDF_VERSION
from sonumio.cli import main


def _approved_source(tmp_path):
    source = tmp_path / "input.wav"
    frames = 293
    t = np.arange(frames, dtype=np.float32)
    stereo = np.stack((0.33 * np.sin(t / 7), 0.27 * np.cos(t / 13))).astype(np.float32)
    with WavPacketWriter(source, WavPacketSpec(24000, 2, frames)) as writer:
        for pos in range(0, frames, 57):
            writer.push(FramePacket(24000, pos, stereo[:, pos:pos + 57]))
        info = writer.finish()
    return source, info.wav_sha256


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
@pytest.mark.parametrize("seed", [0, 7, 4294967295])
def test_two_pass_dithered_pcm_invariant_to_reader_packet_size(tmp_path, bits, seed):
    source, sha = _approved_source(tmp_path)
    outputs = []
    for packet_size in (1, 3, 101, 293):
        target = tmp_path / ("out-" + str(packet_size) + ".wav")
        result = process_wav_file(
            source, target,
            config=WavProcessingConfig(
                packet_frames=packet_size, output_bits=bits,
                gain_db=-2.0, dither_seed=seed,
            ),
            expected_source_sha256=sha,
        )
        assert result.source_sha256 == sha
        assert result.dither_applied is True
        assert result.dither_seed == seed
        assert result.dither_protocol == STREAM_TPDF_VERSION
        assert result.output_sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
        assert result.frames == 293
        outputs.append(target.read_bytes())
    assert outputs[0] == outputs[1] == outputs[2] == outputs[3]


def test_dither_seed_changes_output_but_default_remains_undithered(tmp_path):
    source, sha = _approved_source(tmp_path)
    paths = [tmp_path / x for x in ("one.wav", "two.wav", "none.wav")]
    options = [WavProcessingConfig(dither_seed=10),
               WavProcessingConfig(dither_seed=11),
               WavProcessingConfig()]
    records = [process_wav_file(source, path, config=config,
                                expected_source_sha256=sha)
               for path, config in zip(paths, options)]
    assert paths[0].read_bytes() != paths[1].read_bytes()
    assert records[0].dither_applied is True
    assert records[2].dither_applied is False
    assert records[2].dither_seed is None
    assert records[2].dither_protocol is None


@pytest.mark.parametrize("bad", [-1, 2**32, True, 1.5, "5"])
def test_invalid_dither_seed_fails_at_config_boundary(bad):
    with pytest.raises(AudioContractError, match="dither"):
        WavProcessingConfig(dither_seed=bad)


def test_cli_dither_is_explicit_and_reports_version(tmp_path, capsys):
    source, sha = _approved_source(tmp_path)
    target = tmp_path / "cli.wav"
    code = main([
        "wav-process", str(source), "--output", str(target),
        "--expected-sha256", sha, "--packet-frames", "19",
        "--dither-seed", "1009",
    ])
    assert code == 0
    import json
    report = json.loads(capsys.readouterr().out)
    assert report["dither_applied"] is True
    assert report["dither_seed"] == 1009
    assert report["dither_protocol"] == STREAM_TPDF_VERSION
    assert hashlib.sha256(target.read_bytes()).hexdigest() == report["output_sha256"]


def test_bad_cli_dither_seed_cannot_create_wav(tmp_path, capsys):
    source, _ = _approved_source(tmp_path)
    target = tmp_path / "rejected.wav"
    assert main([
        "wav-process", str(source), "--output", str(target),
        "--dither-seed", "-3",
    ]) == 2
    assert not target.exists()
    assert "dither" in capsys.readouterr().err
