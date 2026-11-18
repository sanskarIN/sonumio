"""Verify full produced WAV digest before any atomic destination publication."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_pipeline import WavProcessingConfig, process_wav_file
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter
from sonumio.cli import main


def _source(tmp_path):
    path = tmp_path / "source.wav"
    clock, frames = 24000, 301
    t = np.arange(frames, dtype=np.float32)
    pcm = np.stack([
        0.21 + 0.19 * np.sin(t / 13),
        -0.15 + 0.23 * np.cos(t / 17),
    ]).astype(np.float32)
    with WavPacketWriter(path, WavPacketSpec(clock, 2, frames)) as writer:
        for start in range(0, frames, 59):
            writer.push(FramePacket(clock, start, pcm[:, start:start+59]))
        record = writer.finish()
    return path, record.wav_sha256, pcm


def _scratch(tmp_path):
    return list(tmp_path.glob(".sonumio-stream-*.partial"))


@pytest.mark.parametrize("packet_frames", [1, 17, 301])
def test_writer_accepts_matching_full_output_hash(tmp_path, packet_frames):
    source, sha, pcm = _source(tmp_path)
    assert sha == hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "confirmed.wav"
    with WavPacketWriter(target, WavPacketSpec(24000, 2, 301)) as writer:
        for start in range(0, 301, packet_frames):
            writer.push(FramePacket(24000, start, pcm[:, start:start+packet_frames]))
        report = writer.finish(expected_sha256=sha)
    assert report.wav_sha256 == sha
    assert target.read_bytes() == source.read_bytes()
    assert not _scratch(tmp_path)


def test_writer_mismatch_does_not_publish_new_file(tmp_path):
    _, _, pcm = _source(tmp_path)
    target = tmp_path / "not-published.wav"
    with WavPacketWriter(target, WavPacketSpec(24000, 2, 301)) as writer:
        writer.push(FramePacket(24000, 0, pcm))
        with pytest.raises(AudioIntegrityError, match="output SHA256 mismatch"):
            writer.finish(expected_sha256="0" * 64)
        assert writer.state == "aborted"
    assert not target.exists()
    assert not _scratch(tmp_path)


def test_writer_mismatch_preserves_explicit_overwrite_destination(tmp_path):
    _, _, pcm = _source(tmp_path)
    target = tmp_path / "do-not-clobber.wav"
    sentinel = b"existing file must survive"
    target.write_bytes(sentinel)
    with WavPacketWriter(
        target, WavPacketSpec(24000, 2, 301), overwrite=True,
    ) as writer:
        writer.push(FramePacket(24000, 0, pcm))
        with pytest.raises(AudioIntegrityError, match="output SHA256 mismatch"):
            writer.finish(expected_sha256="f" * 64)
    assert target.read_bytes() == sentinel
    assert not _scratch(tmp_path)


@pytest.mark.parametrize("invalid", ["", "abc", "A"*64, "g"*64, 1, True])
def test_writer_rejects_invalid_expected_hash_before_finishing(tmp_path, invalid):
    _, _, pcm = _source(tmp_path)
    target = tmp_path / "invalid-hash.wav"
    with WavPacketWriter(target, WavPacketSpec(24000, 2, 301)) as writer:
        writer.push(FramePacket(24000, 0, pcm))
        with pytest.raises(AudioContractError, match="SHA256"):
            writer.finish(expected_sha256=invalid)
        assert writer.state == "open"
        assert not target.exists()
        writer.finish()
    assert target.exists()
    assert not _scratch(tmp_path)


@pytest.mark.parametrize("packet_frames", [1, 19, 301])
def test_two_pass_pipeline_matching_hash_after_opt_in_processing(tmp_path, packet_frames):
    source, source_sha, _ = _source(tmp_path)
    config = WavProcessingConfig(remove_dc=True, output_bits=24,
                                 dither_seed=17, packet_frames=packet_frames)
    baseline = process_wav_file(source, tmp_path / f"baseline-{packet_frames}.wav",
                                config=config, expected_source_sha256=source_sha)
    checked = process_wav_file(
        source, tmp_path / f"verified-{packet_frames}.wav",
        config=config, expected_source_sha256=source_sha,
        expected_output_sha256=baseline.output_sha256,
    )
    assert checked.output_sha256 == baseline.output_sha256
    assert checked.dc_removed and checked.dither_applied
    assert checked.source_sha256 == source_sha
    assert not _scratch(tmp_path)


def test_pipeline_mismatch_does_not_replace_existing_file(tmp_path):
    source, source_sha, _ = _source(tmp_path)
    dest = tmp_path / "preserve.wav"
    original = b"important old result"
    dest.write_bytes(original)
    with pytest.raises(AudioIntegrityError, match="output SHA256 mismatch"):
        process_wav_file(
            source, dest, expected_source_sha256=source_sha,
            expected_output_sha256="0" * 64, overwrite=True,
        )
    assert dest.read_bytes() == original
    assert not _scratch(tmp_path)


def test_pipeline_rejects_bad_digest_without_creating_parent(tmp_path):
    source, _, _ = _source(tmp_path)
    dest = tmp_path / "not-created" / "bad.wav"
    with pytest.raises(AudioContractError, match="SHA256"):
        process_wav_file(
            source, dest, expected_output_sha256="invalid",
            create_parent=True,
        )
    assert not dest.parent.exists()
    assert not _scratch(tmp_path)


def test_cli_gate_matches_then_rejects_without_clobber(tmp_path, capsys):
    source, source_sha, _ = _source(tmp_path)
    expected = tmp_path / "expected.wav"
    assert main(["wav-process", str(source), "--output", str(expected),
                 "--expected-sha256", source_sha, "--remove-dc"]) == 0
    valid_hash = json.loads(capsys.readouterr().out)["output_sha256"]
    target = tmp_path / "verified.wav"
    assert main(["wav-process", str(source), "--output", str(target),
                 "--expected-sha256", source_sha, "--remove-dc",
                 "--expected-output-sha256", valid_hash]) == 0
    assert json.loads(capsys.readouterr().out)["output_sha256"] == valid_hash
    before = target.read_bytes()
    assert main(["wav-process", str(source), "--output", str(target),
                 "--expected-sha256", source_sha, "--remove-dc",
                 "--expected-output-sha256", "0"*64, "--overwrite"]) == 2
    assert "output SHA256 mismatch" in capsys.readouterr().err
    assert target.read_bytes() == before
    assert not _scratch(tmp_path)
