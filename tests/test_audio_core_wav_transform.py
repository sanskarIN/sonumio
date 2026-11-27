"""Canonical PCM source verification, exact output and CLI integration."""
import hashlib
import json
import os
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.waveio import encode_wav_bytes, read_wav_file, inspect_wav_bytes
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError, AudioLimitError
from sonumio.audio_core.time_scale import TimeScaleSettings
from sonumio.audio_core.wav_transform import WavTransformConfig, transform_wav_file
from sonumio.cli import main


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "input.wav"
    t = np.arange(5001)/24000
    wave = (.2*np.sin(2*np.pi*440*t)).astype(np.float32)
    audio = make_buffer(np.stack((wave, -wave)), 24000)
    data = encode_wav_bytes(audio, settings=QuantizationSettings(bits=24))
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("method", ["wsola", "phase_vocoder"])
@pytest.mark.parametrize("operation", [{"ratio": 1.5}, {"target_frames": 3701}, {"semitones": 5}])
def test_verified_operations(source, tmp_path, method, operation):
    path, pin = source
    destination = tmp_path/"result.wav"
    config = WavTransformConfig(**operation, time_scale=TimeScaleSettings(
        method=method, window_frames=512, search_frames=64, peak_ceiling=.85))
    report = transform_wav_file(path, destination, config=config, expected_source_sha256=pin)
    data = destination.read_bytes()
    info, _ = inspect_wav_bytes(data)
    assert info.frames == config.output_frames(5001)
    assert info.channels == 2 and info.sample_rate == 24000
    assert info.bits_per_sample == 24
    assert report.source_sha256 == pin
    assert report.output_sha256 == hashlib.sha256(data).hexdigest()
    assert report.output_bytes == len(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin
    assert report.as_dict()["source_verified_before_publication"]
    signal = read_wav_file(destination).samples
    np.testing.assert_allclose(signal[0], -signal[1], atol=2e-7)


@pytest.mark.parametrize("bits", [8, 16, 24, 32])
def test_seeded_dither_and_packet_partition(source, tmp_path, bits):
    path, pin = source
    from dataclasses import replace
    cfg = WavTransformConfig(target_frames=6001, output_bits=bits,
        dither_seed=321, packet_frames=173,
        time_scale=TimeScaleSettings(window_frames=256, search_frames=32))
    a = tmp_path/"first.wav"
    b = tmp_path/"second.wav"
    first = transform_wav_file(path, a, config=cfg, expected_source_sha256=pin)
    second = transform_wav_file(path, b, config=replace(cfg, packet_frames=4096),
                                expected_source_sha256=pin, expected_output_sha256=first.output_sha256)
    assert a.read_bytes() == b.read_bytes()
    assert first.output_sha256 == second.output_sha256
    assert first.output_bits == bits and first.dither_seed == 321


def test_invalid_source_pin_does_not_create_output(source, tmp_path):
    path, _ = source
    target = tmp_path/"absent"/"result.wav"
    with pytest.raises(AudioIntegrityError):
        transform_wav_file(path, target, config=WavTransformConfig(ratio=1.5),
                           expected_source_sha256="0"*64, create_parent=True)
    assert not target.parent.exists()


def test_output_pin_failure_preserves_existing_destination(source, tmp_path):
    path, pin = source
    target = tmp_path/"result.wav"
    target.write_bytes(b"existing")
    with pytest.raises(AudioIntegrityError):
        transform_wav_file(path, target, config=WavTransformConfig(ratio=1),
                           expected_source_sha256=pin, expected_output_sha256="0"*64,
                           overwrite=True)
    assert target.read_bytes() == b"existing"
    assert not list(tmp_path.glob("*.partial"))


def test_no_clobber_and_source_aliases(source, tmp_path):
    path, pin = source
    cfg = WavTransformConfig(ratio=1)
    target = tmp_path/"existing.wav"
    target.write_bytes(b"do not replace")
    with pytest.raises(AudioContractError):
        transform_wav_file(path, target, config=cfg)
    assert target.read_bytes() == b"do not replace"
    with pytest.raises(AudioContractError, match="must differ"):
        transform_wav_file(path, path, config=cfg, overwrite=True)
    alias = tmp_path/"alias.wav"
    try:
        os.link(path, alias)
    except OSError:
        pytest.skip("hard links unsupported")
    with pytest.raises(AudioContractError, match="file identity"):
        transform_wav_file(path, alias, config=cfg, overwrite=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin


def test_memory_bound_before_publication(source, tmp_path, monkeypatch):
    import sonumio.audio_core.wav_transform as module
    path, _ = source
    monkeypatch.setattr(module, "MAX_TRANSFORM_PCM_BYTES", 1024)
    target = tmp_path/"missing"/"result.wav"
    with pytest.raises(AudioLimitError):
        transform_wav_file(path, target, config=WavTransformConfig(ratio=2),
                           create_parent=True)
    assert not target.parent.exists()


@pytest.mark.parametrize("config", [
    {}, {"ratio": 1, "semitones": 0}, {"ratio": .4}, {"target_frames": True},
    {"output_bits": 20, "ratio": 1}, {"packet_frames": 0, "ratio": 1},
    {"dither_seed": -1, "ratio": 1}, {"semitones": 15},
])
def test_invalid_config(config):
    with pytest.raises(AudioContractError):
        WavTransformConfig(**config)


def test_cli_duration_and_pitch(source, tmp_path, capsys):
    path, pin = source
    for options, filename, operation in [
        (["--ratio", "1.25"], "duration.wav", "duration"),
        (["--semitones", "-5"], "pitch.wav", "pitch"),
    ]:
        target = tmp_path/filename
        assert main(["wav-transform", str(path), "--output", str(target),
                     "--expected-sha256", pin, "--peak-ceiling", ".95", *options]) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["operation"] == operation
        assert report["source_sha256"] == pin and target.exists()


def test_cli_rejects_ambiguous_operation(source, tmp_path):
    path, _ = source
    with pytest.raises(SystemExit) as error:
        main(["wav-transform", str(path), "--output", str(tmp_path/"bad.wav"),
              "--ratio", "1", "--semitones", "0"])
    assert error.value.code == 2


def test_cli_invalid_target(source, tmp_path, capsys):
    path, _ = source
    target = tmp_path/"bad.wav"
    assert main(["wav-transform", str(path), "--output", str(target),
                 "--target-frames", "1"]) == 2
    assert "target duration" in capsys.readouterr().err.lower()
    assert not target.exists()
