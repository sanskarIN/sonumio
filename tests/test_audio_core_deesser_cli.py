"""Verified WAV de-esser CLI: reproducibility, checksum and safe failure paths."""
import hashlib
import json
import os

import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError
from sonumio.audio_core.deesser_cli import export_deessed_wav
from sonumio.audio_core.sibilance_dynamics import DeEsserSettings
from sonumio.audio_core.waveio import read_wav_file, encode_wav_bytes
from sonumio.cli import main


def wav(path, *, rate=16000):
    n = np.arange(8000)
    waveform = .06 * np.sin(2 * np.pi * 400 * n / rate)
    waveform[2000:6200] += .2 * np.sin(
        2 * np.pi * 5800 * n[2000:6200] / rate
    )
    path.write_bytes(encode_wav_bytes(make_buffer(
        waveform.astype(np.float32), rate,
    )))
    return path


def opts():
    return DeEsserSettings(
        n_fft=512, hop_frames=128,
        band_low_hz=4500., band_high_hz=7200.,
        threshold_dbfs=-35., ratio=8.,
        attack_seconds=0., release_seconds=.01,
    )


def test_atomically_verified_wav_and_identical_pinned_replay(tmp_path):
    source = wav(tmp_path / "input.wav")
    reference_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    target = tmp_path / "output.wav"
    report = export_deessed_wav(
        source, target, settings=opts(),
        expected_source_sha256=reference_sha,
    )
    assert report["protocol"] == "sonumio-verified-deesser-wav-v1"
    assert report["source"]["source_sha256"] == reference_sha
    assert report["source_modified"] is False
    assert report["model_inference"] is False
    assert report["output_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert report["output_channels"] == 1
    assert report["output_frames"] == 8000
    assert report["output_bits"] == 16
    decoded = read_wav_file(target)
    assert decoded.frames == 8000
    assert decoded.sample_rate == 16000
    replay = tmp_path / "replay.wav"
    export_deessed_wav(
        source, replay, settings=opts(),
        expected_output_sha256=report["output_sha256"],
    )
    assert replay.read_bytes() == target.read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == reference_sha


def test_pinned_output_failure_preserves_original_target(tmp_path):
    source = wav(tmp_path / "input.wav")
    target = tmp_path / "output.wav"
    target.write_bytes(b"preserve existing bytes")
    with pytest.raises(AudioIntegrityError):
        export_deessed_wav(
            source, target, settings=opts(), overwrite=True,
            expected_output_sha256="0" * 64,
        )
    assert target.read_bytes() == b"preserve existing bytes"


def test_no_clobber_and_explicit_overwrite(tmp_path):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "output.wav"
    output.write_bytes(b"old")
    with pytest.raises(AudioContractError):
        export_deessed_wav(source, output, settings=opts())
    assert output.read_bytes() == b"old"
    result = export_deessed_wav(
        source, output, settings=opts(), overwrite=True,
    )
    assert result["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_input_output_alias_and_hardlink_rejected(tmp_path):
    source = wav(tmp_path / "source.wav")
    before = source.read_bytes()
    with pytest.raises(AudioContractError):
        export_deessed_wav(source, source, settings=opts(), overwrite=True)
    assert source.read_bytes() == before
    alias = tmp_path / "alias.wav"
    try:
        os.link(source, alias)
    except OSError:
        pytest.skip("Hard links unavailable")
    with pytest.raises(AudioContractError):
        export_deessed_wav(source, alias, settings=opts(), overwrite=True)
    assert source.read_bytes() == before


def test_seeded_dither_is_repeatable(tmp_path):
    source = wav(tmp_path / "source.wav")
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    c = tmp_path / "c.wav"
    export_deessed_wav(source, a, settings=opts(), dither_seed=100)
    export_deessed_wav(source, b, settings=opts(), dither_seed=100)
    export_deessed_wav(source, c, settings=opts(), dither_seed=101)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()


def test_cli_emits_receipt_only_on_success(tmp_path, capsys):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "output.wav"
    args = [
        "wav-deess", "--source", str(source), "--output", str(output),
        "--fft-frames", "512", "--hop-frames", "128",
        "--band-low-hz", "4500", "--band-high-hz", "7200",
        "--ratio", "8", "--threshold-dbfs", "-35",
    ]
    assert main(args) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["protocol"] == "sonumio-verified-deesser-wav-v1"
    assert receipt["effect"]["affected_frames"] > 0
    assert receipt["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert main(args) == 2
    assert capsys.readouterr().out == ""


def test_cli_bad_frequency_does_not_create_output(tmp_path, capsys):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "output.wav"
    args = [
        "wav-deess", "--source", str(source), "--output", str(output),
        "--band-high-hz", "12000",
    ]
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "de-esser" in captured.err.lower()
    assert not output.exists()


def test_cli_missing_source_and_invalid_path_fail_closed(tmp_path, capsys):
    output = tmp_path / "output.wav"
    missing = tmp_path / "missing.wav"
    assert main([
        "wav-deess", "--source", str(missing), "--output", str(output),
    ]) == 2
    assert capsys.readouterr().out == ""
    assert not output.exists()


def test_cli_explicit_parent_creation(tmp_path, capsys):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "sub" / "result.wav"
    assert main([
        "wav-deess", "--source", str(source), "--output", str(output),
        "--create-parent",
    ]) == 0
    assert output.exists()
    json.loads(capsys.readouterr().out)


def test_pinned_source_sha_failure_never_creates_output(tmp_path):
    source = wav(tmp_path / "source.wav")
    output = tmp_path / "output.wav"
    with pytest.raises(AudioIntegrityError):
        export_deessed_wav(
            source, output, expected_source_sha256="0" * 64,
            settings=opts(),
        )
    assert not output.exists()
