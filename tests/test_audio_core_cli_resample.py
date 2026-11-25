"""End-to-end CLI coverage for verified local WAV sample-rate conversion."""
import hashlib
import json

import numpy as np
import pytest

from sonumio.cli import main
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.waveio import write_wav_file, read_wav_file


def make_source(path):
    data = .25 * np.sin(2*np.pi*440*np.arange(1000)/24000)
    write_wav_file(make_buffer(data.astype(np.float32), 24000), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_conversion_and_json_contract(tmp_path, capsys):
    source = tmp_path / "source.wav"
    dest = tmp_path / "nested" / "out.wav"
    source_hash = make_source(source)
    assert main(["wav-resample", str(source), "--output", str(dest),
                 "--target-rate", "16000", "--packet-frames", "31",
                 "--expected-sha256", source_hash, "--output-bits", "24",
                 "--create-parent"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result["source_rate"] == 24000
    assert result["target_rate"] == 16000
    assert result["source_frames"] == 1000
    assert result["output_frames"] == 667
    assert result["source_sha256"] == source_hash
    assert result["output_sha256"] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert result["output_bits"] == 24
    assert result["neural_training_data_modified"] is False
    assert read_wav_file(dest).sample_rate == 16000


@pytest.mark.parametrize("extra", [
    ["--target-rate", "7999"],
    ["--target-rate", "16000", "--packet-frames", "8193"],
    ["--target-rate", "16000", "--lobes", "0"],
    ["--target-rate", "16000", "--expected-sha256", "0"*64],
    ["--target-rate", "16000", "--expected-output-sha256", "0"*64],
])
def test_cli_failure_has_no_json_or_published_output(tmp_path, capsys, extra):
    source = tmp_path / "source.wav"
    dest = tmp_path / "out.wav"
    make_source(source)
    assert main(["wav-resample", str(source), "--output", str(dest), *extra]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Sonumio WAV resampling" in captured.err
    assert not dest.exists()
    assert not list(tmp_path.glob(".sonumio-stream-*"))


def test_cli_dither_and_hash_reproducibility(tmp_path, capsys):
    source = tmp_path / "source.wav"
    make_source(source)
    options = ["wav-resample", str(source), "--target-rate", "44100",
               "--dither-seed", "7", "--output-bits", "16"]
    first, second = tmp_path / "a.wav", tmp_path / "b.wav"
    assert main(options + ["--output", str(first), "--packet-frames", "19"]) == 0
    first_report = json.loads(capsys.readouterr().out)
    assert main(options + ["--output", str(second), "--packet-frames", "500",
                            "--expected-output-sha256", first_report["output_sha256"]]) == 0
    assert json.loads(capsys.readouterr().out)["output_sha256"] == first_report["output_sha256"]
    assert first.read_bytes() == second.read_bytes()
