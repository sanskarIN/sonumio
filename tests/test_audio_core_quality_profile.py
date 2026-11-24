"""Small subprocess-level reproducibility and bound checks for WAV profiling."""
from __future__ import annotations

import json

import pytest

from benchmarks.wav_quality_profile import (
    PROTOCOL, _validate, _child, _make_source,
    main, profile_quality_scan,
)


@pytest.mark.parametrize("seconds", [-1, 0, 601, True])
def test_profile_rejects_out_of_budget_durations(seconds):
    with pytest.raises(ValueError):
        _validate(seconds, (1024,), False)


@pytest.mark.parametrize("packet_sizes", [
    (), (True,), (0,), (65537,), (1, 1),
    (1, 2, 3, 4, 5, 6), (1.5,),
])
def test_profile_rejects_invalid_packet_geometry(packet_sizes):
    with pytest.raises(ValueError):
        _validate(1, packet_sizes, False)


def test_profile_rejects_non_boolean_timeline():
    with pytest.raises(ValueError):
        _validate(1, (512,), 1)


def test_tiny_verified_source_is_not_published(tmp_path):
    source = tmp_path / "tiny.wav"
    digest = _make_source(source, 1)
    report = _child(source, digest, 257, True)
    assert report["protocol"] == PROTOCOL
    assert report["frames"] == 8000
    assert report["source_sha256"] == digest
    assert report["source_verified"]
    assert report["timeline_windows"] == 1
    assert report["scan_wall_seconds"] > 0
    assert report["real_time_ratio"] > 0
    assert not report["model_weights_loaded"]


def test_isolated_packet_variants_agree_and_clean_up():
    report = profile_quality_scan(
        seconds=1, packet_frames=(257, 4096), timeline=True,
    )
    assert report["protocol"] == PROTOCOL
    assert report["source_frames"] == 8000
    assert len(report["packet_variants"]) == 2
    assert report["temporary_source_removed"] is True
    assert report["published_audio_artifacts"] is False
    assert report["human_listening_performed"] is False
    sha = report["source_sha256"]
    assert len(sha) == 64
    assert {row["source_sha256"] for row in report["packet_variants"]} == {sha}
    assert {row["timeline_windows"] for row in report["packet_variants"]} == {1}


def test_profile_cli_invalid_controls_return_nonzero_without_json(capsys):
    assert main(["--seconds", "0"]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    assert "seconds" in out.err


def test_profile_cli_emits_one_json_result(capsys):
    assert main(["--seconds", "1", "--packet-frames", "8192"]) == 0
    out = capsys.readouterr()
    payload = json.loads(out.out)
    assert payload["source_frames"] == 8000
    assert payload["source_bytes"] > 44
    assert payload["packet_variants"][0]["packet_frames"] == 8192
    assert payload["packet_variants"][0]["timeline_windows"] is None



def test_profile_preflight_rejects_excessive_packet_iteration_work():
    with pytest.raises(ValueError, match="packet count"):
        _validate(600, (1,), False)
    _validate(1, (1,), False)
