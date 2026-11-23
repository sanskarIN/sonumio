"""Atomic local WAV publication, digest checks, and real CLI failure recovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

from sonumio import atomic_cli_wav as publishing
from sonumio.audio import encode_wav
from sonumio.audio_core.errors import (
    AudioFormatError, AudioIntegrityError, AudioLimitError,
)
from sonumio.cli import main
from sonumio.service import generate_audio


@pytest.fixture
def audio() -> bytes:
    samples = (np.sin(np.arange(2400, dtype=np.float64) / 13) * 0.1).astype(
        np.float32
    )
    return encode_wav(samples, 24000)


def test_wav_publication_has_exact_full_sha_and_valid_audio(tmp_path, audio):
    path = tmp_path / "new" / "render.wav"
    expected = hashlib.sha256(audio).hexdigest()
    result = publishing.publish_generated_wav(
        path, audio, expected_sha256=expected,
    )
    assert result.path == path
    assert result.bytes_written == len(audio)
    assert result.wav_sha256 == expected
    assert path.read_bytes() == audio
    assert len(list(path.parent.iterdir())) == 1
    assert not list(path.parent.glob(".sonumio-cli-*.partial"))


def test_existing_output_never_clobbered_without_opt_in(tmp_path, audio):
    existing = tmp_path / "important.wav"
    existing.write_bytes(b"original personal recording")
    with pytest.raises(AudioFormatError, match="--overwrite"):
        publishing.publish_generated_wav(existing, audio)
    assert existing.read_bytes() == b"original personal recording"
    assert not list(tmp_path.glob(".sonumio-cli-*.partial"))


def test_overwrite_replaces_contents_atomically(tmp_path, audio):
    path = tmp_path / "previous.wav"
    path.write_bytes(b"original")
    result = publishing.publish_generated_wav(path, audio, overwrite=True)
    assert path.read_bytes() == audio
    assert result.wav_sha256 == hashlib.sha256(audio).hexdigest()
    assert not list(tmp_path.glob(".sonumio-cli-*.partial"))


def test_hardlink_source_file_is_not_mutated_on_replacement(tmp_path, audio):
    original = tmp_path / "source.wav"
    second_name = tmp_path / "copy.wav"
    original.write_bytes(b"old linked bytes")
    try:
        os.link(original, second_name)
    except OSError:
        pytest.skip("filesystem does not support hard links")
    publishing.publish_generated_wav(second_name, audio, overwrite=True)
    assert original.read_bytes() == b"old linked bytes"
    assert second_name.read_bytes() == audio


def test_racing_other_writer_cannot_be_overwritten(tmp_path, audio, monkeypatch):
    target = tmp_path / "competing.wav"
    original_link = publishing.os.link
    seen = []

    def competitor(src, dst, *args, **kwargs):
        if Path(dst) == target:
            target.write_bytes(b"concurrent result")
            seen.append(True)
            raise FileExistsError("Competing writer won the race")
        return original_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(publishing.os, "link", competitor)
    with pytest.raises(AudioFormatError, match="--overwrite"):
        publishing.publish_generated_wav(target, audio)
    assert seen
    assert target.read_bytes() == b"concurrent result"
    assert not list(tmp_path.glob(".sonumio-cli-*.partial"))


def test_overwrite_replace_failure_preserves_old_file(tmp_path, audio, monkeypatch):
    dest = tmp_path / "existing.wav"
    dest.write_bytes(b"keep this")
    original = publishing.os.replace

    def failed_replace(src, dst):
        if Path(dst) == dest:
            raise OSError("simulated destination filesystem failure")
        return original(src, dst)

    monkeypatch.setattr(publishing.os, "replace", failed_replace)
    with pytest.raises(OSError, match="simulated"):
        publishing.publish_generated_wav(dest, audio, overwrite=True)
    assert dest.read_bytes() == b"keep this"
    assert not list(tmp_path.glob(".sonumio-cli-*.partial"))


def test_fsync_failure_preserves_original_without_partial(tmp_path, audio, monkeypatch):
    dest = tmp_path / "previous.wav"
    dest.write_bytes(b"original")
    def failed_fsync(fd):
        raise OSError("simulated fsync failure")
    monkeypatch.setattr(publishing.os, "fsync", failed_fsync)
    with pytest.raises(OSError, match="fsync"):
        publishing.publish_generated_wav(dest, audio, overwrite=True)
    assert dest.read_bytes() == b"original"
    assert not list(tmp_path.glob(".sonumio-cli-*.partial"))


def test_checksum_rejection_does_not_create_destination_or_parent(tmp_path, audio):
    target = tmp_path / "not-created" / "output.wav"
    with pytest.raises(AudioIntegrityError, match="SHA256 mismatch"):
        publishing.publish_generated_wav(
            target, audio, expected_sha256="0" * 64,
        )
    assert not target.exists() and not target.parent.exists()


@pytest.mark.parametrize("bad_sha", [
    "", "0" * 63, "F" * 64, "x" * 64, True, 123,
])
def test_bad_expected_digest_rejected_before_mkdir(tmp_path, audio, bad_sha):
    path = tmp_path / "new" / "render.wav"
    with pytest.raises(ValueError):
        publishing.publish_generated_wav(path, audio, expected_sha256=bad_sha)
    assert not path.parent.exists()


@pytest.mark.parametrize("corruption", [
    lambda b: b"JUNK" + b[4:],
    lambda b: b[:8] + b"NOPE" + b[12:],
    lambda b: b[:12] + b"test" + b[16:],
    lambda b: b[:40] + b"JUNK" + b[44:],
    lambda b: b[:4] + struct.pack("<I", 100) + b[8:],
    lambda b: b[:40] + struct.pack("<I", 0) + b[44:],
    lambda b: b[:20] + struct.pack("<H", 3) + b[22:],
    lambda b: b[:-1],
    lambda b: b + b"x",
])
def test_malformed_or_incomplete_wav_does_not_publish(tmp_path, audio, corruption):
    path = tmp_path / "absent" / "not.wav"
    with pytest.raises((AudioFormatError, AudioLimitError, ValueError)):
        publishing.publish_generated_wav(path, corruption(audio))
    assert not path.parent.exists()


def test_wrong_wav_data_type_never_publishes(tmp_path):
    path = tmp_path / "missing" / "output.wav"
    for invalid in (None, "RIFF", bytearray(44), memoryview(b"WAVE"), b""):
        with pytest.raises((AudioFormatError, AudioLimitError)):
            publishing.publish_generated_wav(path, invalid)
    assert not path.parent.exists()


def test_staged_file_size_is_bounded_before_directory_creation(
    tmp_path, audio, monkeypatch,
):
    monkeypatch.setattr(
        publishing, "MAX_GENERATED_WAV_BYTES", len(audio) - 1,
    )
    target = tmp_path / "not-created" / "recording.wav"
    with pytest.raises(AudioLimitError):
        publishing.publish_generated_wav(target, audio)
    assert not target.parent.exists()


@pytest.mark.parametrize("invalid", [
    "wav.txt", "wav.bin", "wav", "example.WAVE",
])
def test_only_wav_destinations_allowed_before_mkdir(tmp_path, audio, invalid):
    target = tmp_path / "never" / invalid
    with pytest.raises(AudioFormatError, match=".wav"):
        publishing.publish_generated_wav(target, audio)
    assert not target.parent.exists()


@pytest.mark.parametrize("overrides", [
    {"overwrite": 1}, {"overwrite": None},
    {"create_parent": 0}, {"create_parent": None},
])
def test_non_boolean_publication_controls_rejected(tmp_path, audio, overrides):
    with pytest.raises(ValueError, match="boolean"):
        publishing.publish_generated_wav(
            tmp_path / "clip.wav", audio, **overrides,
        )
    assert not (tmp_path / "clip.wav").exists()


def test_no_parent_directory_when_creation_disabled(tmp_path, audio):
    dest = tmp_path / "missing" / "file.wav"
    with pytest.raises(AudioFormatError, match="parent"):
        publishing.publish_generated_wav(
            dest, audio, create_parent=False,
        )
    assert not dest.parent.exists()


def test_leaf_symlink_cannot_be_overwritten_even_with_opt_in(
    tmp_path, audio,
):
    protected = tmp_path / "private.txt"
    protected.write_bytes(b"external")
    link = tmp_path / "preview.wav"
    try:
        link.symlink_to(protected)
    except (NotImplementedError, OSError):
        pytest.skip("symlink unavailable on this platform")
    with pytest.raises(AudioFormatError, match="ordinary"):
        publishing.publish_generated_wav(link, audio, overwrite=True)
    assert link.is_symlink()
    assert protected.read_bytes() == b"external"
    assert not list(tmp_path.glob(".sonumio-cli-*.partial"))


def test_symlinked_parent_refused_before_directory_access(
    tmp_path, audio,
):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlink unavailable on this platform")
    with pytest.raises(AudioFormatError, match="symlink"):
        publishing.publish_generated_wav(
            link / "preview.wav", audio,
        )
    assert not (outside / "preview.wav").exists()


def test_existing_directory_with_wav_suffix_is_not_deleted(tmp_path, audio):
    target = tmp_path / "folder.wav"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"leave alone")
    with pytest.raises(AudioFormatError, match="ordinary"):
        publishing.publish_generated_wav(target, audio, overwrite=True)
    assert sentinel.read_bytes() == b"leave alone"


def test_full_cli_generate_requires_explicit_opt_in_to_replace(tmp_path, capsys):
    dest = tmp_path / "new" / "sonumio.wav"
    args = [
        "generate", "rain", "--seconds", "0.5",
        "--seed", "7", "--output", str(dest),
    ]
    assert main(args) == 0
    first = dest.read_bytes()
    out = capsys.readouterr()
    assert "SHA256" in out.out
    assert hashlib.sha256(first).hexdigest() in out.out
    assert main(args) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert dest.read_bytes() == first
    assert main(args + ["--overwrite"]) == 0
    assert dest.read_bytes() == first


def test_cli_generation_expected_digest_and_no_parent_when_wrong(
    tmp_path, capsys,
):
    dest = tmp_path / "not-created" / "sound.wav"
    args = [
        "generate", "wind", "--seconds", "0.5",
        "--output", str(dest),
        "--expected-output-sha256", "0" * 64,
    ]
    assert main(args) == 2
    assert "SHA256" in capsys.readouterr().err
    assert not dest.parent.exists()
    exact = hashlib.sha256(
        generate_audio("wind", 0.5).wav_bytes,
    ).hexdigest()
    assert main(args[:-1] + [exact]) == 0
    assert dest.exists()


def test_cli_scene_publishes_once_and_protects_existing_output(
    tmp_path, capsys,
):
    scene = tmp_path / "instructions.json"
    target = tmp_path / "output.wav"
    scene.write_text(json.dumps({
        "duration_seconds": 1.25,
        "layers": [
            {
                "prompt": "ocean", "duration_seconds": 0.5,
                "start_seconds": 0.25,
            }
        ],
    }), encoding="utf-8")
    opts = ["scene", str(scene), "--output", str(target)]
    assert main(opts) == 0
    original = target.read_bytes()
    assert original.startswith(b"RIFF")
    assert main(opts) == 2
    assert "--overwrite" in capsys.readouterr().err
    assert target.read_bytes() == original
    assert main(opts + ["--overwrite"]) == 0
    assert target.read_bytes() == original


def test_cli_rejects_existing_unrelated_file_and_does_not_erase(
    tmp_path, capsys,
):
    dest = tmp_path / "school-project.wav"
    dest.write_bytes(b"unrelated user data")
    status = main([
        "generate", "rain", "--seconds", "0.5",
        "--output", str(dest),
    ])
    assert status == 2
    assert dest.read_bytes() == b"unrelated user data"
    assert "--overwrite" in capsys.readouterr().err


def test_cli_source_file_not_affected_by_staging_failure(
    tmp_path, capsys,
):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original user data")
    status = main([
        "generate", "rain", "--seconds", "0.5",
        "--output", str(source),
        "--expected-output-sha256", "0" * 64,
        "--overwrite",
    ])
    assert status == 2
    assert source.read_bytes() == b"original user data"
    assert "SHA256" in capsys.readouterr().err
