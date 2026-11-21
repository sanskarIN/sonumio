"""Safety and retention regression tests for local API audio artifacts."""
from concurrent.futures import ThreadPoolExecutor
import io
import os
import wave

import pytest

from sonumio.output_store import LocalAudioStore, OutputStoreError, OutputTooLargeError


def make_wav(frames=4):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\x01\x02" * frames)
    return stream.getvalue()


def test_atomic_round_trip_and_directories(tmp_path):
    store = LocalAudioStore(tmp_path / "api-output")
    expected = make_wav()
    identifier = store.write(expected)
    assert store.valid_id(identifier)
    assert store.read(identifier) == expected
    assert store.usage().files == 1
    assert store.usage().bytes == len(expected)
    assert not list(store.root.glob("*.tmp"))
    assert store.delete(identifier) is True
    assert store.delete(identifier) is False
    with pytest.raises(FileNotFoundError):
        store.read(identifier)


def test_strict_identifiers_and_foreign_files_untouched(tmp_path):
    root = tmp_path / "api"
    store = LocalAudioStore(root)
    identifier = store.write(make_wav())
    foreign = root / "other.wav"
    foreign.write_bytes(b"not a managed export")
    outside = tmp_path / f"{identifier}.wav"
    outside.write_bytes(b"outside")
    assert store.delete("../outside") is False
    with pytest.raises(FileNotFoundError):
        store.read("../outside")
    assert store.delete(identifier)
    assert outside.read_bytes() == b"outside"
    assert foreign.read_bytes() == b"not a managed export"


def test_ttl_expires_outputs_even_after_restart(tmp_path):
    clock = [1000.0]
    root = tmp_path / "managed"
    a = LocalAudioStore(root, ttl_seconds=30, clock=lambda: clock[0])
    identifier = a.write(make_wav())
    clock[0] += 29
    b = LocalAudioStore(root, ttl_seconds=30, clock=lambda: clock[0])
    assert b.read(identifier) == make_wav()
    clock[0] += 1
    with pytest.raises(FileNotFoundError):
        b.read(identifier)
    assert not list(root.glob("*.wav"))


def test_quota_evicts_oldest_and_keeps_newest(tmp_path):
    clock = [1000.0]
    store = LocalAudioStore(tmp_path / "managed", max_files=2, clock=lambda: clock[0])
    a = store.write(make_wav(2))
    clock[0] += 1
    b = store.write(make_wav(3))
    clock[0] += 1
    c = store.write(make_wav(4))
    with pytest.raises(FileNotFoundError):
        store.read(a)
    assert store.read(b) == make_wav(3)
    assert store.read(c) == make_wav(4)
    assert store.usage().files == 2


def test_total_byte_budget_evicts_oldest(tmp_path):
    clock = [1000.0]
    store = LocalAudioStore(tmp_path / "managed", max_total_bytes=120,
                            max_file_bytes=100, clock=lambda: clock[0])
    a = store.write(make_wav(8))
    clock[0] += 1
    b = store.write(make_wav(12))
    with pytest.raises(FileNotFoundError):
        store.read(a)
    assert store.read(b) == make_wav(12)


def test_reject_oversize_or_malformed_audio_without_writing(tmp_path):
    store = LocalAudioStore(tmp_path / "managed", max_total_bytes=100, max_file_bytes=60)
    with pytest.raises(OutputTooLargeError):
        store.write(make_wav(12))
    with pytest.raises(ValueError):
        store.write(b"not a wave")
    assert store.usage().files == 0


def test_no_symlink_root_or_followed_symlink_artifact(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(actual, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("Symlinks not supported on this filesystem")
    with pytest.raises(OutputStoreError):
        LocalAudioStore(alias).write(make_wav())
    store = LocalAudioStore(actual)
    identifier = store.write(make_wav())
    outside = tmp_path / "external.wav"
    outside.write_bytes(make_wav(10))
    (actual / f"{identifier}.wav").unlink()
    (actual / f"{identifier}.wav").symlink_to(outside)
    with pytest.raises(FileNotFoundError):
        store.read(identifier)
    assert store.delete(identifier) is False
    assert outside.exists()


def test_stale_temp_cleanup_only_removes_store_temp_files(tmp_path):
    clock = [3000.0]
    store = LocalAudioStore(tmp_path / "managed", ttl_seconds=60, clock=lambda: clock[0])
    store.root.mkdir()
    scratch = store.root / ".sonumio-dead.tmp"
    scratch.write_bytes(b"partial")
    os.utime(scratch, (2000, 2000))
    unrelated = store.root / "other.tmp"
    unrelated.write_bytes(b"private")
    store.usage()
    assert not scratch.exists()
    assert unrelated.read_bytes() == b"private"


def test_concurrent_writes_generate_unique_complete_files(tmp_path):
    store = LocalAudioStore(tmp_path / "managed", max_files=50)
    expected = make_wav(9)
    with ThreadPoolExecutor(max_workers=10) as pool:
        identifiers = list(pool.map(store.write, [expected] * 32))
    assert len(set(identifiers)) == len(identifiers)
    assert all(store.read(item) == expected for item in identifiers)
    assert store.usage().files == 32


def test_parameter_validation(tmp_path):
    for kwargs in ({"max_files": 0}, {"max_total_bytes": 20},
                   {"max_file_bytes": 10}, {"max_total_bytes": 44, "max_file_bytes": 45},
                   {"ttl_seconds": 0}, {"ttl_seconds": float("nan")}):
        with pytest.raises(ValueError):
            LocalAudioStore(tmp_path, **kwargs)
