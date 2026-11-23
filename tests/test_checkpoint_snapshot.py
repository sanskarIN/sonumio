"""Hostile-file and immutable-archive tests for private neural checkpoints."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import numpy as np
import pytest

from research import checkpoint_snapshot as snapshots


def _write(path: Path, payload: bytes = b"owned numeric checkpoint bytes") -> str:
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_verified_snapshot_and_bad_digest(tmp_path):
    path = tmp_path / "weight.npz"
    sha = _write(path)
    snapshot = snapshots.read_verified_weights(path, sha)
    assert snapshot == path.read_bytes()
    assert snapshots.verified_npz_bytes(snapshot).read() == snapshot
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        snapshots.read_verified_weights(path, "0" * 64)
    with pytest.raises(ValueError, match="SHA256 declaration"):
        snapshots.read_verified_weights(path, "not-a-digest")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None, snapshots.MAX_WEIGHTS_BYTES + 1])
def test_invalid_limits_fail_closed(tmp_path, value):
    path = tmp_path / "weight.npz"
    _write(path)
    with pytest.raises(ValueError, match="size limit"):
        snapshots.read_regular_snapshot(path, max_bytes=value)


def test_empty_oversized_and_directory_are_rejected(tmp_path):
    empty = tmp_path / "empty.npz"
    empty.touch()
    too_large = tmp_path / "large.npz"
    too_large.write_bytes(b"abcdefghijk")
    for path in (empty, too_large, tmp_path):
        with pytest.raises(ValueError, match="Unsafe or oversized"):
            snapshots.read_regular_snapshot(path, max_bytes=10)


def test_hardlink_and_symlink_rejected(tmp_path):
    path = tmp_path / "weight.npz"
    sha = _write(path)
    alias = tmp_path / "alias.npz"
    try:
        os.link(path, alias)
    except (OSError, NotImplementedError):
        pytest.skip("Hardlinks unsupported")
    with pytest.raises(ValueError, match="Unsafe"):
        snapshots.read_verified_weights(path, sha)
    alias.unlink()
    try:
        alias.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks unsupported")
    with pytest.raises(ValueError, match="Unsafe"):
        snapshots.read_verified_weights(alias, sha)
    folder = tmp_path / "alias-folder"
    folder.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="parent directory"):
        snapshots.read_verified_weights(folder / path.name, sha)


@pytest.mark.skipif(not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"),
                    reason="Requires nonblocking FIFO support")
def test_fifo_rejected_without_hanging(tmp_path):
    pipe = tmp_path / "weight.npz"
    os.mkfifo(pipe)
    with pytest.raises(ValueError, match="Unsafe"):
        snapshots.read_regular_snapshot(pipe, max_bytes=100)


def test_file_swap_before_descriptor_open_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "weight.npz"
    _write(path)
    original_open = os.open
    changed = False

    def swapped_open(filename, flags, *args, **kwargs):
        nonlocal changed
        if Path(filename) == path and not changed:
            changed = True
            path.unlink()
            path.write_bytes(b"untrusted replacement")
        return original_open(filename, flags, *args, **kwargs)

    monkeypatch.setattr(snapshots.os, "open", swapped_open)
    with pytest.raises(ValueError, match="changed before reading"):
        snapshots.read_regular_snapshot(path, max_bytes=1000)


def test_file_mutation_after_first_fstat_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "weight.npz"
    _write(path)
    original_fstat = os.fstat
    count = 0

    def mutate_on_second_fstat(fd):
        nonlocal count
        count += 1
        if count == 2:
            path.write_bytes(b"different content same length")
        return original_fstat(fd)

    monkeypatch.setattr(snapshots.os, "fstat", mutate_on_second_fstat)
    with pytest.raises(ValueError, match="changed during reading"):
        snapshots.read_regular_snapshot(path, max_bytes=1000)


def test_manifest_strict_json_and_duplicate_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"format":"v1","report":{"steps":2}}', encoding="utf-8")
    assert snapshots.read_checkpoint_manifest(path)["report"]["steps"] == 2
    for bad in ('{"a":1,"a":2}', '{"a":{"x":1,"x":2}}',
                '{"loss":NaN}', '{"loss":Infinity}',
                '["not an object"]', 'not json', '{'):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            snapshots.read_checkpoint_manifest(path)


def test_oversized_manifest_rejected_before_json_parsing(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_bytes(b" " * (snapshots.MAX_MANIFEST_BYTES + 1))
    with pytest.raises(ValueError, match="Unsafe or oversized"):
        snapshots.read_checkpoint_manifest(path)


def test_numeric_snapshot_is_independent_from_subsequent_file_changes(tmp_path):
    from research.checkpoint_archive import validate_numeric_archive
    path = tmp_path / "weight.npz"
    np.savez_compressed(path, test=np.array([1, 2, 3], dtype=np.float32))
    class ExpectedTensor:
        shape = (3,)

        def numel(self):
            return 3

        def element_size(self):
            return 4

    expected = {"test": ExpectedTensor()}
    snapshot = snapshots.read_verified_weights(path, hashlib.sha256(path.read_bytes()).hexdigest())
    np.savez_compressed(path, test=np.array([9, 9, 9], dtype=np.float32))
    validate_numeric_archive(snapshot, expected)
    with np.load(snapshots.verified_npz_bytes(snapshot), allow_pickle=False) as archive:
        assert archive["test"].tolist() == [1, 2, 3]


@pytest.mark.parametrize("family", ["wave", "text", "latent"])
def test_all_neural_experiments_load_original_checked_bytes_after_disk_swap(
    tmp_path, monkeypatch, family,
):
    torch = pytest.importorskip("torch")
    from research.wave_diffusion import TinyWaveDenoiser, DiffusionSchedule
    if family == "wave":
        from research import wave_diffusion as module
        model = TinyWaveDenoiser(16)
        save, filename = module.save_checkpoint, "weights.npz"
    elif family == "text":
        from research import text_audio as module
        model = module.TextWaveDenoiser(channels=16)
        save, filename = module.save_checkpoint, "text_weights.npz"
    else:
        from research import latent_audio as module
        model = module.LatentResearchModel(16)
        save, filename = module._save_checkpoint, "latent_weights.npz"
    save(tmp_path, model, DiffusionSchedule(8), {"source": "synthetic"})
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    actual_validator = module.validate_numeric_archive
    calls = []

    def replace_checkpoint_after_digest(verified_snapshot, expected):
        assert isinstance(verified_snapshot, bytes)
        calls.append(True)
        (tmp_path / filename).write_bytes(b"tampered after digest")
        return actual_validator(verified_snapshot, expected)

    monkeypatch.setattr(module, "validate_numeric_archive", replace_checkpoint_after_digest)
    loaded, schedule, manifest = module.load_checkpoint(tmp_path)
    assert calls and schedule.steps == 8
    assert manifest["weights_publication_authorized"] is False
    assert all(torch.equal(state[name], value) for name, value in loaded.state_dict().items())
