"""Regression tests for bounded experimental model checkpoint imports."""
import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.checkpoint_archive import validate_numeric_archive
from research.text_audio import (
    TextWaveDenoiser, save_checkpoint as save_text_checkpoint,
    load_checkpoint as load_text_checkpoint,
)
from research.wave_diffusion import (
    TinyWaveDenoiser, DiffusionSchedule,
    save_checkpoint as save_wave_checkpoint,
    load_checkpoint as load_wave_checkpoint,
)


@pytest.fixture(params=("wave", "text"))
def case(tmp_path, request):
    if request.param == "wave":
        save_wave_checkpoint(tmp_path, TinyWaveDenoiser(channels=16),
                             DiffusionSchedule(8), {"source": "test"})
        return tmp_path, "weights.npz", "manifest.json", load_wave_checkpoint
    save_text_checkpoint(tmp_path, TextWaveDenoiser(channels=16),
                         DiffusionSchedule(8), {"source": "test"})
    return tmp_path, "text_weights.npz", "text_manifest.json", load_text_checkpoint


def refresh_digest(folder, archive_name, manifest_name):
    archive_path = folder / archive_name
    manifest_path = folder / manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["weights_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_unmodified_checkpoint_import_succeeds(case):
    folder, archive_name, _, load = case
    model, schedule, _ = load(folder)
    assert schedule.steps == 8
    validate_numeric_archive(folder / archive_name, model.state_dict())


@pytest.mark.parametrize("mutation,message", (
    ("oversize", "exceeds expected byte budget"),
    ("shape", "Invalid checkpoint tensor header"),
    ("extra", "Unexpected checkpoint tensor names"),
))
def test_reject_tampered_archive_even_with_matching_sha256(case, mutation, message):
    folder, archive_name, manifest_name, load = case
    path = folder / archive_name
    if mutation == "extra":
        with ZipFile(path, mode="a") as archive:
            archive.writestr("unknown.npy", b"not a tensor")
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        first = next(iter(arrays))
        if mutation == "oversize":
            arrays[first] = np.zeros((arrays[first].size + 5000,), dtype=np.float32)
        else:
            arrays[first] = arrays[first].reshape(-1)
            assert arrays[first].shape != ()
        np.savez_compressed(path, **arrays)
    refresh_digest(folder, archive_name, manifest_name)
    with pytest.raises(ValueError, match=message):
        load(folder)


def test_wave_checkpoint_rejects_absurd_dimensions_before_allocating(case):
    folder, _, manifest_name, load = case
    if manifest_name != "manifest.json":
        return
    manifest_path = folder / manifest_name
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata["channels"] = 8_000_000
    manifest_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported waveform diffusion checkpoint configuration"):
        load(folder)
