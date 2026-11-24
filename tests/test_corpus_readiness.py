import hashlib
import json
import wave
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.audio_dataset import ClipProvenance, LoadedSplit, load_reviewed_splits
from research.corpus_readiness import (
    GroupBalancedSampler, ReadinessPolicy, audit_loaded_splits, clip_quality, main,
)
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES


def synthetic_splits():
    rng = np.random.default_rng(17)
    result = {}
    line = 0
    for name, groups in (("train", 2), ("validation", 1), ("test", 0)):
        waves, labels, captions, provenance = [], [], [], []
        for label, category in enumerate(CATEGORIES):
            for group in range(groups):
                # Unequal numbers of crops per asset test group-level balancing.
                line += 1
                for crop in range(1, (5 if group == 0 else 1) + 1):
                    waves.append(rng.normal(0, 0.08, CLIP_FRAMES).astype(np.float32))
                    labels.append(label)
                    captions.append(category)
                    provenance.append(ClipProvenance(
                        f"{name}-asset-{label}-{group}", f"{name}-group-{label}-{group}",
                        "en", line, crop,
                    ))
        result[name] = LoadedSplit(
            np.asarray(waves, dtype=np.float32).reshape(-1, 1, CLIP_FRAMES),
            np.asarray(labels, dtype=np.int64), len({p.asset_id for p in provenance}),
            hashlib.sha256(name.encode()).hexdigest(), tuple(captions), tuple(provenance), name,
        )
    return result


def test_quality_distinguishes_dc_silence_and_clipping():
    policy = ReadinessPolicy()
    assert set(clip_quality(np.full(CLIP_FRAMES, 0.2, np.float32), policy)["reasons"]) == {
        "low_centered_rms", "low_activity", "dc_offset",
    }
    assert clip_quality(np.zeros(CLIP_FRAMES, np.float32), policy)["centered_rms_dbfs"] is None
    clipped = np.tile(np.array([-1, 1], dtype=np.float32), CLIP_FRAMES // 2)
    assert clip_quality(clipped, policy)["reasons"] == ["near_clipping"]


def test_ready_coverage_uses_groups_not_clip_counts():
    splits = synthetic_splits()
    result = audit_loaded_splits(splits)
    assert result.report["readiness_gate_passed"]
    rain = result.report["splits"]["train"]["by_category"]["rain"]
    assert rain["clips"] == 6
    assert rain["eligible_recording_groups"] == 2
    assert rain["eligible_clips_by_language"] == {"en": 6, "hi": 0}
    assert result.report["splits"]["test"]["eligible_clips"] == 0
    assert not result.report["rights_and_voice_claims_independently_verified"]
    assert len(result.eligible_indices["train"]) == 36
    text = json.dumps(result.report, allow_nan=False)
    assert "train-asset" not in text and "train-group" not in text


def test_duplicate_crop_and_silence_reduce_group_coverage():
    splits = synthetic_splits()
    train = splits["train"]
    train.clips[5] = train.clips[0]
    train.clips[11] = 0
    result = audit_loaded_splits(splits)
    assert not result.report["readiness_gate_passed"]
    reasons = result.report["splits"]["train"]["rejection_reason_counts"]
    assert reasons["repeated_exact_crop"] == 1
    assert reasons["low_centered_rms"] == 1
    assert result.report["splits"]["train"]["by_category"]["rain"]["eligible_recording_groups"] == 1
    with pytest.raises(ValueError, match="failed readiness"):
        GroupBalancedSampler(splits, result, seed=1)


def test_sampler_balanced_and_chunk_invariant():
    splits = synthetic_splits()
    readiness = audit_loaded_splits(splits)
    a = GroupBalancedSampler(splits, readiness, seed=42)
    b = GroupBalancedSampler(splits, readiness, seed=42)
    first = np.concatenate([a.draw(7), a.draw(5993)])
    second = b.draw(6000)
    np.testing.assert_array_equal(first, second)
    assert a.describe() == b.describe()
    train = splits["train"]
    labels = train.labels[first]
    for start in range(0, len(labels), 6):
        assert sorted(labels[start:start+6].tolist()) == list(range(6))
    rain_indices = first[labels == 0]
    smaller_group = sum(train.provenance[i].group_id.endswith("-1") for i in rain_indices)
    assert 400 <= smaller_group <= 600  # group choice, not crop-frequency weighting
    assert set(first.tolist()).issubset(readiness.eligible_indices["train"])


def test_sampler_detects_changed_audited_audio():
    splits = synthetic_splits()
    readiness = audit_loaded_splits(splits)
    splits["train"].clips[0, 0, 0] += 0.01
    with pytest.raises(ValueError, match="changed since audit"):
        GroupBalancedSampler(splits, readiness, seed=1)


def test_cross_split_crop_rejected():
    splits = synthetic_splits()
    splits["validation"].clips[0] = splits["train"].clips[0]
    with pytest.raises(ValueError, match="Exact audio crop"):
        audit_loaded_splits(splits)


def test_cross_split_group_rejected():
    splits = synthetic_splits()
    val = splits["validation"]
    rows = list(val.provenance)
    rows = [replace(p, group_id=splits["train"].provenance[0].group_id)
            if p.asset_id == rows[0].asset_id else p for p in rows]
    splits["validation"] = replace(val, provenance=tuple(rows))
    with pytest.raises(ValueError, match="crosses split"):
        audit_loaded_splits(splits)


@pytest.mark.parametrize("kwargs", [
    {"min_train_groups_per_category": True}, {"min_validation_groups_per_category": 0},
    {"min_centered_rms_dbfs": float("nan")}, {"max_absolute_dc": 2},
])
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        ReadinessPolicy(**kwargs)


def test_actual_wav_ingestion_retains_provenance_and_cli_coverage(tmp_path, monkeypatch, capsys):
    root = tmp_path / "private-audio"
    root.mkdir()
    rng = np.random.default_rng(21)
    records = []
    for i, split in enumerate(("train", "validation")):
        path = root / f"private-{i}.wav"
        pcm = np.clip(rng.normal(0, 2000, 8000), -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(8000)
            out.writeframes(pcm.tobytes())
        records.append({
            "asset_id": f"private-asset-{i}", "group_id": f"private-group-{i}",
            "source_url": "https://example.org/owned", "local_path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "caption": "rain" if i == 0 else "wind", "language": "en", "split": split,
            "license": "CC0-1.0", "source_kind": "cc0", "rights_reviewed": True,
            "creator_permission": False, "voice_present": False,
            "sample_rate": 8000, "duration_seconds": 1.0,
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    splits, _ = load_reviewed_splits(manifest, root)
    assert splits["train"].provenance[0].asset_id == "private-asset-0"
    assert splits["train"].provenance[1].crop_index == 2
    assert splits["validation"].provenance[0].source_line == 2
    monkeypatch.setattr("sys.argv", ["corpus_readiness", str(manifest), str(root), "--require-ready"])
    with pytest.raises(SystemExit) as err:
        main()
    assert err.value.code == 2
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["wav_bytes_verified"]
    assert not result["readiness_gate_passed"]
    assert "private-" not in output


def test_group_balanced_mode_trains_with_audited_eligible_indices(monkeypatch):
    import math
    from research import audio_dataset, wave_diffusion

    splits = synthetic_splits()
    # Keep all categories covered while excluding a known silent crop.
    splits["train"].clips[0] = 0
    splits["validation"].clips[0] = 0
    monkeypatch.setattr(audio_dataset, "load_reviewed_splits",
                        lambda *args, **kwargs: (splits, {"wav_bytes_verified": True}))
    options = dict(steps=2, channels=8, diffusion_steps=8, seed=17,
                   manifest="reviewed.jsonl", audio_root="recordings", group_balanced=True)
    first = wave_diffusion.train(**options)
    second = wave_diffusion.train(**options)
    assert first["group_balanced_training"]
    assert first["training_examples"] == 35
    assert first["available_training_examples"] == 36
    assert first["heldout_examples"] == 29
    assert first["available_heldout_examples"] == 30
    assert first["corpus_readiness"]["readiness_gate_passed"]
    assert first["training_sampler"]["draws"] == 16
    assert first["training_sampler"] == second["training_sampler"]
    assert first["eligible_split_fingerprints"] == second["eligible_split_fingerprints"]
    assert math.isfinite(first["heldout_noise_mse"])
    assert first["weights_published"] is False


def test_group_balanced_mode_rejects_inadequate_corpus_before_optimization(monkeypatch):
    from research import audio_dataset, wave_diffusion
    splits = synthetic_splits()
    splits["train"].clips[5] = 0
    monkeypatch.setattr(audio_dataset, "load_reviewed_splits",
                        lambda *args, **kwargs: (splits, {"wav_bytes_verified": True}))
    monkeypatch.setattr(wave_diffusion, "training_step",
                        lambda *args, **kwargs: pytest.fail("Training ran despite failed coverage"))
    with pytest.raises(ValueError, match="failed group-balanced training readiness"):
        wave_diffusion.train(steps=1, channels=8, diffusion_steps=8,
                             manifest="reviewed.jsonl", audio_root="recordings", group_balanced=True)


@pytest.mark.parametrize("options", [{"group_balanced": True}, {"group_balanced": 1}])
def test_group_balanced_requires_reviewed_manifest(options):
    from research.wave_diffusion import train
    with pytest.raises(ValueError, match="reviewed manifest|boolean"):
        train(steps=1, channels=8, diffusion_steps=8, **options)
