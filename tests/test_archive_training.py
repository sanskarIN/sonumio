"""Archive training readiness equivalence, sampler replay and paired CPU evaluation."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_readiness import audit_archive, main as readiness_main
from research.archive_sampler import CounterGroupSampler, MAX_DRAWS
from research.archive_validation import evaluate_archive_validation, model_fingerprint
from research.archive_wave_training import ArchiveTrainingConfig, train_archive
from research.audio_dataset import ClipProvenance, LoadedSplit
from research.corpus_archive import CorpusArchiveReader, _write_archive
from research.corpus_archive_format import ArchiveError
from research.corpus_readiness import audit_loaded_splits, ReadinessPolicy
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES, TinyWaveDenoiser, DiffusionSchedule


def fixture():
    rng = np.random.default_rng(61)
    result = {}
    source_line = 0
    for split, groups in (("train", 2), ("validation", 1), ("test", 1)):
        data, labels, captions, provenance = [], [], [], []
        for label, name in enumerate(CATEGORIES):
            for group in range(groups):
                source_line += 1
                count = 3 if group == 0 else 1
                for crop in range(1, count+1):
                    data.append(rng.normal(0, .075, CLIP_FRAMES).astype(np.float32))
                    labels.append(label)
                    captions.append(name)
                    provenance.append(ClipProvenance(
                        f"private-{split}-asset-{label}-{group}",
                        f"private-{split}-group-{label}-{group}",
                        "en", source_line, crop,
                    ))
        result[split] = LoadedSplit(
            np.asarray(data, dtype=np.float32).reshape(-1,1,CLIP_FRAMES),
            np.asarray(labels, dtype=np.int64),
            len({item.asset_id for item in provenance}),
            hashlib.sha256(split.encode()).hexdigest(),
            tuple(captions), tuple(provenance), split,
        )
    # Keep coverage but exercise exact duplicate and quiet exclusions.
    result["train"].clips[0] = result["train"].clips[1]
    result["validation"].clips[0] = 0
    return result


def build(path, splits, shard_clips=4):
    report = {
        "wav_bytes_verified": True, "declaration": {"manifest_fingerprint": "a"*64},
        "split_fingerprints": {name: value.fingerprint for name,value in splits.items()},
        "clip_counts": {name: len(value.labels) for name,value in splits.items()},
    }
    return _write_archive(splits, report, path, shard_clips=shard_clips)


@pytest.fixture
def corpus(tmp_path):
    splits = fixture()
    path = tmp_path / "fixture.scorpus"
    report = build(path, splits)
    return path, report, splits


def test_streamed_audit_matches_in_memory_policy_and_fingerprints(corpus):
    path, report, splits = corpus
    original = audit_loaded_splits(splits)
    with CorpusArchiveReader(path, expected_sha256=report.archive_sha256) as reader:
        streamed = audit_archive(reader)
    assert original.report["readiness_gate_passed"]
    assert streamed.report["readiness_gate_passed"]
    assert streamed.report["splits"] == original.report["splits"]
    assert streamed.report["coverage_failures"] == original.report["coverage_failures"]
    assert streamed.eligible_fingerprints == original.eligible_fingerprints
    for name in splits:
        assert streamed.indices(name) == original.eligible_indices[name]
    assert "private-train-" not in json.dumps(streamed.report)
    assert streamed.report["pcm_processed_one_shard_at_a_time"]


def test_readiness_independent_of_shard_packing(tmp_path):
    splits = fixture()
    a, b = tmp_path / "one.scorpus", tmp_path / "two.scorpus"
    first = build(a, splits, 1)
    second = build(b, splits, 17)
    assert first.archive_sha256 != second.archive_sha256
    with CorpusArchiveReader(a) as reader:
        x = audit_archive(reader)
    with CorpusArchiveReader(b) as reader:
        y = audit_archive(reader)
    assert x.eligible_fingerprints == y.eligible_fingerprints
    assert x.report["splits"] == y.report["splits"]
    np.testing.assert_array_equal(CounterGroupSampler(x,seed=3).draw(64),
                                  CounterGroupSampler(y,seed=3).draw(64))


def test_readiness_failure_does_not_start_sampler(corpus):
    path, _, _ = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader, ReadinessPolicy(min_train_groups_per_category=3))
    assert not audit.report["readiness_gate_passed"]
    with pytest.raises(ArchiveError, match="failed readiness"):
        CounterGroupSampler(audit, seed=3)


def test_cli_readiness_aggregate_only(corpus, capsys):
    path, report, _ = corpus
    assert readiness_main([str(path), "--expected-sha256", report.archive_sha256,
                           "--require-ready"]) == 0
    text = capsys.readouterr().out
    assert json.loads(text)["readiness_gate_passed"]
    assert "private-train" not in text


def test_counter_sampler_partition_random_access_and_replay(corpus):
    path, _, splits = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader)
    first = CounterGroupSampler(audit, seed=19)
    second = CounterGroupSampler(audit, seed=19)
    prefix = first.draw(7)
    state = json.loads(json.dumps(first.describe()))
    next_chunk = first.draw(101)
    direct = second.draw(108)
    np.testing.assert_array_equal(np.concatenate((prefix,next_chunk)), direct)
    assert first.describe() == second.describe()
    np.testing.assert_array_equal(first.draw_at(7,101), next_chunk)
    assert first.position == 108
    restored = CounterGroupSampler.restore(audit, state)
    np.testing.assert_array_equal(restored.draw(101), next_chunk)
    assert restored.describe() == first.describe()
    labels = splits["train"].labels[direct]
    for start in range(0,len(labels),len(CATEGORIES)):
        assert sorted(labels[start:start+len(CATEGORIES)]) == list(range(len(CATEGORIES)))
    assert set(direct.tolist()) <= set(audit.indices("train"))
    assert not np.array_equal(CounterGroupSampler(audit,seed=20).draw(108), direct)


def test_sampler_balances_groups_not_crop_frequency(corpus):
    path, _, splits = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader)
    sampler = CounterGroupSampler(audit, seed=27)
    draws = np.concatenate([sampler.draw(600) for _ in range(10)])
    rain = draws[splits["train"].labels[draws] == 0]
    smaller = sum(splits["train"].provenance[i].group_id.endswith("-1") for i in rain)
    assert 400 <= smaller <= 600


@pytest.mark.parametrize("change", [
    {"draws": -1}, {"draws": MAX_DRAWS+1}, {"seed": True},
    {"index_trace_sha256": "0"*64}, {"plan_sha256": "0"*64},
    {"eligible_training_sha256": "0"*64}, {"protocol": "different"},
    {"extra": True},
])
def test_sampler_state_tampering_rejected(corpus, change):
    path, _, _ = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader)
    sampler = CounterGroupSampler(audit,seed=7)
    sampler.draw(9)
    with pytest.raises(ArchiveError):
        CounterGroupSampler.restore(audit, {**sampler.describe(), **change})


def test_sampler_eligibility_changes_rejected(corpus):
    path, _, _ = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader)
    bad = replace(audit, eligible={**audit.eligible,"train":audit.eligible["train"][1:]})
    with pytest.raises(ArchiveError, match="fingerprint changed"):
        CounterGroupSampler(bad,seed=3)


def test_sampler_bounds_and_failed_draw_does_not_advance(corpus):
    path, _, _ = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader)
    sampler = CounterGroupSampler(audit,seed=0)
    for count in (0, True, 1025):
        with pytest.raises(ArchiveError):
            sampler.draw(count)
    with pytest.raises(ArchiveError):
        sampler.draw_at(MAX_DRAWS-1,2)
    assert sampler.position == 0


def test_fixed_noise_validation_uses_only_validation_and_restores_mode(corpus, monkeypatch):
    path, _, _ = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader)
        old = reader.iter_shards
        def validation_only(split):
            assert split == "validation"
            yield from old(split)
        monkeypatch.setattr(reader, "iter_shards", validation_only)
        torch.manual_seed(3)
        model = TinyWaveDenoiser(8)
        schedule = DiffusionSchedule(8)
        before = model_fingerprint(model)
        rng_before = torch.random.get_rng_state().clone()
        first = evaluate_archive_validation(reader,audit,model,schedule,seed=7,repeats=2,batch_size=3)
        second = evaluate_archive_validation(reader,audit,model,schedule,seed=7,repeats=2,batch_size=8)
        assert first["noise_trace_sha256"] == second["noise_trace_sha256"]
        assert first["clip_weighted"]["zero_noise_mse"] == second["clip_weighted"]["zero_noise_mse"]
        assert first["clip_weighted"]["model_noise_mse"] == pytest.approx(
            second["clip_weighted"]["model_noise_mse"], rel=1e-6
        )
        assert first["clip_weighted"]["examples"] == len(audit.indices("validation"))*2
        assert first["group_macro"]["zero_noise_mse"] > 0
        assert not first["test_split_used"]
        assert model.training and model_fingerprint(model) == before
        assert torch.equal(rng_before,torch.random.get_rng_state())


def test_validation_rejects_wrong_archive_identity(corpus):
    path, _, _ = corpus
    with CorpusArchiveReader(path) as reader:
        audit = audit_archive(reader)
        bad = replace(audit,archive_sha256="0"*64)
        with pytest.raises(ArchiveError,match="another archive"):
            evaluate_archive_validation(reader,bad,TinyWaveDenoiser(8),DiffusionSchedule(8),seed=1)


def test_archive_training_deterministic_in_memory_and_bounded_batches(corpus, monkeypatch):
    path, report, _ = corpus
    original = CorpusArchiveReader.read_train_batch
    calls = []
    def tracked(reader, indices):
        calls.append(len(indices))
        return original(reader,indices)
    monkeypatch.setattr(CorpusArchiveReader,"read_train_batch",tracked)
    config = ArchiveTrainingConfig(steps=2,batch_size=3,channels=8,diffusion_steps=8,
                                   validation_repeats=1,validation_batch_size=4,seed=5)
    rng_before = torch.random.get_rng_state().clone()
    first = train_archive(path,expected_sha256=report.archive_sha256,config=config)
    second = train_archive(path,expected_sha256=report.archive_sha256,config=config)
    assert torch.equal(rng_before,torch.random.get_rng_state())
    assert calls == [3,3,3,3]
    assert first["sampler"]["draws"] == 6
    assert first["sampler"] == second["sampler"]
    assert first["trained_model_sha256"] == second["trained_model_sha256"]
    assert first["initial_model_sha256"] != first["trained_model_sha256"]
    assert first["validation"]["noise_trace_sha256"] == second["validation"]["noise_trace_sha256"]
    assert first["validation"]["model_state_unchanged"]
    assert first["weights_or_audio_files_written"] is False
    assert first["test_split_used_for_training_or_validation"] is False
    assert not list(path.parent.glob("*.npz"))
    assert "private-train" not in json.dumps(first)


def test_training_readiness_fails_before_optimizer(tmp_path, monkeypatch):
    splits = fixture()
    splits["train"].clips[3] = 0  # last eligible crop from a rain group
    path = tmp_path / "inadequate.scorpus"
    report = build(path,splits)
    monkeypatch.setattr("research.archive_wave_training.training_step",
                        lambda *a,**k: pytest.fail("optimizer ran before readiness"))
    with pytest.raises(ArchiveError,match="failed group-balanced training readiness"):
        train_archive(path,expected_sha256=report.archive_sha256,
                      config=ArchiveTrainingConfig(steps=1,channels=8,diffusion_steps=8))


@pytest.mark.parametrize("changes", [
    {"steps":0},{"seed":True},{"batch_size":65},{"channels":9},
    {"diffusion_steps":7},{"diffusion_profile":"unknown"},
    {"steps":50_000,"batch_size":64},{"validation_repeats":0},
])
def test_training_config_strict_bounds(changes):
    with pytest.raises(ArchiveError):
        ArchiveTrainingConfig(**changes)
