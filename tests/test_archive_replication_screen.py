"""Replication diagnostics, numerical controls, private archive/model integration."""
import json
import hashlib

import numpy as np
import pytest

from research.archive_replication_screen import (
    signature, similarities, ScreenPolicy, FRAMES, MEL_BANK,
    evaluate_replication, screen_checkpoint,
)


def random_wave(seed):
    return np.random.default_rng(seed).normal(0, .1, FRAMES).astype(np.float32)


def test_mel_filter_geometry_and_normalization():
    assert MEL_BANK.shape == (129, 24)
    assert np.all(MEL_BANK >= 0)
    np.testing.assert_allclose(MEL_BANK.sum(axis=0), 1)


@pytest.mark.parametrize("gain", [.2, 2., -.3, -2.])
def test_same_recording_affine_controls(gain):
    a = random_wave(1)
    b = (gain*a + .04).astype(np.float32)
    wave, mel = similarities(signature(a), signature(b))
    assert wave == pytest.approx(1, abs=1e-12)
    assert mel == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("shift", [-128, -64, 64, 128])
def test_mel_lag_retains_shifted_recording(shift):
    a = random_wave(2)
    b = random_wave(3)
    n = FRAMES-abs(shift)
    i, j = max(-shift, 0), max(shift, 0)
    b[j:j+n] = -.4*a[i:i+n] + .025
    wave, mel = similarities(signature(a), signature(b))
    assert wave < .2
    assert mel > .99
    np.testing.assert_allclose((wave, mel), similarities(signature(b), signature(a)), atol=1e-12)


def test_independent_random_waves_are_not_near_exact():
    wave, mel = similarities(signature(random_wave(4)), signature(random_wave(5)))
    assert wave < .2 and mel < .995


def test_constant_inputs_are_unscorable():
    assert signature(np.zeros(FRAMES, np.float32)) is None
    assert signature(np.full(FRAMES, .1, np.float32)) is None


@pytest.mark.parametrize("bad", [
    np.zeros(FRAMES, np.float64), np.zeros(FRAMES-1, np.float32),
    np.full(FRAMES, np.nan, np.float32), np.full(FRAMES, np.inf, np.float32),
    np.full(FRAMES, 1e7, np.float32),
])
def test_invalid_waveform_rejected(bad):
    with pytest.raises(ValueError):
        signature(bad)


@pytest.mark.parametrize("settings", [
    {"pair_budget": 0}, {"pair_budget": 100001}, {"pair_budget": True},
    {"mel_threshold": 1.}, {"waveform_threshold": float("nan")},
    {"mel_threshold": .8}, {"mel_threshold": True},
])
def test_policy_bounds(settings):
    with pytest.raises(ValueError):
        ScreenPolicy(**settings)


@pytest.fixture
def corpus(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from test_archive_training import fixture, build
    path = tmp_path/"fixture.scorpus"
    splits = fixture()
    result = build(path, splits)
    return path, result.archive_sha256, splits


def test_full_gallery_heldout_control_and_positive_replication(corpus, monkeypatch):
    from research.archive_generated_assay import GenerationConfig
    import research.archive_generated_assay as generated
    from research.archive_readiness import audit_archive
    from research.corpus_archive import CorpusArchiveReader
    from research.wave_diffusion import TinyWaveDenoiser, DiffusionSchedule

    path, sha, splits = corpus
    with CorpusArchiveReader(path, expected_sha256=sha) as reader:
        ready = audit_archive(reader)
        source = splits["train"].clips[ready.indices("train")[0], 0]
        calls = []
        def fake(model, schedule, noise, category, *, guidance, reverse_steps):
            calls.append((category, guidance))
            return source.copy() if guidance else random_wave(11)
        monkeypatch.setattr(generated, "sample_unclipped", fake)
        original = reader.iter_shards
        accessed = []
        def guarded(split):
            accessed.append(split)
            assert split in ("train", "validation")
            yield from original(split)
        monkeypatch.setattr(reader, "iter_shards", guarded)
        report = evaluate_replication(reader, ready, TinyWaveDenoiser(8), DiffusionSchedule(8),
                                      generation=GenerationConfig((7,9), 2, (0.,1.)))
    assert accessed == ["validation", "train"]
    assert calls.count((0,0)) == 2 and len(calls) == 14
    assert report["generated_queries"] == 14
    assert report["planned_pairs"] == report["gallery_crops"]*(14+report["validation_queries"])
    assert report["panels"]["guidance_1.0"]["waveform"]["review_candidates"] == 12
    assert report["panels"]["guidance_1.0"]["mel"]["review_candidates"] == 12
    assert report["panels"]["guidance_0.0"]["waveform"]["review_candidates"] == 0
    assert report["panels"]["heldout_validation"]["waveform"]["review_candidates"] == 0
    assert not report["memorization_established"]
    assert not report["memorization_absence_established"]
    assert "private-train" not in json.dumps(report, allow_nan=False)


def test_budget_checked_before_sampling(corpus, monkeypatch):
    import research.archive_generated_assay as generated
    from research.archive_generated_assay import GenerationConfig
    from research.archive_readiness import audit_archive
    from research.corpus_archive import CorpusArchiveReader
    from research.wave_diffusion import TinyWaveDenoiser, DiffusionSchedule
    path, sha, _ = corpus
    def forbidden(*args, **kwargs):
        pytest.fail("Budget must be checked before generation")
    monkeypatch.setattr(generated, "sample_unclipped", forbidden)
    with CorpusArchiveReader(path, expected_sha256=sha) as reader:
        ready = audit_archive(reader)
        with pytest.raises(ValueError, match="budget"):
            evaluate_replication(reader, ready, TinyWaveDenoiser(8), DiffusionSchedule(8),
                                 generation=GenerationConfig((7,9), 2, (0.,1.)),
                                 policy=ScreenPolicy(pair_budget=1))


def test_real_private_checkpoint_screen(corpus, tmp_path, monkeypatch):
    import torch
    from research.archive_checkpoint import CheckpointOptions
    from research.archive_generated_assay import GenerationConfig
    from research.archive_wave_training import ArchiveTrainingConfig, train_archive

    path, sha, _ = corpus
    checkpoint = tmp_path/"private.stcheckpoint"
    training = ArchiveTrainingConfig(steps=1,batch_size=2,channels=8,diffusion_steps=8,
                                     validation_repeats=1)
    result = train_archive(path, expected_sha256=sha, config=training,
                           checkpoint=CheckpointOptions(final_path=checkpoint))
    digest = result["checkpoint_receipts"][0]["sha256"]
    def forbidden(*args, **kwargs):
        pytest.fail("Screening must not update the checkpoint model")
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    report = screen_checkpoint(path, archive_sha256=sha, checkpoint=checkpoint,
                               checkpoint_sha256=digest, training=training,
                               generation=GenerationConfig((7,9),2,(0.,1.)))
    assert report["model_sha256"] == result["trained_model_sha256"]
    assert report["completed_training_steps"] == 1
    assert report["generated_queries"] == 14
    assert report["panels"]["guidance_0.0"]["queries"] == 2
    assert report["panels"]["guidance_1.0"]["queries"] == 12
    assert report["files_written"] == 0
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    assert not list(tmp_path.glob("*.wav"))
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError):
        screen_checkpoint(path, archive_sha256=sha, checkpoint=checkpoint,
                          checkpoint_sha256="0"*64, training=training)


def test_independent_maxima_do_not_imply_same_reference(monkeypatch):
    import research.archive_replication_screen as screen
    policy=ScreenPolicy()
    query=screen._Nearest("control", object())
    monkeypatch.setattr(screen,"similarities",lambda a,b:b)
    query.compare((.9999,.80),policy)
    query.compare((.80,.9999),policy)
    report=screen._summarize([query],policy)["cross_metric_agreement"]
    assert report=={
        "jointly_scorable_queries":1,
        "both_independent_maxima_above_threshold":1,
        "same_reference_above_both_thresholds":0,
        "independent_maxima_only":1,
    }
    query.compare((.9998,.997),policy)
    report=screen._summarize([query],policy)["cross_metric_agreement"]
    assert report["same_reference_above_both_thresholds"]==1
    assert report["independent_maxima_only"]==0


def test_joint_unscorable_is_not_a_match(monkeypatch):
    import research.archive_replication_screen as screen
    policy=ScreenPolicy()
    query=screen._Nearest("control", object())
    monkeypatch.setattr(screen,"similarities",lambda a,b:b)
    query.compare((1.,None),policy)
    silent=screen._Nearest("control",None)
    silent.compare((1.,1.),policy)
    report=screen._summarize([query,silent],policy)["cross_metric_agreement"]
    assert all(value==0 for value in report.values())
