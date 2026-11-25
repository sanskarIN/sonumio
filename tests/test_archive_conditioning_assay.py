"""Research controls for paired class interventions on private archive fixtures."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_conditioning_assay import (
    AssayConfig, evaluate_conditioning, assay_checkpoint, timestep_panel, _PairedSummary,
)
from research.archive_readiness import audit_archive
from research.archive_checkpoint import CheckpointOptions
from research.archive_wave_training import ArchiveTrainingConfig,train_archive
from research.archive_validation import model_fingerprint
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError
from research.wave_diffusion import CATEGORIES,TinyWaveDenoiser,DiffusionSchedule
from test_archive_training import fixture,build


@pytest.fixture
def corpus(tmp_path):
    path=tmp_path/"fixture.scorpus"
    result=build(path,fixture())
    return path,result.archive_sha256


def test_zero_model_ties_every_category_and_matches_zero_control(corpus):
    path,sha=corpus
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        model=TinyWaveDenoiser(8)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        report=evaluate_conditioning(reader,ready,model,DiffusionSchedule(8),config=AssayConfig(repeats=1))
    scores=report["clip_weighted"]
    assert scores["matched_mse"]==scores["zero_predictor_mse"]
    assert scores["mean_wrong_minus_matched"]==pytest.approx(0,abs=1e-12)
    assert scores["hardest_wrong_minus_matched"]==0
    assert scores["null_minus_matched"]==0
    assert scores["mean_rank"]==3.5
    assert scores["strict_win_fraction"]==0
    assert scores["best_tie_fraction"]==1
    assert scores["cases"]==report["eligible_validation_clips"]*3
    for matrix in report["error_matrices_by_noise_level"].values():
        for row in matrix:
            assert len(row)==7
            np.testing.assert_allclose(row,row[0],rtol=0,atol=0)


def test_paired_summary_distinguishes_best_wrong_and_ties():
    summary=_PairedSummary()
    summary.add(np.array([1.,2.,3.,4.,5.,6.,7.]),0,1.)
    summary.add(np.array([1.,1.,2.,3.,4.,5.,6.]),0,1.)
    result=summary.report()
    assert result["strict_win_fraction"]==.5
    assert result["best_tie_fraction"]==.5
    assert result["mean_rank"]==1.25
    assert result["hardest_wrong_minus_matched"]==.5


def test_assay_trace_batch_invariant_and_validation_only(corpus,monkeypatch):
    path,sha=corpus
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        original=reader.iter_shards
        def checked(split):
            assert split=="validation"
            yield from original(split)
        monkeypatch.setattr(reader,"iter_shards",checked)
        model=TinyWaveDenoiser(8)
        schedule=DiffusionSchedule(8)
        initial=model_fingerprint(model)
        rng=torch.get_rng_state().clone()
        first=evaluate_conditioning(reader,ready,model,schedule,config=AssayConfig(repeats=1,batch_size=2))
        second=evaluate_conditioning(reader,ready,model,schedule,config=AssayConfig(repeats=1,batch_size=7))
        assert first["noise_trace_sha256"]==second["noise_trace_sha256"]
        for key in ("matched_mse","mean_wrong_mse","null_mse","zero_predictor_mse"):
            assert first["clip_weighted"][key]==pytest.approx(second["clip_weighted"][key],rel=1e-6)
        assert first["prediction_rows"]==len(ready.indices("validation"))*3*7
        assert first["group_macro"]["recording_groups"]==6
        assert first["timesteps"]=={"early":0,"middle":3,"late":7}
        assert torch.equal(rng,torch.get_rng_state())
        assert model.training and model_fingerprint(model)==initial
        assert not first["test_split_scored"]
        assert "private-validation-asset-" not in json.dumps(first)
        assert "private-validation-group-" not in json.dumps(first)


def test_changed_eligibility_rejected(corpus):
    path,sha=corpus
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        changed=replace(ready,eligible={**ready.eligible,"validation":ready.eligible["validation"][1:]})
        with pytest.raises(ArchiveError,match="fingerprint changed"):
            evaluate_conditioning(reader,changed,TinyWaveDenoiser(8),DiffusionSchedule(8))


def test_checkpoint_assay_read_only(corpus,tmp_path,monkeypatch):
    path,sha=corpus
    checkpoint=tmp_path/"private.stcheckpoint"
    training=ArchiveTrainingConfig(steps=2,channels=8,diffusion_steps=8,batch_size=2,
                                   validation_repeats=1,validation_batch_size=3)
    report=train_archive(path,expected_sha256=sha,config=training,
                         checkpoint=CheckpointOptions(final_path=checkpoint))
    pin=report["checkpoint_receipts"][0]["sha256"]
    before=checkpoint.read_bytes()
    rng=torch.get_rng_state().clone()
    def forbidden(*args,**kwargs):
        pytest.fail("Assay must not optimize")
    monkeypatch.setattr(torch.optim.AdamW,"step",forbidden)
    result=assay_checkpoint(path,archive_sha256=sha,checkpoint=checkpoint,checkpoint_sha256=pin,
                            training=training,config=AssayConfig(repeats=1))
    assert result["model_sha256"]==report["trained_model_sha256"]
    assert result["completed_training_steps"]==2
    assert result["files_written"]==0
    assert result["optimizer_steps_executed"]==0
    assert result["generated_audio_semantics_established"] is False
    assert checkpoint.read_bytes()==before
    assert torch.equal(rng,torch.get_rng_state())
    with pytest.raises(ArchiveError,match="SHA256 mismatch"):
        assay_checkpoint(path,archive_sha256=sha,checkpoint=checkpoint,checkpoint_sha256="0"*64,
                         training=training)


@pytest.mark.parametrize("options",[{"seed":True},{"seed":-1},{"repeats":0},{"repeats":5},{"batch_size":17}])
def test_config_validation(options):
    with pytest.raises(ArchiveError):
        AssayConfig(**options)


def test_panel_bounds():
    assert timestep_panel(1000)==(0,499,999)
    with pytest.raises(ArchiveError):
        timestep_panel(7)


def test_matched_seeded_initialization_comparison(corpus,tmp_path):
    path,sha=corpus
    checkpoint=tmp_path/"trained.stcheckpoint"
    training=ArchiveTrainingConfig(steps=2,channels=8,diffusion_steps=8,batch_size=2,
                                   validation_repeats=1,validation_batch_size=3)
    report=train_archive(path,expected_sha256=sha,config=training,
                         checkpoint=CheckpointOptions(final_path=checkpoint))
    result=assay_checkpoint(path,archive_sha256=sha,checkpoint=checkpoint,
                            checkpoint_sha256=report["checkpoint_receipts"][0]["sha256"],
                            training=training,config=AssayConfig(repeats=1),compare_initial=True)
    initial=result["initialization_assay"]
    comparison=result["initialization_comparison"]
    assert initial["model_sha256"]==report["initial_model_sha256"]
    assert comparison["trained_model_sha256"]==report["trained_model_sha256"]
    assert initial["noise_trace_sha256"]==result["noise_trace_sha256"]
    assert comparison["matched_corruption_trace_verified"]
    assert comparison["clip_weighted"]["matched_mse"]==(
        result["clip_weighted"]["matched_mse"]-initial["clip_weighted"]["matched_mse"])
    assert comparison["group_macro"]["hardest_wrong_minus_matched"]==(
        result["group_macro"]["hardest_wrong_minus_matched"]-
        initial["group_macro"]["hardest_wrong_minus_matched"])
    assert comparison["statistical_significance_claimed"] is False
    assert comparison["generated_audio_semantics_established"] is False


def test_comparison_rejects_unpaired_noise_trace(corpus):
    from research.archive_conditioning_assay import compare_initialization
    path,sha=corpus
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        result=evaluate_conditioning(reader,ready,TinyWaveDenoiser(8),DiffusionSchedule(8),
                                     config=AssayConfig(repeats=1))
    equal=compare_initialization(result,result)
    assert all(x==0 for x in equal["clip_weighted"].values())
    with pytest.raises(ArchiveError,match="different assay panel"):
        compare_initialization(result,{**result,"noise_trace_sha256":"0"*64})
