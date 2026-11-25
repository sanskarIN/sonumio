"""Numerical reference fitting and paired unclipped generation regressions."""
import hashlib
import json
import numpy as np
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("scipy")

from research.archive_signal_reference import (
    spectral_features,signal_statistics,fit_reference,validation_reference_report,classify,FEATURE_NAMES,
)
from research.archive_generated_assay import (
    GenerationConfig,initial_noise,reverse_indices,sample_unclipped,evaluate_generated,assay_checkpoint,
)
from research.archive_readiness import audit_archive
from research.archive_validation import model_fingerprint
from research.archive_checkpoint import CheckpointOptions
from research.archive_wave_training import ArchiveTrainingConfig,train_archive
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError
from research.wave_diffusion import CLIP_FRAMES,CATEGORIES,DiffusionSchedule,TinyWaveDenoiser
from test_archive_training import fixture,build


def tone(freq):
    return (.2*np.sin(2*np.pi*freq*np.arange(CLIP_FRAMES)/8000)).astype(np.float32)


def test_feature_gain_dc_invariance_and_frequency_sensitivity():
    wave=tone(250)
    a=spectral_features(wave)
    b=spectral_features((wave*2+.1).astype(np.float32))
    assert a.shape==(len(FEATURE_NAMES),)
    np.testing.assert_allclose(a,b,atol=.02,rtol=.01)
    high=spectral_features(tone(2000))
    assert high[-3]>a[-3]
    assert spectral_features(np.full(CLIP_FRAMES,.1,dtype=np.float32)) is None


def test_statistics_expose_overrange_before_clip():
    signal=tone(440)*20
    stats=signal_statistics(signal)
    assert stats["peak"]>1
    assert stats["overrange_fraction"]>0
    assert stats["centered_rms"]>1
    assert not stats["silent_or_constant"]
    assert signal_statistics(np.zeros(CLIP_FRAMES,dtype=np.float32))["crest_factor"] is None
    with pytest.raises(ArchiveError):
        spectral_features(np.full(CLIP_FRAMES,np.nan,dtype=np.float32))


@pytest.fixture
def corpus(tmp_path):
    path=tmp_path/"fixture.scorpus"
    result=build(path,fixture())
    return path,result.archive_sha256


def test_fit_and_validation_split_isolation(corpus,monkeypatch):
    path,sha=corpus
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        old=reader.iter_shards
        calls=[]
        def watched(split):
            calls.append(split)
            assert split in ("train","validation")
            yield from old(split)
        monkeypatch.setattr(reader,"iter_shards",watched)
        probe=fit_reference(reader,ready)
        validation=validation_reference_report(reader,ready,probe)
        assert calls==["train","validation"]
        assert probe.centers.shape==(6,len(FEATURE_NAMES))
        assert np.all(probe.scale>0)
        with pytest.raises(ValueError):
            probe.centers.setflags(write=True)
        assert sum(map(sum,validation["confusion_counts"]))==len(ready.indices("validation"))
        assert not validation["heldout_validation_used_for_fitting"]
        assert not validation["test_split_scored"]
        assert 0<=validation["category_balanced_group_accuracy"]<=1
        assert classify(probe,None,0)["predicted"] is None


def test_reference_independent_of_shard_partition(tmp_path):
    splits=fixture()
    a,b=tmp_path/"a.scorpus",tmp_path/"b.scorpus"
    first=build(a,splits,1)
    second=build(b,splits,17)
    with CorpusArchiveReader(a,expected_sha256=first.archive_sha256) as reader:
        pa=fit_reference(reader,audit_archive(reader))
    with CorpusArchiveReader(b,expected_sha256=second.archive_sha256) as reader:
        pb=fit_reference(reader,audit_archive(reader))
    assert pa.fingerprint==pb.fingerprint
    np.testing.assert_array_equal(pa.centers,pb.centers)


def test_ddim_zero_predictor_analytic_unclipped_output():
    model=TinyWaveDenoiser(8)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    schedule=DiffusionSchedule(8,profile="cosine_zero_terminal_v1")
    noise=initial_noise(7)
    before=noise.clone()
    rng=torch.get_rng_state().clone()
    actual=sample_unclipped(model,schedule,noise,2,guidance=1,reverse_steps=4)
    expected=(noise[0,0]/schedule.alpha_bars[-1].sqrt()).numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-4)
    assert np.max(np.abs(actual))>1
    assert torch.equal(noise,before)
    assert torch.equal(rng,torch.get_rng_state())
    assert model.training


def test_null_guidance_category_invariant():
    model=TinyWaveDenoiser(8)
    schedule=DiffusionSchedule(8)
    noise=initial_noise(4)
    a=sample_unclipped(model,schedule,noise,0,guidance=0,reverse_steps=3)
    b=sample_unclipped(model,schedule,noise,5,guidance=0,reverse_steps=3)
    np.testing.assert_array_equal(a,b)
    assert not torch.equal(noise,initial_noise(5))


def test_generated_panel_reports_pairing_and_raw_signal(corpus):
    path,sha=corpus
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        probe=fit_reference(reader,ready)
        reference=validation_reference_report(reader,ready,probe)
    model=TinyWaveDenoiser(8)
    before=model_fingerprint(model)
    config=GenerationConfig(seeds=(7,9),reverse_steps=2,guidances=(0.,1.))
    result=evaluate_generated(model,DiffusionSchedule(8),probe,reference,config=config)
    assert len(result["cases"])==2*6*2
    assert result["forward_calls"]==48
    assert len(result["null_output_sha256_by_seed"])==2
    assert result["by_guidance"]["0.0"]["paired_margin_delta_vs_null"]["mean"]==0
    assert result["model_sha256"]==model_fingerprint(model)==before
    assert result["test_split_scored"] is False
    assert result["perceptual_quality_established"] is False
    assert result["files_written"]==0
    assert "private-validation-group-" not in json.dumps(result)


def test_checkpoint_generation_uses_verified_state(corpus,tmp_path,monkeypatch):
    path,sha=corpus
    output=tmp_path/"private.stcheckpoint"
    training=ArchiveTrainingConfig(steps=1,batch_size=2,channels=8,diffusion_steps=8,
                                   validation_repeats=1)
    report=train_archive(path,expected_sha256=sha,config=training,
                         checkpoint=CheckpointOptions(final_path=output))
    pin=report["checkpoint_receipts"][0]["sha256"]
    def forbidden(*args,**kwargs):
        pytest.fail("Generation evaluation must not optimize")
    monkeypatch.setattr(torch.optim.AdamW,"step",forbidden)
    result=assay_checkpoint(path,archive_sha256=sha,checkpoint=output,checkpoint_sha256=pin,
                            training=training,config=GenerationConfig((7,9),2,(0.,1.)))
    assert result["model_sha256"]==report["trained_model_sha256"]
    assert hashlib.sha256(output.read_bytes()).hexdigest()==pin
    assert result["completed_training_steps"]==1
    assert not list(tmp_path.glob("*.wav"))


@pytest.mark.parametrize("settings",[
    {"seeds":(1,)},{"seeds":(1,1)},{"seeds":(True,2)},{"reverse_steps":1},
    {"reverse_steps":65},{"guidances":(1.,)},{"guidances":(0.,float("nan"))},
    {"guidances":(0.,0.)},{"guidances":(0.,5.)},
])
def test_generation_config_rejects_invalid_panels(settings):
    with pytest.raises(ArchiveError):
        GenerationConfig(**settings)


def test_reverse_step_endpoints_unique():
    assert reverse_indices(8,4)==(7,4,2,0)
    assert reverse_indices(8,8)==tuple(range(7,-1,-1))
    with pytest.raises(ArchiveError):
        reverse_indices(8,9)
