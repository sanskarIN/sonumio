"""Known-source stress controls, deterministic edits and complete gallery checks."""
import json
import numpy as np
import pytest

from research.archive_replication_controls import (
    ControlPolicy, EDIT_NAMES, THRESHOLDS, edit_waveform, study_controls, main,
)
from research.archive_replication_screen import FRAMES, signature, similarities


def wave():
    return np.random.default_rng(20).normal(0,.1,FRAMES).astype(np.float32)


@pytest.mark.parametrize("name", EDIT_NAMES)
def test_edits_are_deterministic_and_do_not_mutate_source(name):
    x=wave()
    original=x.copy()
    y=edit_waveform(x,name,seed=41)
    assert y.dtype==np.float32 and y.shape==x.shape
    assert np.isfinite(y).all()
    np.testing.assert_array_equal(y,edit_waveform(x,name,seed=41))
    np.testing.assert_array_equal(x,original)
    assert not np.shares_memory(x,y)


@pytest.mark.parametrize("name,db", [("noise_30db",30),("noise_15db",15)])
def test_noise_snr(name,db):
    x=wave().astype(np.float64)
    y=edit_waveform(x.astype(np.float32),name,seed=3).astype(np.float64)
    assert 10*np.log10(np.mean((x-x.mean())**2)/np.mean((y-x)**2))==pytest.approx(db,abs=1e-4)
    assert not np.array_equal(y,edit_waveform(x.astype(np.float32),name,seed=4))


@pytest.mark.parametrize("name,shift", [("shift_64",64),("shift_128",128)])
def test_shift_is_not_circular(name,shift):
    x=wave()
    y=edit_waveform(x,name,seed=0)
    assert np.all(y[:shift]==0)
    np.testing.assert_array_equal(y[shift:],x[:-shift])


def test_pcm16_encoding_and_affine_similarity():
    x=wave()
    y=edit_waveform(x,"pcm16",seed=0)
    assert np.max(np.abs(x-y))<=1/65536
    wave_score,mel_score=similarities(signature(x),signature(edit_waveform(x,"gain_dc_polarity",seed=0)))
    assert wave_score==pytest.approx(1,abs=1e-12)
    assert mel_score==pytest.approx(1,abs=1e-12)


@pytest.mark.parametrize("kwargs",[{"seed":True},{"seed":-1},{"seed":2**32},{"pair_budget":0},{"pair_budget":100001}])
def test_policy_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        ControlPolicy(**kwargs)


@pytest.fixture
def corpus(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    from test_archive_training import fixture,build
    splits=fixture()
    path=tmp_path/"private.scorpus"
    receipt=build(path,splits)
    return path,receipt.archive_sha256,splits


def test_complete_gallery_group_balanced_controls(corpus,monkeypatch):
    from research.corpus_archive import CorpusArchiveReader
    from research.archive_readiness import audit_archive
    path,sha,_=corpus
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        original=reader.iter_shards
        seen=[]
        def guarded(split):
            seen.append(split)
            assert split in ("train","validation")
            yield from original(split)
        monkeypatch.setattr(reader,"iter_shards",guarded)
        result=study_controls(reader,ready)
    assert seen==["train","validation","train"]
    assert result["training_anchor_groups"]==12
    assert result["validation_anchor_groups"]==6
    assert result["planned_pairs"]==result["examined_pairs"]==result["gallery_crops"]*(12*8+6)
    for name in ("waveform","mel"):
        identity=result["panels"]["identity"][name]["known_source"]
        assert identity["scorable"]==12
        assert all(x["candidate_count"]==12 for x in identity["thresholds"])
        nonmember=result["panels"]["validation_nonmember"][name]
        assert "known_source" not in nonmember
        assert nonmember["nearest_gallery"]["queries"]==6
        for edit in EDIT_NAMES:
            summary=result["panels"][edit][name]
            for known,nearest in zip(summary["known_source"]["thresholds"],summary["nearest_gallery"]["thresholds"],strict=True):
                assert known["threshold"]==nearest["threshold"]
                assert known["candidate_count"]<=nearest["candidate_count"]
    assert not result["thresholds_fitted"] and not result["test_split_scored"]
    assert "private-train" not in json.dumps(result,allow_nan=False)


def test_shard_partition_does_not_change_control_metrics(corpus,tmp_path):
    from test_archive_training import build
    from research.corpus_archive import CorpusArchiveReader
    from research.archive_readiness import audit_archive
    a,sha,splits=corpus
    b=tmp_path/"different.scorpus"
    receipt=build(b,splits,17)
    reports=[]
    for path,digest in ((a,sha),(b,receipt.archive_sha256)):
        with CorpusArchiveReader(path,expected_sha256=digest) as reader:
            reports.append(study_controls(reader,audit_archive(reader)))
    assert reports[0]["panels"]==reports[1]["panels"]
    assert reports[0]["eligible_training_sha256"]==reports[1]["eligible_training_sha256"]


def test_budget_refusal_before_features(corpus,monkeypatch):
    import research.archive_replication_controls as controls
    from research.corpus_archive import CorpusArchiveReader
    from research.archive_readiness import audit_archive
    path,sha,_=corpus
    def forbidden(*args,**kwargs):
        pytest.fail("Pair budget must be checked first")
    monkeypatch.setattr(controls,"signature",forbidden)
    with CorpusArchiveReader(path,expected_sha256=sha) as reader:
        ready=audit_archive(reader)
        with pytest.raises(ValueError,match="budget"):
            study_controls(reader,ready,policy=ControlPolicy(pair_budget=1))


def test_cli_digest_and_budget_fail_without_json(corpus,capsys):
    path,sha,_=corpus
    for args in ([str(path),"--expected-sha256","0"*64],
                 [str(path),"--expected-sha256",sha,"--pair-budget","1"]):
        with pytest.raises(SystemExit) as err:
            main(args)
        assert err.value.code==2
        captured=capsys.readouterr()
        assert captured.out=="" and str(path) not in captured.err


def test_control_joint_threshold_requires_same_gallery_crop(monkeypatch):
    import research.archive_replication_controls as controls
    monkeypatch.setattr(controls,"similarities",lambda a,b:b)
    q=controls._Query("identity",object(),0)
    q.compare((.9999,.80),0)
    q.compare((.80,.9999),1)
    report=controls._panel_report([q])
    assert report["waveform"]["nearest_gallery"]["maximum"]==.9999
    assert report["mel"]["nearest_gallery"]["maximum"]==.9999
    assert report["joint_same_reference"]["nearest_gallery"]["maximum"]==.80
    assert report["joint_same_reference"]["known_source"]["maximum"]==.80
    assert all(x["candidate_count"]==0 for x in report["joint_same_reference"]["nearest_gallery"]["thresholds"])
    q.compare((.9998,.997),2)
    assert controls._panel_report([q])["joint_same_reference"]["nearest_gallery"]["maximum"]==.997


def test_control_joint_undefined_scores_remain_unscorable(monkeypatch):
    import research.archive_replication_controls as controls
    monkeypatch.setattr(controls,"similarities",lambda a,b:b)
    q=controls._Query("identity",object(),0)
    q.compare((1.,None),0)
    report=controls._panel_report([q])["joint_same_reference"]
    assert report["nearest_gallery"]["unscorable"]==1
    assert report["known_source"]["maximum"] is None
    assert all(x["fraction_of_scorable"] is None for x in report["known_source"]["thresholds"])
