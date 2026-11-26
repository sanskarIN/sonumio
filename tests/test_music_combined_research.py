"""Combined learned structure and melody pipeline with paired controls."""
from dataclasses import replace
import json
import pytest

torch=pytest.importorskip("torch")

from sonumio.arrangement import ArrangementPlan,compose_arrangement,arrangement_digest
from sonumio.music_analysis import analyze_score
from research.music_combined_study import (
    CombinedPlan,ARMS,source_panel,training_panel,build_variant,generate_panel,
    run_study,_batch_indices,
)
from research.music_structure_data import extract as extract_structure,fingerprint as structure_hash
from research.music_melody_data import extract as extract_melody,fingerprint as melody_hash
from research.music_structure_model import StructureGRU
from research.music_melody_model import MelodyGRU
from research.weight_ema import model_state_sha256


def small_plan():
    return CombinedPlan(model_seeds=(17,29),training_seeds=(0,1,2),
                        development_seeds=(201,202),sampling_seeds=(701,),
                        bars=8,width=32,updates=2,batch_size=2)


def test_paired_source_panel_identity():
    a,ha=source_panel((17,29),8)
    b,hb=source_panel((17,29),8)
    assert a==b and ha==hb
    assert len(a)==12
    assert source_panel((29,17),8)[1]!=ha
    for plan,s,m in a:
        original=compose_arrangement(plan)
        assert extract_structure(original)==s
        assert extract_melody(original)==m


def test_novel_subsets_exclude_exact_training_matches():
    train,dev,ns,nm,receipt=training_panel(small_plan())
    structures={structure_hash(s,tokens_only=True) for _,s,_ in train}
    melodies={melody_hash(m,pitches_only=True) for _,_,m in train}
    assert len(train)==18 and len(dev)==12
    assert all(structure_hash(s,tokens_only=True) not in structures for _,s,_ in ns)
    assert all(melody_hash(m,pitches_only=True) not in melodies for _,_,m in nm)
    assert receipt["novel_structure_rows"]+receipt["development_structure_exact_overlap_rows"]==len(dev)
    assert receipt["novel_melody_rows"]+receipt["development_melody_exact_overlap_rows"]==len(dev)


def test_batch_schedule_is_shared_and_reproducible():
    p=small_plan()
    a,ha=_batch_indices(p,17,18)
    b,hb=_batch_indices(p,17,18)
    c,hc=_batch_indices(p,29,18)
    assert a==b and ha==hb
    assert a!=c and ha!=hc
    assert len(a)==2 and all(len(x)==2 for x in a)


@pytest.mark.parametrize("style",["calm","cinematic","rhythmic"])
@pytest.mark.parametrize("mode",["major","minor"])
def test_combined_variants_are_valid_and_paired(style,mode):
    plan=ArrangementPlan(style=style,mode=mode,bars=8,bpm=120,seed=17)
    structure=StructureGRU(32)
    melody=MelodyGRU(32)
    rng=torch.get_rng_state().clone()
    before=(model_state_sha256(structure),model_state_sha256(melody))
    results={arm:build_variant(plan,structure,melody,701,arm) for arm in ARMS}
    assert results["procedural"]==compose_arrangement(plan)
    # When only melody is changed, the full bar-level chord/onset
    # sequence is identical. When both are learned, structure sampling
    # is the same as the structure-only arm.
    assert extract_structure(results["procedural"])==extract_structure(results["learned_melody"])
    assert extract_structure(results["learned_structure"])==extract_structure(results["learned_both"])
    for arm,score in results.items():
        stats=analyze_score(score)
        assert stats["final_lead_tonic"]
        assert stats["strong_beat_chord_hits"]==stats["strong_beat_notes"]
        assert score.plan==plan
        assert score.harmony[-2:]==(4,0)
        assert score==build_variant(plan,structure,melody,701,arm)
    assert before==(model_state_sha256(structure),model_state_sha256(melody))
    assert torch.equal(rng,torch.get_rng_state())


def test_combined_melody_uses_generated_rhythm_and_never_reference_pitch(monkeypatch):
    import research.music_combined_study as module
    plan=ArrangementPlan(bars=8,bpm=120,seed=17)
    structure=StructureGRU(32)
    melody=MelodyGRU(32)
    first=build_variant(plan,structure,melody,701,"learned_both")
    original=module.reconstruct
    def changed(*args):
        # Change only provisional melody pitches, retaining legal
        # chord/scale pitches and the exact generated onset schedule.
        from research.music_melody_data import valid_tokens,replace_melody
        score=original(*args)
        sequence=extract_melody(score)
        altered=tuple(valid_tokens(c,final=i==len(sequence.pitches)-1)[0]
                      for i,c in enumerate(sequence.contexts))
        return replace_melody(score,altered)
    monkeypatch.setattr(module,"reconstruct",changed)
    second=build_variant(plan,structure,melody,701,"learned_both")
    assert first==second


def test_combined_generation_panel_compares_each_arm():
    p=small_plan()
    training,dev,_,_,_=training_panel(p)
    structure=StructureGRU(32)
    melody=MelodyGRU(32)
    reports=generate_panel(structure,melody,training,dev,p)
    assert tuple(reports)==ARMS
    for arm,report in reports.items():
        assert report["sample_count"]==6
        assert all(r["final_tonic"] and r["strong_chord_hits"]==r["strong_notes"] for r in report["rows"])
        audio=report["first_audio_render"]
        assert audio["channels"]==2
        assert audio["sample_rate"]==24000
        assert audio["wav_bytes"]==44+audio["frames"]*4
        assert audio["metrics"]["clipping_fraction"]==0


@pytest.mark.parametrize("kwargs",[
    {"model_seeds":(17,)},{"training_seeds":(201,)},{"sampling_seeds":(1,1)},
    {"bars":17},{"updates":257},{"width":16},{"batch_size":True},
    {"model_seeds":(1,2,3,4),"updates":256,"batch_size":32},
])
def test_invalid_combined_plan(kwargs):
    with pytest.raises(ValueError):
        CombinedPlan(**kwargs)


def test_training_smoke_and_complete_receipts():
    p=small_plan()
    before=torch.get_rng_state().clone()
    report=run_study(p)
    assert torch.equal(torch.get_rng_state(),before)
    assert report["training_examples"]==16
    assert report["models_trained_separately"]
    assert report["generation_conditions_on_sampled_structure"]
    assert report["constraints_not_learned"]
    assert not report["perceptual_quality_established"]
    assert not report["independent_test_scored"]
    for run in report["runs"]:
        assert all(a!=b for a,b in zip(run["initial_model_sha256"],run["trained_model_sha256"]))
        assert set(run["generation"])==set(ARMS)
        assert run["trained"]["structure_all"]["bars"]>0
        assert run["trained"]["melody_novel"]["tokens"]>0
        if report["data"]["novel_structure_rows"]==0:
            assert run["trained"]["structure_novel"] is None
    json.dumps(report,allow_nan=False)
