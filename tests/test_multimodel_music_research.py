"""Multi-model routing and full-score research integration contracts."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch=pytest.importorskip("torch")

from research.instrument_note_data import GridPlan, NoteSpec
from research.instrument_harmonic_study import StudyPlan as ShortPlan
from research.instrument_long_data import LongGrid, LongNote
from research.instrument_long_study import StudyPlan as LongPlan
from research.percussion_data import DrumGrid
from research.percussion_study import StudyPlan as DrumPlan
from research.multimodel_music_routes import (
    FAMILIES, Exclusions, describe_routes, route_score, event_excitation,
)
from research.multimodel_music_render import (
    RenderPlan, compare_score, render_score, mix_feature_comparison,
)
from research.multimodel_music_study import StudyPlan, Sources, initialize, panel, run_study
from research.hybrid_music_study import ScoreCase
from sonumio.arrangement import TRACKS, compose_arrangement, arrangement_digest
from sonumio.music_mix import render_stems


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ScoreCase("rhythmic","minor",57,160,131,.15).to_plan())


@pytest.fixture(scope="module")
def models():
    return initialize(StudyPlan(),17)


def small_plan():
    seeds=(17,29)
    return StudyPlan(
        short=ShortPlan(grid=GridPlan(training_pitches=(60,),evaluation_pitches=(66,),
                         training_velocities=(500,),evaluation_velocities=(750,),
                         training_frames=(4096,),evaluation_frames=(6144,)),
                         model_seeds=seeds,width=32,updates=3,batch_size=2),
        sustained=LongPlan(grid=LongGrid(training_pitches=(60,),evaluation_pitches=(66,),
                         training_velocities=(500,),evaluation_velocities=(750,),
                         training_frames=(12000,),evaluation_frames=(16000,)),
                         seeds=seeds,width=32,updates=3,batch_size=2),
        percussion=DrumPlan(grid=DrumGrid(training_seeds=(7,),evaluation_seeds=(103,),
                         training_velocities=(500,),evaluation_velocities=(750,),
                         training_frames=(4096,),evaluation_frames=(6144,)),
                         seeds=seeds,width=32,updates=3,batch_size=2),
        cases=(ScoreCase("rhythmic","minor",57,160,131,.15),))


def test_routes_partition_every_score_event(score):
    routes=route_score(score)
    coverage=describe_routes(routes)
    assert tuple(row.event for row in routes)==score.events
    assert len({row.identity for row in routes})==len(routes)
    assert sum(coverage["events_by_family"].values())==len(routes)
    assert sum(coverage["samples_by_family"].values())==coverage["event_samples"]
    assert all(coverage["events_by_family"][family]>0 for family in FAMILIES)
    for row in routes:
        assert row.start==score.plan.frame(row.event.tick)
        assert row.stop==score.plan.frame(row.event.tick+row.event.length)
        if row.family=="short":
            assert 2048<=row.frames<=8192
        if row.family=="sustained":
            assert 12000<=row.frames<=96000
        if row.family=="percussion":
            assert row.event.track in ("kick","snare","hat")
            assert 2048<=row.frames<=8192


def test_training_exclusions_are_exact(score):
    routes=route_score(score)
    short=next(row for row in routes if row.family=="short")
    long=next(row for row in routes if row.family=="sustained")
    exclusions=Exclusions(
        frozenset((NoteSpec(short.event.track,short.event.pitch,short.velocity_milli,short.frames).identity,)),
        frozenset((LongNote(long.event.track,long.event.pitch,long.velocity_milli,long.frames).identity,)),
        frozenset((score.plan.seed,)))
    blocked=route_score(score,exclusions)
    assert next(r for r in blocked if r.identity==short.identity).reason=="training_control"
    assert next(r for r in blocked if r.identity==long.identity).reason=="training_control"
    assert all(r.family!="percussion" for r in blocked)


@pytest.mark.parametrize("kwargs",[
    {"short":set()},{"short":frozenset(("invalid",))},
    {"sustained":frozenset(("F"*64,))},{"drum_seeds":frozenset((-1,))},
])
def test_exclusions_reject_invalid_identity_contracts(kwargs):
    with pytest.raises(ValueError):
        Exclusions(**kwargs)


def test_independent_event_noise_replays(score):
    rows=[r for r in route_score(score) if r.family=="percussion"]
    a=event_excitation(rows[0],17)
    assert a.shape==(rows[0].frames+2,)
    np.testing.assert_array_equal(a,event_excitation(rows[0],17))
    assert not np.array_equal(a,event_excitation(rows[0],18))
    assert not np.array_equal(a,event_excitation(rows[1],17))


def test_reference_is_bit_exact(score):
    original=render_stems(score)
    candidate,report=render_score(score)
    assert sum(report["substituted_events"].values())==0
    for track in TRACKS:
        np.testing.assert_array_equal(candidate[track],original[track])


def test_model_inference_replays_and_preserves_state(score,models):
    models.short.train()
    models.sustained.eval()
    models.percussion.train()
    state=models.states()
    a,ar=render_score(score,RenderPlan("learned",replica=701),models=models)
    b,br=render_score(score,RenderPlan("learned",replica=701),models=models)
    assert ar==br
    assert state==models.states()==ar["model_sha256"]
    assert models.short.training and not models.sustained.training and models.percussion.training
    assert sum(ar["substituted_events"].values())==len(ar["notes"])
    assert ar["procedural_events"]+len(ar["notes"])==len(score.events)
    for track in TRACKS:
        np.testing.assert_array_equal(a[track],b[track])
        assert not a[track].flags.writeable


def test_ablation_only_changes_selected_tracks(score,models):
    reference=render_stems(score)
    candidate,report=render_score(score,RenderPlan("learned",families=("percussion",)),models=models)
    assert report["substituted_events"]["short"]==0
    assert report["substituted_events"]["sustained"]==0
    assert report["substituted_events"]["percussion"]>0
    for track in TRACKS:
        if not report["substituted_by_track"][track]:
            np.testing.assert_array_equal(candidate[track],reference[track])


@pytest.mark.parametrize("kwargs",[
    {"arm":"invalid"},{"families":()},{"families":("percussion","short")},
    {"families":("short","short")},{"replica":-1},{"pitched_offset":1},
    {"arm":"learned","drum_offset":3},
])
def test_bad_plan(kwargs):
    with pytest.raises(ValueError):
        RenderPlan(**kwargs)


def test_model_bundle_must_match_arm(score,models):
    with pytest.raises(ValueError):
        render_score(score,RenderPlan("learned"))
    with pytest.raises(ValueError):
        render_score(score,RenderPlan("generic"),models=models)


def test_generic_and_wrong_label_controls_use_same_routes(score,models):
    generic=compare_score(score,RenderPlan("generic"))
    matched=compare_score(score,RenderPlan("learned"),models=models)
    wrong=compare_score(score,RenderPlan("learned",drum_offset=1),models=models)
    assert generic["reference_wav_sha256"]==matched["reference_wav_sha256"]==wrong["reference_wav_sha256"]
    assert generic["routing"]["eligible"]==matched["routing"]["eligible"]==wrong["routing"]["eligible"]
    assert matched["routing"]["note_diagnostics"]["short"]==wrong["routing"]["note_diagnostics"]["short"]
    assert matched["routing"]["note_diagnostics"]["sustained"]==wrong["routing"]["note_diagnostics"]["sustained"]
    assert matched["routing"]["note_diagnostics"]["percussion"]!=wrong["routing"]["note_diagnostics"]["percussion"]
    json.dumps(matched,allow_nan=False)


def test_feature_comparison_is_phase_insensitive_and_handles_tail():
    rng=np.random.default_rng(17)
    a=rng.normal(0,.05,size=(2,4097)).astype(np.float32)
    assert all(value==0. for value in mix_feature_comparison(a,a).values())
    assert all(value==0. for value in mix_feature_comparison(-a,a).values())
    assert mix_feature_comparison(a*.5,a)["objective"]>0


def test_case_rejects_source_seed_overlap():
    plan=replace(small_plan(),cases=(ScoreCase("rhythmic","minor",57,160,7,.15),))
    with pytest.raises(ValueError,match="percussion training"):
        panel(plan,Sources.build(plan))


def test_study_rejects_mismatched_seed_panels():
    plan=small_plan()
    with pytest.raises(ValueError,match="replicate seeds"):
        replace(plan,percussion=replace(plan.percussion,seeds=(17,41)))


def test_small_actual_multimodel_study():
    report=run_study(small_plan())
    assert len(report["replicates"])==2
    assert set(report["family_summary"])==set(FAMILIES)
    for replica in report["replicates"]:
        assert replica["training"]["short"]["optimizer_updates"]==3
        assert replica["training"]["sustained"]["updates"]==3
        assert replica["training"]["percussion"]["updates"]==3
        case=replica["cases"][0]
        assert set(case["ablations"])==set(FAMILIES)
        assert set(case["summary"]["family_note_errors"])==set(FAMILIES)
    assert report["evaluation_used_for_training"] is False
    assert report["model_weights_published"] is False
    json.dumps(report,allow_nan=False)
