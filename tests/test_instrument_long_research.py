"""Sustained-note absolute phase, source split and actual optimizer tests."""
from dataclasses import replace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.instrument_harmonic_model import initialize_model, instrument_loss
from research.instrument_long_model import (
    initialize_windowed, interpolate_anchors, render_complete, sine_window, MAX_WINDOW,
    MAX_NOTE_FRAMES,
)
from research.instrument_long_data import (
    LongNote, LongGrid, Example, Corpus, build_corpus, render_example,
    training_schedule, collate, region_span, REGIONS, PANELS, INSTRUMENTS,
)
from research.instrument_long_study import StudyPlan, train, evaluate_note, run_study
from research.weight_ema import model_state_sha256


def controls():
    return torch.tensor([0, 1]), torch.tensor([60, 69]), torch.tensor([500, 800])


def small_grid():
    return LongGrid(training_pitches=(60,), evaluation_pitches=(66,),
                    training_velocities=(500,), evaluation_velocities=(750,),
                    training_frames=(12000,), evaluation_frames=(16000,))


@pytest.fixture(scope="module")
def corpus():
    return build_corpus(small_grid())


@pytest.mark.parametrize("frames",[12000,36000,96000])
def test_window_partition_preserves_absolute_clock(frames):
    model = initialize_windowed(32,17)
    args = controls()
    a = torch.cat([x for _,x in render_complete(model,*args,frames,block=4096)],dim=1)
    b = torch.cat([x for _,x in render_complete(model,*args,frames,block=7001)],dim=1)
    assert a.shape == (2,frames)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    assert torch.count_nonzero(a[:,0]) == 0
    assert torch.count_nonzero(a[:,-1]) == 0
    assert a.abs().max() <= .800001


def test_windowed_short_domain_compatible_with_original():
    original = initialize_model(32,17)
    extended = initialize_windowed(32,17)
    assert model_state_sha256(original) == model_state_sha256(extended)
    args = controls()
    old = original(*args,4096)
    new = extended(*args,4096)
    torch.testing.assert_close(old,new,rtol=1e-4,atol=2e-6)
    with pytest.raises(ValueError):
        original(*args,12000)
    with pytest.raises(ValueError):
        extended(*args,12000)


def test_absolute_anchor_interpolation_and_gradient():
    anchors = torch.arange(16,dtype=torch.float32).reshape(1,1,16).requires_grad_()
    clock = torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float32)
    result = interpolate_anchors(anchors,clock)
    torch.testing.assert_close(result[0,0],clock*15.)
    result.sum().backward()
    assert anchors.grad is not None and torch.isfinite(anchors.grad).all()


@pytest.mark.parametrize("frames,start,count",[
    (96001,0,2048),(12000,-1,2048),(12000,0,0),
    (12000,0,MAX_WINDOW+1),(12000,11999,2),
    (12000,True,2048),(12000,0,2048.0),
])
def test_invalid_window_geometry(frames,start,count):
    with pytest.raises(ValueError):
        initialize_windowed(32,17).render_window(*controls(),frames,start,count)


def test_sine_has_no_boundary_reset():
    args = controls()
    a = sine_window(*args,24000,2000,4096)
    b = torch.cat([sine_window(*args,24000,2000,1000),
                   sine_window(*args,24000,3000,3096)],dim=1)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_actual_gradients_at_middle_of_long_note():
    model = initialize_windowed(32,17)
    note = render_example(LongNote("pad",60,500,36000))
    labels = torch.tensor([INSTRUMENTS.index("pad")])
    output = model.render_window(labels,torch.tensor([60]),torch.tensor([500]),36000,16000,2048)
    target = torch.from_numpy(note.samples[16000:18048].copy())[None,:]
    loss = instrument_loss(output,target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(torch.count_nonzero(p.grad)>0 for p in model.parameters())


@pytest.mark.parametrize("change",[
    {"training_pitches":(60,60)},{"evaluation_pitches":(60,)},
    {"evaluation_velocities":(500,)},{"evaluation_frames":(12000,)},
    {"training_frames":(11999,)},{"training_frames":(96001,)},
])
def test_invalid_or_overlapping_source_axes(change):
    with pytest.raises(ValueError):
        replace(small_grid(),**change)


def test_source_panels_and_pcm_are_disjoint(corpus):
    training = corpus.get("train")
    ids = {r.note.identity for r in training}
    hashes = {r.sha256 for r in training}
    for panel in PANELS[1:]:
        rows = corpus.get(panel)
        assert len(rows) == 5
        assert not ids & {r.note.identity for r in rows}
        assert not hashes & {r.sha256 for r in rows}
        assert all(not r.samples.flags.writeable for r in rows)
    assert build_corpus(small_grid()).receipt() == corpus.receipt()


def test_reject_duplicate_waveform_under_different_metadata(corpus):
    panels = list(corpus.panels)
    rows = list(panels[1])
    rows[0] = Example(rows[0].note,panels[0][0].raw)
    panels[1] = tuple(rows)
    with pytest.raises(ValueError,match="PCM overlaps"):
        Corpus(corpus.grid,tuple(panels))


def test_training_windows_cover_all_regions_and_replay():
    grid = replace(small_grid(),training_frames=(12000,24000))
    rows = build_corpus(grid).get("train")
    a, digest = training_schedule(rows,updates=12,batch_size=3,seed=17)
    b, replay = training_schedule(rows,updates=12,batch_size=3,seed=17)
    assert a == b and digest == replay
    assert {(batch.frames,batch.region) for batch in a} == {(n,r) for n in (12000,24000) for r in REGIONS}
    for batch in a:
        lo, hi = region_span(batch.frames,batch.region)
        assert lo <= batch.start <= hi-batch.count
        labels,pitches,velocity,targets = collate(rows,batch)
        assert targets.shape == (3,2048)
        np.testing.assert_array_equal(targets[0].numpy(),rows[batch.indices[0]].samples[batch.start:batch.start+2048])


def test_whole_note_metrics_cover_every_frame(corpus):
    example = corpus.get("joint")[0]
    model = initialize_windowed(32,17)
    before = model_state_sha256(model)
    metrics = evaluate_note(example,model=model)
    assert metrics["full"]["samples"] == example.note.frames
    assert sum(metrics[r]["samples"] for r in REGIONS) == example.note.frames
    assert all(metrics[r]["relative_wave_mse"] >= 0 for r in REGIONS)
    assert before == model_state_sha256(model)
    json.dumps(metrics,allow_nan=False)


def test_train_validates_schedule_and_changes_weights(corpus):
    plan = StudyPlan(grid=corpus.grid,seeds=(17,29),width=32,updates=3,batch_size=2)
    schedule, digest = training_schedule(corpus.get("train"),updates=3,batch_size=2,seed=plan.sampler_seed)
    model = initialize_windowed(32,17)
    before = model_state_sha256(model)
    with pytest.raises(ValueError,match="schedule"):
        train(model,corpus,plan,tuple(reversed(schedule)))
    assert before == model_state_sha256(model)
    report = train(model,corpus,plan,schedule)
    assert report["trained_sha256"] != before
    assert report["schedule_sha256"] == digest
    assert report["window_samples_by_region"] == dict.fromkeys(REGIONS,4096)


def test_small_replicated_full_duration_experiment():
    plan = StudyPlan(grid=small_grid(),seeds=(17,29),width=32,updates=3,batch_size=2)
    report = run_study(plan)
    assert report["full_note_evaluation"]
    assert report["evaluation_used_for_training"] is False
    assert report["model_weights_published"] is False
    assert len(report["results"]) == 2
    for result in report["results"]:
        assert result["training"]["updates"] == 3
        for panel in PANELS[1:]:
            assert result["evaluation"]["wrong"][panel]["wrong_conditions_per_note"] == 4
            assert set(result["paired"][panel]) == {"full",*REGIONS}
    json.dumps(report,allow_nan=False)
