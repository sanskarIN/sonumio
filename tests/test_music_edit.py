"""Score edits preserve source identity and reject invalid transactions."""
from dataclasses import replace
import pytest

from sonumio.arrangement import (
    Arrangement,ArrangementPlan,Event,Section,PITCHED,event_key,
    compose_arrangement,arrangement_digest,
)
from sonumio.music_edit import (
    Retime,Transpose,TrackArticulation,RemoveTrack,LeadBar,QuantizeVelocity,edit,
)
from sonumio.music_score_io import encode_score,decode_score
from sonumio.errors import GenerationError


def source():
    return compose_arrangement(ArrangementPlan(bars=8,bpm=120,root=60,seed=17))


def test_retime_preserves_symbolic_structure():
    a=source()
    b=Retime(160,.3).apply(a)
    assert a.events==b.events and a.sections==b.sections and a.harmony==b.harmony
    assert b.plan.frame(b.plan.ticks)==round(a.plan.frame(a.plan.ticks)*.75)
    assert arrangement_digest(a)!=arrangement_digest(b)


def test_transpose_preserves_intervals_and_percussion():
    a=source()
    b=Transpose(1).apply(a)
    assert b.plan.root==a.plan.root+1
    for track in {e.track for e in a.events}:
        x=[e for e in a.events if e.track==track]
        y=[e for e in b.events if e.track==track]
        assert len(x)==len(y)
        for first,second in zip(x,y,strict=True):
            assert second==replace(first,pitch=first.pitch+1) if track in PITCHED else second==first
    assert Transpose(-1).apply(b)==a


def test_transpose_rejects_out_of_range_without_octave_wrapping():
    a=source()
    before=arrangement_digest(a)
    with pytest.raises(GenerationError):
        Transpose(23).apply(a)
    assert arrangement_digest(a)==before


def test_track_articulation_affects_only_selected_bar_range():
    a=source()
    b=TrackArticulation("lead",1,3,2.,-.25).apply(a)
    for x,y in zip(a.events,b.events,strict=True):
        if x.track=="lead" and 32<=x.tick<96:
            assert y.velocity==round(min(1.,2*x.velocity),6)
            assert y.pan==round(x.pan-.25,6)
            assert replace(y,velocity=x.velocity,pan=x.pan)==x
        else:
            assert x==y


def test_remove_track_range_is_exact():
    a=source()
    b=RemoveTrack("hat",1,3).apply(a)
    assert b.events==tuple(e for e in a.events if not(e.track=="hat" and 32<=e.tick<96))
    assert a.harmony==b.harmony


def test_lead_bar_replacement_preserves_other_events():
    a=source()
    b=LeadBar(1,((0,8,60,.6),(16,8,63,.7))).apply(a)
    expected=[e for e in a.events if not(e.track=="lead" and 32<=e.tick<64)]
    expected.extend((Event("lead",60,32,8,.6,.18),Event("lead",63,48,8,.7,.18)))
    assert b.events==tuple(sorted(expected,key=event_key))


def test_lead_bar_rejects_scale_violation():
    a=source()
    with pytest.raises(GenerationError):
        LeadBar(1,((0,8,61,.6),)).apply(a)


def test_lead_bar_rejects_note_crossing_edit_boundary():
    a=source()
    p=a.plan
    events=(Event("lead",60,28,8,.5,0),Event("keys",60,0,4,.5,0))
    custom=replace(a,events=tuple(sorted(events,key=event_key)))
    with pytest.raises(GenerationError):
        LeadBar(1,()).apply(custom)
    with pytest.raises(GenerationError):
        LeadBar(0,()).apply(custom)


def test_quantized_velocity_grid():
    a=source()
    b=QuantizeVelocity(8).apply(a)
    for x,y in zip(a.events,b.events,strict=True):
        assert y.velocity==max(1,round(x.velocity*8))/8
        assert replace(y,velocity=x.velocity)==x


def test_edit_pipeline_receipt_and_roundtrip():
    a=source()
    digest=arrangement_digest(a)
    ops=(Retime(140,.3),Transpose(1),TrackArticulation("lead",0,8,.7,.1),
         RemoveTrack("hat",0,8),QuantizeVelocity(16))
    b,report=edit(a,ops)
    assert report["source_score_sha256"]==digest
    assert report["result_score_sha256"]==arrangement_digest(b)
    assert report["original_unchanged"]
    assert len(report["operations"])==len(ops)
    assert report["operations"][0]["before_sha256"]==digest
    for first,second in zip(report["operations"],report["operations"][1:]):
        assert first["after_sha256"]==second["before_sha256"]
    assert decode_score(encode_score(b))==b
    assert arrangement_digest(a)==digest


def test_failed_transaction_leaves_original_unchanged():
    a=source()
    before=encode_score(a)
    with pytest.raises(GenerationError):
        edit(a,(Retime(160),Transpose(23)))
    assert encode_score(a)==before
    with pytest.raises(GenerationError):
        edit(a,())
    with pytest.raises(GenerationError):
        edit(a,(Retime(160),)*33)
    with pytest.raises(GenerationError):
        edit(a,[Retime(160)])


@pytest.mark.parametrize("constructor",[
    lambda:Retime(),lambda:Retime(0),lambda:Retime(swing=float("nan")),
    lambda:Transpose(True),lambda:Transpose(24),
    lambda:TrackArticulation("unknown",0,1),
    lambda:TrackArticulation("lead",1,1),lambda:TrackArticulation("lead",0,1,0),
    lambda:RemoveTrack("pad",-1,1),lambda:LeadBar(-1,()),
    lambda:LeadBar(0,((0,8,60,.5),(7,2,60,.5))),
    lambda:LeadBar(0,((31,2,60,.5),)),
    lambda:LeadBar(0,((0,4,60,float("nan")),)),
    lambda:QuantizeVelocity(1),lambda:QuantizeVelocity(128),
])
def test_invalid_edits(constructor):
    with pytest.raises(GenerationError):
        constructor()
