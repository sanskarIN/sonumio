"""Long-form symbolic structure, clock and validation contracts."""
from dataclasses import replace
import math
import numpy as np
import pytest

from sonumio.arrangement import (
    ArrangementPlan, Arrangement, Event, Section, TRACKS, PITCHED, MAX_FRAMES,
    compose_arrangement, arrangement_digest, chord_classes, event_key,
)
from sonumio.music import STYLES, MODES
from sonumio.music_analysis import analyze_score
from sonumio.errors import GenerationError


@pytest.mark.parametrize("bpm",[60,73,96,137,160])
@pytest.mark.parametrize("swing",[0.,.17,.33,.6])
def test_clock_absolute_beat_and_swing_endpoints(bpm,swing):
    p=ArrangementPlan(bpm=bpm,bars=4,swing=swing)
    clocks=[p.frame(i) for i in range(p.ticks+1)]
    assert clocks[0]==0 and all(b>a for a,b in zip(clocks,clocks[1:]))
    for beat in range(17):
        assert abs(clocks[beat*8]-beat*60*24000/bpm)<=.5
    for beat in range(16):
        midpoint=(beat+(1+swing)/2)*60*24000/bpm
        assert abs(clocks[beat*8+4]-midpoint)<=.5
    assert clocks[-1]==round(16*60*24000/bpm)


@pytest.mark.parametrize("bars",[4,5,6,7,8,12,16,24,32])
def test_sections_cover_entire_arrangement(bars):
    score=compose_arrangement(ArrangementPlan(bars=bars,bpm=120))
    assert sum(s.bars for s in score.sections)==bars
    assert score.sections[0].kind=="intro"
    assert score.sections[-1].kind=="finale"
    assert score.harmony[-2:]==(4,0)
    assert score.sections[-1].first_bar+score.sections[-1].bars==bars
    assert score.plan.frame(score.plan.ticks)<=MAX_FRAMES
    assert len(score.events)<=4096


@pytest.mark.parametrize("style",STYLES)
@pytest.mark.parametrize("mode",tuple(MODES))
@pytest.mark.parametrize("root",[48,60,71])
def test_pitch_and_voice_constraints(style,mode,root):
    plan=ArrangementPlan(style,root,mode,137,8,17,.33)
    score=compose_arrangement(plan)
    scale={(root+x)%12 for x in MODES[mode]}
    for event in score.events:
        assert event.tick+event.length<=plan.ticks
        if event.track in PITCHED:
            assert event.pitch%12 in scale
        if event.track=="pad":
            assert event.pitch%12 in chord_classes(plan,score.harmony[event.tick//32])
    stats=analyze_score(score)
    assert stats["final_lead_tonic"]
    assert stats["strong_beat_chord_hits"]==stats["strong_beat_notes"]
    assert stats["tracks"]["lead"]["max_polyphony"]==1
    assert stats["tracks"]["pad"]["max_polyphony"]==3
    assert stats["events"]==sum(t["events"] for t in stats["tracks"].values())
    assert sum(s["events"] for s in stats["sections"])==stats["events"]
    assert all(0<=t["active_fraction"]<=1 for t in stats["tracks"].values())
    if style=="calm":
        assert not any(t["events"] for name,t in stats["tracks"].items() if name not in PITCHED)


def test_seed_replay_and_swing_preserves_symbolic_events():
    plan=ArrangementPlan(bars=8,seed=13)
    a=compose_arrangement(plan)
    b=compose_arrangement(plan)
    c=compose_arrangement(replace(plan,swing=.5))
    d=compose_arrangement(replace(plan,seed=14))
    assert a==b
    assert arrangement_digest(a)==arrangement_digest(b)
    assert a.events==c.events and a.harmony==c.harmony
    assert arrangement_digest(a)!=arrangement_digest(c)
    assert a.events!=d.events
    assert a.sections==d.sections


@pytest.mark.parametrize("kwargs",[
    {"style":[]},{"mode":None},{"root":47},{"root":True},{"bpm":161},
    {"bars":3},{"bars":33},{"bars":32,"bpm":60},{"seed":-1},
    {"swing":float("nan")},{"swing":float("inf")},{"swing":.61},
])
def test_invalid_arrangement_plans(kwargs):
    with pytest.raises(GenerationError):
        ArrangementPlan(**kwargs)


@pytest.mark.parametrize("kwargs",[
    {"track":None},{"track":"invalid"},{"pitch":97},{"tick":-1},
    {"length":0},{"velocity":float("nan")},{"velocity":0},{"pan":1.1},
])
def test_invalid_event(kwargs):
    data={"track":"lead","pitch":60,"tick":0,"length":4,"velocity":.5,"pan":0}
    data.update(kwargs)
    with pytest.raises(GenerationError):
        Event(**data)


def test_invalid_score_structure_and_note_overlap():
    p=ArrangementPlan(bars=4)
    a=compose_arrangement(p)
    with pytest.raises(GenerationError):
        replace(a,sections=(Section("finale",1,4),))
    with pytest.raises(GenerationError):
        replace(a,events=tuple(reversed(a.events)))
    first=a.events[0]
    with pytest.raises(GenerationError):
        replace(a,events=tuple(sorted(a.events+(first,),key=event_key)))
    lead=next(e for e in a.events if e.track=="lead")
    overlap=replace(lead,velocity=.21)
    with pytest.raises(GenerationError):
        replace(a,events=tuple(sorted(a.events+(overlap,),key=event_key)))
    with pytest.raises(GenerationError):
        replace(a,events=(Event("lead",60,0,100,.5,0),))
    with pytest.raises(GenerationError):
        replace(a,events=(Event("lead",61,0,4,.5,0),))


def test_analysis_of_minimal_valid_custom_score():
    p=ArrangementPlan(bars=4)
    score=Arrangement(p,(Section("finale",0,4),),(0,0,4,0),
                      (Event("keys",60,0,8,.5,0),))
    result=analyze_score(score)
    assert result["events"]==1
    assert result["tracks"]["keys"]["active_fraction"]==8/p.ticks
    assert result["tracks"]["lead"]["events"]==0
    assert result["mean_absolute_lead_interval"] is None
    assert result["repeated_phrase_fraction"] is None
    assert not result["final_lead_tonic"]
