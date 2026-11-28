"""Preflight is a pure, audio-engine-free description of finite-Fourier work."""
from dataclasses import replace

import pytest

from sonumio.midi_workstation.contracts import MidiLimitError, MidiWorkstationError
from sonumio.midi_workstation.multitrack_preflight import (
    PROTOCOL, MAX_POLYPHONY, build_multitrack_preflight,
)
from sonumio.midi_workstation.multitrack_preview import (
    MultiTrackPreviewSettings, TrackPatch,
)
from sonumio.midi_workstation.score import Score, ScoreTrack, ScoreNote, note_sort_key


def note(identifier, start=0, duration=480, pitch=69, pan=64):
    return ScoreNote(identifier, start, duration, pitch, pan=pan)


def track(identifier,channel=0,*notes,muted=False,solo=False):
    return ScoreTrack(identifier,identifier.title(),channel,0,0,0,
        tuple(sorted(notes,key=note_sort_key)),muted,solo)


def score(*tracks,end=960):
    return replace(Score.empty(end=end),tracks=tuple(tracks))


def settings(*patches,channels=2,rate=8000,**kwargs):
    return MultiTrackPreviewSettings(tuple(sorted(patches,key=lambda p:p.track_id)),
        channels=channels,sample_rate=rate,**kwargs)


def patch(name,release=0):
    return TrackPatch(name,'sine',.05,
        attack_frames=0,decay_frames=0,sustain_level=1.,release_frames=release)


def duet():
    return score(track('a',0,note('a1',0,480)),
                 track('b',1,note('b1',480,480,pitch=60)))


def test_plan_is_pure_deterministic_and_byte_budgeted():
    score_a=duet()
    digest=score_a.sha256
    config=settings(patch('a'),patch('b',release=32))
    first=build_multitrack_preflight(score_a,config)
    second=build_multitrack_preflight(score_a,config)
    assert first==second
    assert first.source_score_sha256==digest
    assert first.output_frames==8032
    assert first.rendered_notes==2
    assert first.skipped_percussion_notes==0
    assert first.maximum_polyphony==1
    assert first.active_voice_frames==8032
    assert first.float_workspace_bytes==8032*2*16
    assert first.work_samples==8032*2*2
    assert first.schedule.output_frames==8032
    assert [x.track_id for x in first.stems]==['a','b']
    assert [(x.notes,x.gate_frames,x.release_voice_frames)
            for x in first.stems]==[(1,4000,4000),(1,4000,4032)]
    report=first.as_dict()
    assert report['protocol']==PROTOCOL
    assert report['source_score_sha256']==digest
    assert report['audio_rendered'] is False
    assert report['trained_model_inference'] is False
    assert report==second.as_dict()
    assert score_a.sha256==digest


def test_missing_patch_or_extra_patch_rejected_without_audio_core():
    for choices in ((patch('a'),),(patch('a'),patch('b'),patch('c'))):
        with pytest.raises(MidiWorkstationError,match='cover exactly'):
            build_multitrack_preflight(duet(),settings(*choices))


def test_optional_percussion_omission_requires_explicit_permission():
    composition=score(track('a',0,note('melody')),
                      track('kick',9,note('drum',pitch=36)))
    with pytest.raises(MidiWorkstationError,match='Percussion requires'):
        build_multitrack_preflight(composition,settings(patch('a')))
    allowed=build_multitrack_preflight(composition,
        settings(patch('a'),allow_skip_percussion=True))
    assert allowed.skipped_percussion_notes==1
    assert allowed.rendered_notes==1
    assert allowed.stems[0].notes==1


def test_muted_track_excluded_unless_explicitly_enabled():
    composition=score(track('a',0,note('a-note')),
                      track('b',1,note('b-note'),muted=True))
    default=build_multitrack_preflight(composition,settings(patch('a')))
    assert default.rendered_notes==1
    included=build_multitrack_preflight(composition,
        settings(patch('a'),patch('b'),include_muted=True))
    assert included.rendered_notes==2


def test_cannot_render_mono_when_note_pan_not_centered():
    composition=score(track('a',0,note('left',pan=0)))
    with pytest.raises(MidiWorkstationError,match='centered note pan'):
        build_multitrack_preflight(composition,settings(patch('a'),channels=1))


def test_polyphony_counts_release_tail_at_boundaries():
    composition=score(track('a',0,note('n1',0,480),
                            note('n2',480,480,pitch=60)))
    detached=build_multitrack_preflight(composition,settings(patch('a')))
    overlapping=build_multitrack_preflight(composition,
                                          settings(patch('a',release=32)))
    assert detached.maximum_polyphony==1
    assert detached.stems[0].maximum_polyphony==1
    assert overlapping.maximum_polyphony==2
    assert overlapping.stems[0].maximum_polyphony==2


def test_scheduling_rejects_more_than_sixty_four_voices():
    chord=tuple(note(f'n{i}',pitch=40+i%32) for i in range(MAX_POLYPHONY+1))
    composition=score(track('a',0,*chord))
    with pytest.raises(MidiLimitError,match='64 simultaneous voices'):
        build_multitrack_preflight(composition,settings(patch('a')))


def test_note_count_budget_is_checked_before_audio_dependency():
    chord=tuple(note(f'n{i}',pitch=60+i) for i in range(3))
    composition=score(track('a',0,*chord))
    with pytest.raises(MidiLimitError,match='note count budget'):
        build_multitrack_preflight(composition,
            settings(patch('a'),maximum_notes=2))


def test_work_budget_is_checked_before_audio_dependency():
    composition=duet()
    with pytest.raises(MidiLimitError,match='track work budget'):
        build_multitrack_preflight(composition,
            settings(patch('a'),patch('b'),maximum_work_samples=100))


def test_audio_frame_memory_budget_checked_before_audio_dependency():
    composition=score(track('a',0,note('n')),end=115200)
    with pytest.raises(MidiLimitError,match='float PCM memory budget'):
        build_multitrack_preflight(composition,
            settings(patch('a'),rate=24000))


def test_invalid_types_rejected():
    with pytest.raises(MidiWorkstationError):
        build_multitrack_preflight(None,settings(patch('a')))
    with pytest.raises(MidiWorkstationError):
        build_multitrack_preflight(duet(),None)


def test_long_score_rejected_at_synthesizer_duration_boundary():
    # A MIDI transport can represent long schedules, but this renderer must
    # stay within its underlying finite-Fourier engine's 120-second contract.
    composition=score(track('a',0,note('n')),end=480*4*121)
    with pytest.raises(MidiLimitError,match='120-second budget'):
        build_multitrack_preflight(composition,settings(patch('a')))
