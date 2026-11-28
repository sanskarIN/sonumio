"""Legacy arrangement to new workstation score real-repository integration."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

pytest.importorskip("sonumio.arrangement")
from sonumio.arrangement import (
    ArrangementPlan, Event, compose_arrangement, arrangement_digest,
)
from sonumio.music_score_io import encode_score as encode_legacy_score
from sonumio.midi_workstation.arrangement_import import (
    LegacyInstrumentRoute, LegacyImportPolicy, arrangement_to_score,
)
from sonumio.midi_workstation.arrangement_bridge import (
    ArrangementRoute, ArrangementBridgePolicy, bridge_to_arrangement,
)
from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.contracts import MidiWorkstationError, MidiLimitError
from sonumio.midi_workstation.score_io import decode_score as decode_workstation_score


def legacy(*events, swing=0., root=60, mode="major", bpm=120):
    base=compose_arrangement(ArrangementPlan(
        style="calm",bars=4,root=root,mode=mode,bpm=bpm,swing=swing,seed=2026,
    ))
    return replace(base,events=tuple(sorted(events,key=lambda e:(
        e.tick,("keys", "pad", "bass", "lead", "pluck", "kick", "snare", "hat").index(e.track),
        e.pitch,e.length,e.velocity,e.pan))))


def original(*,velocity=1.,pan=0.,swing=0.):
    return legacy(Event("lead",60,0,8,velocity,pan),swing=swing)


def route(instrument="lead",track_id="melody",channel=0,program=0):
    return LegacyInstrumentRoute(instrument,track_id,channel,program)


def policy(*routes,**kwargs):
    return LegacyImportPolicy(tuple(sorted(routes or (route(),),key=lambda r:r.instrument)),**kwargs)


def test_exact_expressive_note_import_without_quantization_flag():
    source=original()
    original_sha=arrangement_digest(source)
    score,ledger=arrangement_to_score(source,policy())
    assert score.note_count==1
    assert score.ppqn==480
    assert score.end==4*4*480
    assert score.tempo[0].microseconds_per_quarter==500000
    assert score.meter[0].numerator==4
    track=score.tracks[0]
    assert (track.id,track.channel,track.program)==("melody",0,0)
    note=track.notes[0]
    assert (note.start,note.duration,note.pitch,note.velocity,note.pan)==(0,480,60,127,64)
    assert ledger["velocity_quantized"]==0
    assert ledger["pan_quantized"]==0
    assert ledger["source_arrangement_sha256"]==original_sha
    assert ledger["score_sha256"]==score.sha256
    assert arrangement_digest(source)==original_sha
    assert ledger["sections_harmony_and_root_metadata_lost"] is True


def test_legacy_arrangement_note_returns_to_identical_event_when_expression_exact():
    source=original()
    score,_=arrangement_to_score(source,policy())
    recovered,receipt=bridge_to_arrangement(score,source,ArrangementBridgePolicy((
        ArrangementRoute("melody","lead"),
    )))
    assert recovered==source
    assert receipt["output_arrangement_sha256"]==arrangement_digest(source)


def test_default_policy_refuses_velocity_quantization():
    with pytest.raises(MidiWorkstationError,match="velocity cannot"):
        arrangement_to_score(original(velocity=.8),policy())


def test_default_policy_refuses_pan_quantization():
    with pytest.raises(MidiWorkstationError,match="pan cannot"):
        arrangement_to_score(original(pan=.3),policy())


def test_quantization_opt_in_is_auditable_and_bounded():
    source=original(velocity=.8,pan=.3)
    score,receipt=arrangement_to_score(source,policy(
        allow_velocity_quantization=True,allow_pan_quantization=True,
    ))
    note=score.tracks[0].notes[0]
    assert 1<=note.velocity<=127
    assert 0<=note.pan<=127
    assert receipt["velocity_quantized"]==1
    assert receipt["pan_quantized"]==1
    assert 0<receipt["maximum_velocity_error"]<=1/127
    assert 0<receipt["maximum_pan_error"]<=1/63
    assert receipt["exact_original_score_reconstructed"] is False


def test_swing_rejected_without_approval():
    with pytest.raises(MidiWorkstationError,match="swing loss requires explicit opt in"):
        arrangement_to_score(original(swing=.2),policy())


def test_swing_removal_is_reported():
    score,report=arrangement_to_score(original(swing=.2),policy(allow_swing_loss=True))
    assert score.note_count==1
    assert report["swing_discarded"] is True
    assert report["legacy_sample_frame_rounding_preserved"] is False


def test_user_selected_ppqn_preserves_eighth_beats():
    source=legacy(Event("lead",60,1,2,1.,0.))
    score,receipt=arrangement_to_score(source,policy(ppqn=960))
    note=score.tracks[0].notes[0]
    assert (note.start,note.duration)==(120,240)
    assert score.end==4*4*960
    assert receipt["ppqn"]==960


def test_program_channel_and_patch_explicit_per_route():
    source=original()
    specified=LegacyInstrumentRoute("lead","solo-5",3,81,4,12)
    score,receipt=arrangement_to_score(source,policy(specified))
    track=score.tracks[0]
    assert (track.id,track.channel,track.program,track.bank_msb,track.bank_lsb)==(
        "solo-5",3,81,4,12)
    assert receipt["routes"][0]["program"]==81


def test_drum_notes_map_to_channel_nine():
    source=legacy(Event("kick",36,0,4,1.,0.))
    score,receipt=arrangement_to_score(source,policy(route("kick","drums",9)))
    assert score.tracks[0].channel==9
    assert score.tracks[0].notes[0].pitch==36
    assert receipt["notes"]==1


def test_multi_instrument_legacy_maps_to_independent_tracks():
    source=legacy(Event("keys",60,0,8,1.,0.),Event("lead",64,8,8,1.,0.))
    score,receipt=arrangement_to_score(source,policy(route("keys","keys-1",0),
                                                    route("lead","lead-2",1)))
    assert [t.id for t in score.tracks]==["keys-1","lead-2"]
    assert [(n.pitch,t.channel) for t in score.tracks for n in t.notes]==[(60,0),(64,1)]
    assert receipt["events"]==2


def test_missing_or_extraneous_mapping_rejected():
    source=original()
    with pytest.raises(MidiWorkstationError,match="exactly note-bearing"):
        arrangement_to_score(source,policy(route("keys","keys",0)))
    with pytest.raises(MidiWorkstationError,match="exactly note-bearing"):
        arrangement_to_score(source,policy(route("keys","keys",0),route()))


def test_output_bounded_by_explicit_event_budget():
    source=legacy(Event("lead",60,0,8,1.,0.),Event("lead",64,8,8,1.,0.))
    with pytest.raises(MidiLimitError,match="event budget"):
        arrangement_to_score(source,policy(maximum_events=1))


def test_invalid_legacy_pitches_caught_by_arrangement_constructor():
    with pytest.raises(Exception):
        legacy(Event("lead",23,0,8,1.,0.))


def test_cli_publishes_canonical_workstation_score_with_receipt(tmp_path,capsys):
    source=original()
    filename=tmp_path/"legacy.json"
    output=tmp_path/"editable.json"
    filename.write_bytes(encode_legacy_score(source))
    args=["arrangement-import",str(filename),"--route","lead:melody:0:0",
          "--output",str(output)]
    assert main(args)==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["conversion"]["source_arrangement_sha256"]==arrangement_digest(source)
    assert receipt["conversion"]["events"]==1
    score=decode_workstation_score(output.read_bytes())
    assert score.note_count==1
    assert main(args)==2
    assert capsys.readouterr().out==""


def test_cli_requires_explicit_7bit_loss_authorization(tmp_path,capsys):
    source=original(velocity=.72,pan=.25)
    filename=tmp_path/"legacy.json"
    output=tmp_path/"editable.json"
    filename.write_bytes(encode_legacy_score(source))
    args=["arrangement-import",str(filename),"--route","lead:melody:0:0",
          "--output",str(output)]
    assert main(args)==2
    assert not output.exists()
    capsys.readouterr()
    assert main(args+["--allow-velocity-quantization","--allow-pan-quantization"])==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["conversion"]["velocity_quantized"]==1
    assert receipt["conversion"]["pan_quantized"]==1


def test_cli_wrong_input_digest_refuses_publication(tmp_path,capsys):
    source=tmp_path/"legacy.json"
    output=tmp_path/"editable.json"
    source.write_bytes(encode_legacy_score(original()))
    assert main(["arrangement-import",str(source),"--route","lead:melody:0:0",
                 "--output",str(output),"--expected-sha256","0"*64])==2
    assert not output.exists()
    assert capsys.readouterr().out==""


def test_cli_refuses_source_as_output_with_overwrite(tmp_path,capsys):
    source=tmp_path/"legacy.json"
    old=encode_legacy_score(original())
    source.write_bytes(old)
    assert main(["arrangement-import",str(source),"--route","lead:melody:0:0",
                 "--output",str(source),"--overwrite"])==2
    assert source.read_bytes()==old
    assert capsys.readouterr().out==""


def test_cli_refuses_symlink_input(tmp_path,capsys):
    real=tmp_path/"real.json"
    alias=tmp_path/"alias.json"
    output=tmp_path/"editable.json"
    real.write_bytes(encode_legacy_score(original()))
    try:
        alias.symlink_to(real)
    except (OSError,NotImplementedError):
        pytest.skip("symlinks unsupported")
    assert main(["arrangement-import",str(alias),"--route","lead:melody:0:0",
                 "--output",str(output)])==2
    assert not output.exists()
    assert capsys.readouterr().out==""


def test_repeated_import_is_deterministic_and_never_mutates_legacy():
    source=original(velocity=.70,pan=.33)
    policy_value=policy(allow_velocity_quantization=True,allow_pan_quantization=True)
    orig_sha=arrangement_digest(source)
    a,ledger_a=arrangement_to_score(source,policy_value)
    b,ledger_b=arrangement_to_score(source,policy_value)
    assert a.sha256==b.sha256
    assert ledger_a==ledger_b
    assert arrangement_digest(source)==orig_sha
