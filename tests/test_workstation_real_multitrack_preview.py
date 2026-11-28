"""Full Sonumio audio-core integration for explicit per-track sound synthesis."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

pytest.importorskip("sonumio.audio_core.wavetable")
from sonumio.audio_core.waveio import read_wav_file
from sonumio.midi_workstation.cli import main
from sonumio.midi_workstation.contracts import MidiLimitError,MidiWorkstationError
from sonumio.midi_workstation.multitrack_preview import (
    TrackPatch,MultiTrackPreviewSettings,render_multitrack_preview,
    export_multitrack_preview,
)
from sonumio.midi_workstation.score import Score,ScoreNote,ScoreTrack,note_sort_key
from sonumio.midi_workstation.score_io import encode_score


def note(identifier="one",pitch=69,start=0,duration=480,velocity=100,pan=64):
    return ScoreNote(identifier,start,duration,pitch,velocity,0,pan)


def source(*tracks, end=960):
    plan=Score.empty(ppqn=480,end=end,title="Original multitrack")
    return replace(plan,tracks=tuple(tracks))


def track(identifier,channel,program=0,*notes,muted=False,solo=False):
    return ScoreTrack(identifier,identifier.title(),channel,program,0,0,
                      tuple(sorted(notes,key=note_sort_key)),muted,solo)


def patch(identifier,wave="sine",gain=.10,release=0):
    return TrackPatch(identifier,wave,gain,
                      attack_frames=0,decay_frames=0,sustain_level=1.,
                      release_frames=release)


def config(*patches,channels=2,rate=8000,**kwargs):
    return MultiTrackPreviewSettings(tuple(sorted(patches,key=lambda p:p.track_id)),
                                     sample_rate=rate,channels=channels,**kwargs)


def two_tracks():
    return source(
        track("melody",0,0,note("a4",69,0,480)),
        track("bass",1,33,note("c3",48,480,480)),
    )


def test_per_track_synthesis_is_bounded_deterministic_and_immutable():
    score=two_tracks()
    original_digest=score.sha256
    settings=config(patch("bass","square",.10),patch("melody","sine",.10))
    first=render_multitrack_preview(score,settings)
    second=render_multitrack_preview(score,settings)
    assert first.audio.sample_rate==8000
    assert first.audio.frames==8000
    assert first.audio.channels==2
    assert first.audio.samples.shape==(2,8000)
    assert not first.audio.samples.flags.writeable
    np.testing.assert_array_equal(first.audio.samples,second.audio.samples)
    assert first.report==second.report
    assert first.report["rendered_notes"]==2
    assert first.report["tracks_rendered"]==2
    assert first.report["skipped_percussion_notes"]==0
    assert [s["track_id"] for s in first.report["stems"]]==["bass","melody"]
    assert first.report["stems"][0]["pcm_sha256"]!=first.report["stems"][1]["pcm_sha256"]
    assert first.report["trained_model_inference"] is False
    assert score.sha256==original_digest


def test_stereo_pan_keeps_left_and_right_separate():
    score=source(track("a",0,0,note("left",69,0,480,pan=0)),
                 track("b",1,2,note("right",60,480,480,pan=127)))
    result=render_multitrack_preview(score,config(patch("a"),patch("b")))
    samples=result.audio.samples
    assert float(np.max(np.abs(samples[0,:4000])))>.001
    assert float(np.max(np.abs(samples[1,:4000])))<1e-7
    assert float(np.max(np.abs(samples[1,4000:])))>.001
    assert float(np.max(np.abs(samples[0,4000:])))<1e-7


def test_mono_centered_pan_supported():
    score=source(track("one",0,0,note()))
    preview=render_multitrack_preview(score,config(patch("one"),channels=1))
    assert preview.audio.samples.shape==(1,8000)
    assert preview.report["mix_peak"]>0


def test_mono_noncenter_pan_rejected():
    score=source(track("one",0,0,note(pan=127)))
    with pytest.raises(MidiWorkstationError,match="centered note pan"):
        render_multitrack_preview(score,config(patch("one"),channels=1))


def test_missing_patch_mapping_rejected():
    with pytest.raises(MidiWorkstationError,match="cover exactly"):
        render_multitrack_preview(two_tracks(),config(patch("melody")))


def test_unknown_patch_track_rejected():
    score=source(track("one",0,0,note()))
    with pytest.raises(MidiWorkstationError,match="cover exactly"):
        render_multitrack_preview(score,config(patch("unrecognized")))


def test_muted_track_remains_silent_without_explicit_override():
    score=source(track("active",0,0,note("left")),
                 track("muted",1,0,note("hidden"),muted=True))
    result=render_multitrack_preview(score,config(patch("active")))
    assert result.report["rendered_notes"]==1
    included=render_multitrack_preview(score,config(patch("active"),patch("muted"),include_muted=True))
    assert included.report["rendered_notes"]==2


def test_percussion_requires_explicit_skip():
    score=source(track("melody",0,0,note()),
                 track("drum",9,0,note("kick",36)))
    with pytest.raises(MidiWorkstationError,match="explicit skip approval"):
        render_multitrack_preview(score,config(patch("melody")))
    result=render_multitrack_preview(score,config(patch("melody"),allow_skip_percussion=True))
    assert result.report["rendered_notes"]==1
    assert result.report["skipped_percussion_notes"]==1


def test_high_gain_mix_never_clips_silently():
    score=source(track("a",0,0,note("a",69,velocity=127)),
                 track("b",1,0,note("b",69,velocity=127)))
    patches=(patch("a","square",1.),patch("b","square",1.))
    with pytest.raises(MidiWorkstationError,match="exceeds declared peak"):
        render_multitrack_preview(score,config(*patches))
    safe=render_multitrack_preview(score,config(*patches,normalization="peak",maximum_peak=.9))
    assert safe.report["applied_scale"]<1.
    assert safe.report["mix_peak"]<=.90001
    assert safe.report["mix_peak_before_normalization"]>.9


def test_global_polyphony_guard_catches_many_tracks():
    notes=tuple(note(f"n{i}",pitch=60+(i%12)) for i in range(65))
    score=source(track("one",0,0,*notes))
    with pytest.raises(MidiLimitError,match="64 simultaneous"):
        render_multitrack_preview(score,config(patch("one")))


def test_track_work_budget_before_wavetable_synthesis():
    score=source(track("a",0,0,note("a")),end=115200)
    with pytest.raises(MidiLimitError,match="float PCM memory budget"):
        render_multitrack_preview(score,config(patch("a"),rate=24000))


def test_no_clobber_and_wav_metadata(tmp_path):
    score=source(track("a",0,0,note()))
    filename=tmp_path/"out.wav"
    settings=config(patch("a"))
    receipt=export_multitrack_preview(score,filename,settings,output_bits=16)
    wav=read_wav_file(filename)
    assert wav.frames==8000
    assert wav.sample_rate==8000
    assert wav.channels==2
    assert receipt["render"]["rendered_notes"]==1
    assert len(receipt["wav_sha256"])==64
    with pytest.raises(Exception):
        export_multitrack_preview(score,filename,settings)


def test_wrong_output_sha_prevents_publication(tmp_path):
    score=source(track("a",0,0,note()))
    filename=tmp_path/"out.wav"
    with pytest.raises(Exception):
        export_multitrack_preview(score,filename,config(patch("a")),
                                  expected_output_sha256="0"*64)
    assert not filename.exists()


def test_score_source_cannot_be_overwritten_by_wav(tmp_path):
    score=source(track("a",0,0,note()))
    path=tmp_path/"score.wav"
    old=b"my score path content"
    path.write_bytes(old)
    with pytest.raises(Exception):
        export_multitrack_preview(score,path,config(patch("a")),
                                  source_path=path,overwrite=True)
    assert path.read_bytes()==old


def test_cli_multi_instrument_wav_and_no_clobber(tmp_path,capsys):
    score=two_tracks()
    source_file=tmp_path/"source.json"
    output=tmp_path/"original.wav"
    source_file.write_bytes(encode_score(score))
    argv=["multitrack-preview-wav",str(source_file),"--output",str(output),
          "--sample-rate","8000","--patch","melody:sine:0.10",
          "--patch","bass:triangle:0.10"]
    assert main(argv)==0
    receipt=json.loads(capsys.readouterr().out)
    assert receipt["preview"]["render"]["rendered_notes"]==2
    assert read_wav_file(output).sample_rate==8000
    assert main(argv)==2
    assert capsys.readouterr().out==""


def test_cli_rejects_percussion_without_explicit_optin(tmp_path,capsys):
    score=source(track("melody",0,0,note()),
                 track("drum",9,0,note("kick",36)))
    infile=tmp_path/"source.json"
    out=tmp_path/"out.wav"
    infile.write_bytes(encode_score(score))
    argv=["multitrack-preview-wav",str(infile),"--output",str(out),
          "--patch","melody:sine:0.1","--sample-rate","8000"]
    assert main(argv)==2
    assert not out.exists()
    capsys.readouterr()
    assert main(argv+["--allow-skip-percussion"])==0
    report=json.loads(capsys.readouterr().out)
    assert report["preview"]["render"]["skipped_percussion_notes"]==1
