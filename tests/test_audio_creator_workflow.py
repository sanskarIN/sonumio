"""Cross-feature regression of the original audio creator workflow.

The tests intentionally use public APIs across multiple modules, pin
semantic checksums and verify the final stereo PCM byte geometry.
"""
from dataclasses import replace
import hashlib
import io
import wave

import numpy as np
import pytest

from sonumio.audio_prompt import parse_prompt,build_source
from sonumio.music_edit import Retime,Transpose,RemoveTrack,TrackArticulation,edit
from sonumio.music_score_io import encode_score,decode_score
from sonumio.music_midi import encode_midi
from sonumio.music_analysis import analyze_score,analyze_audio
from sonumio.arrangement import arrangement_digest
from sonumio.nature_scene import scene_digest
from sonumio.nature_scene_io import encode_scene,decode_scene
from sonumio.soundtrack import Placement,Ducking,SoundtrackPlan,generate_soundtrack
from sonumio.music_mix import MixPlan
from sonumio.errors import GenerationError


def test_bilingual_creator_workflow_has_exact_replay():
    music=build_source(parse_prompt("संगीत सिनेमैटिक माइनर कुंजी C गति 120 बार 4 सीड 17"))
    edited,receipt=edit(music,(Retime(160,.15),Transpose(1),
                                TrackArticulation("lead",0,4,.8,-.1),
                                RemoveTrack("hat",1,3)))
    assert receipt["source_score_sha256"]==arrangement_digest(music)
    assert receipt["result_score_sha256"]==arrangement_digest(edited)
    original_bytes=encode_score(music)
    score_bytes=encode_score(edited)
    replay=decode_score(score_bytes,expected_sha256=receipt["result_score_sha256"])
    assert replay==edited and encode_score(music)==original_bytes
    midi=encode_midi(replay)
    assert midi.startswith(b"MThd") and midi==encode_midi(edited)
    score_report=analyze_score(replay)
    assert score_report["final_lead_tonic"]
    assert score_report["strong_beat_chord_hits"]==score_report["strong_beat_notes"]

    nature=build_source(parse_prompt("प्रकृति जंगल सेकंड 2 सीड 29"))
    nature_bytes=encode_scene(nature)
    scene=decode_scene(nature_bytes,expected_sha256=scene_digest(nature))
    plan=SoundtrackPlan(48000,Placement(12000,0,48000,-3,240,480),
                        Placement(0,0,48000,-6,240,480),
                        MixPlan(mute=("hat",)),Ducking(True))
    wav,report=generate_soundtrack(replay,scene,plan)
    same,other=generate_soundtrack(edited,nature,plan)
    assert wav==same and report==other
    assert report["score_sha256"]==receipt["result_score_sha256"]
    assert report["scene_sha256"]==scene_digest(nature)
    assert report["wav_sha256"]==hashlib.sha256(wav).hexdigest()
    assert report["metrics"]["clipping_fraction"]==0
    assert not report["learned_model"]
    with wave.open(io.BytesIO(wav),"rb") as reader:
        assert reader.getnchannels()==2 and reader.getframerate()==24000
        assert reader.getsampwidth()==2 and reader.getnframes()==48000
        pcm=np.frombuffer(reader.readframes(48000),dtype="<i2").reshape(-1,2)
    stereo=(pcm.T.astype(np.float32)/32768)
    measured=analyze_audio(stereo)
    assert measured["frames"]==48000
    assert max(measured["channel_peak"])<=.9001
    assert max(measured["channel_rms"])>0
    assert measured["centered_stereo_correlation"] is not None
    assert len(measured["timeline"])==4


def test_checksum_and_geometry_failure_do_not_change_sources():
    music=build_source(parse_prompt("music calm bars 4 bpm 160 seed 17"))
    nature=build_source(parse_prompt("nature creek seconds 2 seed 29"))
    a=encode_score(music)
    b=encode_scene(nature)
    with pytest.raises(GenerationError):
        decode_score(a,expected_sha256="0"*64)
    with pytest.raises(GenerationError):
        decode_scene(b,expected_sha256="0"*64)
    plan=SoundtrackPlan(48000,Placement(0,0,48000),Placement(0,0,48000))
    with pytest.raises(GenerationError):
        generate_soundtrack(music,nature,replace(plan,music=Placement(143999,0,48000)))
    assert encode_score(music)==a and encode_scene(nature)==b


def test_score_edits_cannot_exceed_music_duration_budget():
    score=build_source(parse_prompt("music rhythmic bars 32 bpm 160 seed 17"))
    assert score.plan.frame(score.plan.ticks)==1152000
    before=arrangement_digest(score)
    with pytest.raises(GenerationError):
        edit(score,(Retime(60),))
    assert arrangement_digest(score)==before
