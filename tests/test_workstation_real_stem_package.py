"""Real finite-Fourier source-build tests for sample-aligned stem ZIP export.

These cases exercise the original Sonumio audio engine on GitHub CI. They are
skipped only when an isolated review bundle lacks the main audio_core package.
"""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import wave
import zipfile

import numpy as np
import pytest

pytest.importorskip("sonumio.audio_core.wavetable")
from sonumio.midi_workstation.score import Score,ScoreNote,ScoreTrack
from sonumio.midi_workstation.stem_package import (
    StemPatch, StemPackagePlan, preflight_stems, render_stem_pack,
    inspect_stem_pack, publish_stem_pack,
)
from sonumio.midi_workstation.publish import PublicationOptions


def example():
    score=Score.empty(end=480)
    return replace(score,tracks=(
        ScoreTrack("bass","Bass",0,0,0,0,
             (ScoreNote("bass1",0,240,48,velocity=100),)),
        ScoreTrack("lead","Lead",1,0,0,0,
             (ScoreNote("lead1",240,240,72,velocity=90),)),
    ))


def settings(*, seed=None, bits=16):
    return StemPackagePlan((
        StemPatch("bass", waveform="sine", gain=.08,
                  attack_frames=0,decay_frames=0,sustain_level=1.,release_frames=0),
        StemPatch("lead", waveform="triangle", gain=.08,
                  attack_frames=0,decay_frames=0,sustain_level=1.,release_frames=160),
    ),sample_rate=8000,channels=2,bits=bits,dither_seed=seed)


def members(binary):
    with zipfile.ZipFile(BytesIO(binary)) as package:
        return {name:package.read(name) for name in package.namelist()}


def decode_pcm(blob):
    with wave.open(BytesIO(blob),"rb") as stream:
        assert stream.getsampwidth()==2
        channels=stream.getnchannels()
        count=stream.getnframes()
        rate=stream.getframerate()
        pcm=np.frombuffer(stream.readframes(count),dtype="<i2")
    return pcm.reshape(count,channels).astype(np.float64)/32768.,rate


def test_real_audio_engine_renders_distinct_valid_wavs():
    source=example()
    data,report=render_stem_pack(source,settings())
    assert report["stems"]==2
    assert report["frames"]==4160
    wavs=members(data)
    assert sorted(wavs)==["manifest.json","stems/bass.wav","stems/lead.wav"]
    a,rate_a=decode_pcm(wavs["stems/bass.wav"])
    b,rate_b=decode_pcm(wavs["stems/lead.wav"])
    assert rate_a==rate_b==8000
    assert a.shape==b.shape==(4160,2)
    assert np.max(np.abs(a[:2000]))>.01
    assert np.max(np.abs(b[2000:4000]))>.01
    assert np.max(np.abs(a[4000:]))==0
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    assert report["source_score_sha256"]==source.sha256


def test_real_audio_stems_keep_original_patch_releases_not_longer():
    data,_=render_stem_pack(example(),settings())
    wavs=members(data)
    first,_=decode_pcm(wavs["stems/bass.wav"])
    second,_=decode_pcm(wavs["stems/lead.wav"])
    # Bass note ends at tick240 -> frame2000. No tail requested.
    assert not np.any(first[2000:])
    # Lead's final note tail exists up to its own release end.
    assert not np.any(second[:2000])
    assert second.shape[0]==4160


def test_real_audio_export_has_deterministic_digest():
    source=example()
    first,one=render_stem_pack(source,settings())
    second,two=render_stem_pack(source,settings())
    assert first==second
    assert one==two
    assert one["package_sha256"]==sha256(first).hexdigest()
    assert inspect_stem_pack(first)==one


def test_seeded_dither_is_deterministic_but_distinct_for_tracks():
    source=example()
    plain,_=render_stem_pack(source,settings(seed=None))
    a,_=render_stem_pack(source,settings(seed=12))
    b,_=render_stem_pack(source,settings(seed=12))
    assert a==b
    assert plain!=a
    assert members(a)["stems/bass.wav"]!=members(plain)["stems/bass.wav"]


def test_complete_source_file_remains_unchanged_on_publication(tmp_path):
    source=example()
    before=source.sha256
    archive=tmp_path/"result.zip"
    receipt=publish_stem_pack(source,settings(),archive)
    assert archive.is_file()
    assert source.sha256==before
    assert receipt["source_unchanged"] is True
    with pytest.raises(Exception):
        publish_stem_pack(source,settings(),archive)
    assert receipt["package_sha256"]==sha256(archive.read_bytes()).hexdigest()


def test_real_archive_read_back_validates_wav_duration_and_pcm_depth():
    source=example()
    packed,_=render_stem_pack(source,settings(bits=24))
    report=inspect_stem_pack(packed)
    assert report["frames"]==preflight_stems(source,settings(bits=24)).frames
    assert report["bits"]==24
    wavs=members(packed)
    for name in ("stems/bass.wav","stems/lead.wav"):
        with wave.open(BytesIO(wavs[name])) as recording:
            assert recording.getnframes()==4160
            assert recording.getsampwidth()==3
            assert recording.getnchannels()==2
            assert recording.getframerate()==8000
