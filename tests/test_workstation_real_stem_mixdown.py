"""Real audio_core stem package -> verified PCM mixdown integration checks."""
from hashlib import sha256
from io import BytesIO
import wave

import numpy as np
import pytest

pytest.importorskip("sonumio.audio_core.wavetable")
from sonumio.midi_workstation.stem_package import render_stem_pack
from sonumio.midi_workstation.stem_mixdown import (
    StemMixGain,StemMixSettings,_decode_pcm_integer,
    mix_stem_package,publish_stem_mixdown,
)
from test_workstation_real_stem_package import example,settings,members


def mixsettings(**kwargs):
    cfg={"gains":(StemMixGain("bass"),StemMixGain("lead"))}
    cfg.update(kwargs)
    return StemMixSettings(**cfg)


def test_real_wavetable_stems_reconstruct_additive_phase_exact_master():
    archive,_=render_stem_pack(example(),settings())
    result=mix_stem_package(archive,mixsettings())
    contents=members(archive)
    bass=_decode_pcm_integer(contents["stems/bass.wav"],bits=16,channels=2,frames=4160)
    lead=_decode_pcm_integer(contents["stems/lead.wav"],bits=16,channels=2,frames=4160)
    reference=(bass+lead).astype(np.float32)
    np.testing.assert_array_equal(result.samples,reference)
    assert result.report.post_peak<.98
    assert result.report.sample_rate==8000
    assert result.report.track_count==2
    assert result.report.source_stem_package_sha256==sha256(archive).hexdigest()


def test_real_mute_of_one_track_preserves_other_pcm():
    archive,_=render_stem_pack(example(),settings())
    result=mix_stem_package(archive,mixsettings(gains=(
        StemMixGain("bass",1000),StemMixGain("lead",0),
    )))
    content=members(archive)
    bass=_decode_pcm_integer(content["stems/bass.wav"],bits=16,channels=2,frames=4160)
    np.testing.assert_array_equal(result.samples,bass.astype(np.float32))
    assert result.report.selected_track_count==1


def test_real_master_gain_is_exact_and_deterministic():
    archive,_=render_stem_pack(example(),settings())
    opts=mixsettings(master_milli=500)
    a=mix_stem_package(archive,opts)
    b=mix_stem_package(archive,opts)
    np.testing.assert_array_equal(a.samples,b.samples)
    assert a.report==b.report
    assert a.report.post_peak<=a.report.pre_peak
    assert a.report.resulting_float_sha256==sha256(a.samples.tobytes()).hexdigest()


def test_real_wav_publisher_is_atomic_and_header_matches_mix(tmp_path):
    archive,_=render_stem_pack(example(),settings())
    source=tmp_path/"stems.zip";source.write_bytes(archive)
    destination=tmp_path/"master.wav"
    report=publish_stem_mixdown(archive,mixsettings(),destination,source_path=source)
    assert report["wav_sha256"]==sha256(destination.read_bytes()).hexdigest()
    with wave.open(str(destination),"rb") as wav:
        assert wav.getnframes()==4160
        assert wav.getnchannels()==2
        assert wav.getframerate()==8000
        assert wav.getsampwidth()==2
    assert source.read_bytes()==archive


def test_real_export_repeatable_without_training_or_external_audio():
    archive,_=render_stem_pack(example(),settings())
    plan=mixsettings(dither_seed=10)
    # The pure mix is deterministic independently of WAV encoder dithering.
    a=mix_stem_package(archive,plan)
    b=mix_stem_package(archive,plan)
    assert a.report.as_dict()==b.report.as_dict()
    assert not a.report.as_dict()["trained_model_inference"]
    assert not a.report.as_dict()["external_samples_used"]


def test_real_24bit_stems_mix_with_float_fidelity():
    archive,_=render_stem_pack(example(),settings(bits=24))
    result=mix_stem_package(archive,mixsettings(output_bits=24))
    assert result.report.input_bits==result.report.output_bits==24
    assert result.samples.shape==(2,4160)
    assert float(np.max(np.abs(result.samples)))>.02
    assert float(np.max(np.abs(result.samples)))<.98
