"""Bit-depth exact mixdown of pinned integer-PCM stem containers."""
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import struct
import wave

import numpy as np
import pytest

from sonumio.midi_workstation.contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
)
from sonumio.midi_workstation.stem_mixdown import (
    StemMixGain,StemMixSettings,_decode_pcm_integer,
    mix_stem_package,
)
from test_workstation_stem_package import fixture_pack,pcm,score,plan,pack_zip


def options(**change):
    args={"gains":(StemMixGain("alpha"),StemMixGain("beta"))}
    args.update(change)
    return StemMixSettings(**args)


def pcm_of(values, *, channels=1,rate=8000,bits=16):
    """Create exact, portable signed integer PCM examples with no DSP."""
    flat=np.asarray(values,dtype=np.int64).reshape(-1)
    if bits==8:
        payload=bytes((int(x)+128)&255 for x in flat)
    elif bits==16:
        payload=b"".join(struct.pack("<h",int(x)) for x in flat)
    elif bits==24:
        payload=b"".join((int(x)&0xffffff).to_bytes(3,"little") for x in flat)
    elif bits==32:
        payload=b"".join(struct.pack("<i",int(x)) for x in flat)
    else:raise ValueError(bits)
    buf=BytesIO()
    with wave.open(buf,"wb") as writer:
        writer.setnchannels(channels)
        writer.setframerate(rate)
        writer.setsampwidth(bits//8)
        writer.writeframes(payload)
    return buf.getvalue()


@pytest.mark.parametrize("bits,values,denominator",[
    (8,[-128,-1,0,1,127],128.),
    (16,[-32768,-1000,0,1000,32767],32768.),
    (24,[-8388608,-65535,0,65535,8388607],8388608.),
    (32,[-2147483648,-123456,0,123456,2147483647],2147483648.),
])
def test_pcm_bit_depth_roundtrip(bits,values,denominator):
    wav=pcm_of(values,bits=bits)
    output=_decode_pcm_integer(wav,bits=bits,channels=1,frames=len(values))
    assert output.shape==(1,len(values))
    assert output.dtype==np.float64
    np.testing.assert_allclose(output[0],np.asarray(values)/denominator,atol=0,rtol=0)


@pytest.mark.parametrize("bits",[8,16,24,32])
def test_pcm_channel_first_stereo_preserves_interleaving(bits):
    values=[(1,2),(3,4),(-1,-2),(0,0)]
    wav=pcm_of(values,channels=2,bits=bits)
    pcm=_decode_pcm_integer(wav,bits=bits,channels=2,frames=4)
    assert pcm.shape==(2,4)
    np.testing.assert_allclose(pcm[0,:3]/pcm[1,:3],np.array([.5,.75,.5]),rtol=0,atol=0)
    assert pcm[0,3]==pcm[1,3]==0


def test_equal_gains_sum_two_independent_silent_stems():
    result=mix_stem_package(fixture_pack(),options())
    assert result.samples.shape==(2,4064)
    assert result.samples.dtype==np.float32
    assert result.samples.flags.writeable is False
    assert result.report.pre_peak==0.
    assert result.report.applied_normalization==1.
    assert result.report.selected_track_count==2
    assert result.report.as_dict()["trained_model_inference"] is False


def test_zero_gain_is_explicit_mute_not_missing_stem():
    result=mix_stem_package(fixture_pack(),options(gains=(
        StemMixGain("alpha",0),StemMixGain("beta",1000),
    )))
    assert result.report.selected_track_count==1
    assert result.report.track_count==2
    assert not np.any(result.samples)


def test_gain_scaling_tracks_exact_expected_pcm_magnitude():
    wav_a=pcm_of([16000]*8,channels=1)
    wav_b=pcm_of([-8000]*8,channels=1)
    original=fixture_pack(a=wav_a,b=wav_b,frames=8,channels=1)
    result=mix_stem_package(original,options(gains=(
        StemMixGain("alpha",500),StemMixGain("beta",500),
    ),master_milli=1000))
    expected=(16000-8000)/32768.*.5
    assert result.samples.shape==(1,8)
    np.testing.assert_allclose(result.samples,expected,atol=1e-7)
    assert result.report.pre_peak==pytest.approx(expected)


def test_overlapping_full_level_tracks_refuse_clipping():
    wav=pcm_of([30000]*8,channels=1)
    original=fixture_pack(a=wav,b=wav,frames=8,channels=1)
    with pytest.raises(MidiWorkstationError,match="exceeds configured peak"):
        mix_stem_package(original,options())


def test_peak_normalization_is_explicit_and_receipted():
    wav=pcm_of([30000]*8,channels=1)
    original=fixture_pack(a=wav,b=wav,frames=8,channels=1)
    opts=options(peak_policy="normalize",target_peak_milli=900)
    result=mix_stem_package(original,opts)
    assert np.max(np.abs(result.samples))==pytest.approx(.9,abs=1e-7)
    assert result.report.pre_peak>1.
    assert result.report.post_peak==pytest.approx(.9)
    assert result.report.applied_normalization<1.
    assert result.report.peak_policy=="normalize"


def test_clipping_policy_is_not_implicit_normalization():
    wav=pcm_of([30000]*8,channels=1)
    original=fixture_pack(a=wav,b=wav,frames=8,channels=1)
    opts=options(peak_policy="reject",target_peak_milli=900)
    with pytest.raises(MidiWorkstationError):
        mix_stem_package(original,opts)


def test_mixdown_is_bitwise_deterministic_for_verified_package():
    archive=fixture_pack()
    one=mix_stem_package(archive,options())
    two=mix_stem_package(archive,options())
    np.testing.assert_array_equal(one.samples,two.samples)
    assert one.report==two.report
    assert one.report.resulting_float_sha256==sha256(one.samples.tobytes()).hexdigest()


def test_no_digest_mismatch_can_reach_decoder():
    data=fixture_pack(manifest_changes=lambda manifest:manifest["stems"][0].__setitem__("sha256","0"*64))
    with pytest.raises(MidiIntegrityError):
        mix_stem_package(data,options())


@pytest.mark.parametrize("kwargs",[
    {"gains":tuple()},
    {"gains":(StemMixGain("beta"),StemMixGain("alpha"))},
    {"gains":(StemMixGain("alpha"),StemMixGain("alpha"))},
    {"master_milli":-1},{"master_milli":2001},
    {"peak_policy":"silent-normalize"},
    {"target_peak_milli":0},{"target_peak_milli":1001},
    {"output_bits":9},{"dither_seed":-1},{"dither_seed":2**32},
    {"maximum_samples":0},{"maximum_samples":900001},
])
def test_settings_reject_bad_contracts(kwargs):
    with pytest.raises(MidiWorkstationError):
        options(**kwargs)


@pytest.mark.parametrize("bad",[
    {"track_id":"../bad"},{"track_id":""},
    {"gain_milli":-1},{"gain_milli":2001},
    {"gain_milli":True},
])
def test_track_gain_rejects_untrusted_values(bad):
    args={"track_id":"alpha"};args.update(bad)
    with pytest.raises(MidiWorkstationError):
        StemMixGain(**args)


def test_incomplete_gain_ledger_rejected():
    with pytest.raises(MidiWorkstationError,match="every stem"):
        mix_stem_package(fixture_pack(),options(gains=(StemMixGain("alpha"),)))


def test_extra_gain_track_is_also_rejected():
    with pytest.raises(MidiWorkstationError):
        mix_stem_package(fixture_pack(),options(gains=(
            StemMixGain("alpha"),StemMixGain("beta"),StemMixGain("missing"))))


def test_read_only_source_mixing_has_no_side_effects(tmp_path):
    path=tmp_path/"input.zip"
    data=fixture_pack()
    path.write_bytes(data)
    receipt=mix_stem_package(data,options()).report
    assert path.read_bytes()==data
    assert receipt.source_stem_package_sha256==sha256(data).hexdigest()
    assert sorted(tmp_path.iterdir())==[path]


def test_resource_ceiling_fails_before_allocation():
    with pytest.raises(MidiLimitError,match="sample budget"):
        mix_stem_package(fixture_pack(),options(maximum_samples=100))


def test_mismatched_pcm_header_never_mixes():
    wav_bad=pcm(frames=4064,channels=1)
    with pytest.raises(MidiFormatError):
        mix_stem_package(fixture_pack(a=wav_bad),options())
