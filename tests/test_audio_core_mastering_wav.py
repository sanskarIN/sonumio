"""Verified inspection, encoded peak policy and no-clobber mastering export."""
import hashlib
import json
import os
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.dither import QuantizationSettings
from sonumio.audio_core.waveio import encode_wav_bytes, read_wav_file
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError, AudioProcessingError
from sonumio.audio_core.interpolated_peak import PeakEstimateSettings, interpolation_kernels
from sonumio.audio_core.mastering import MasteringSettings, measure_mastering
from sonumio.audio_core.mastering_wav import (
    MasteringWavOptions, inspect_mastering_wav, master_wav_file,
    encode_mastered_audio, quantization_peak_reserve,
)
from sonumio.audio_core.mastering_cli import main


@pytest.fixture
def source(tmp_path):
    t=np.arange(9601)/24000
    x=(.19*np.sin(2*np.pi*440*t)).astype(np.float32)
    x[4800]=.9
    data=encode_wav_bytes(make_buffer(np.stack((x,-x)),24000),
                          settings=QuantizationSettings(bits=24))
    path=tmp_path/"source.wav"
    path.write_bytes(data)
    return path,hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("packet_frames",[17,1024,8192])
def test_streaming_inspection_matches_decoded_reference(source,packet_frames):
    path,pin=source
    report=inspect_mastering_wav(path,expected_sha256=pin,packet_frames=packet_frames)
    expected=measure_mastering(read_wav_file(path))
    assert report.source_sha256==pin
    assert report.source_bits==24
    assert report.source_bytes==path.stat().st_size
    assert report.measurement.energy.gated_rms_dbfs==expected.energy.gated_rms_dbfs
    assert report.measurement.peak.estimated_peak==pytest.approx(expected.peak.estimated_peak,abs=1e-13)
    assert report.as_dict()["streaming_measurement"]


@pytest.mark.parametrize("mode",["static","lookahead"])
@pytest.mark.parametrize("bits",[8,16,24,32])
def test_mastered_integer_pcm_reverified(source,tmp_path,mode,bits):
    path,pin=source
    settings=MasteringSettings(target_rms_dbfs=-15,max_boost_db=12,
                               estimated_peak_ceiling_dbfs=-2,mode=mode)
    options=MasteringWavOptions(settings,output_bits=bits,dither_seed=75)
    target=tmp_path/"result.wav"
    result=master_wav_file(path,target,options=options,expected_sha256=pin)
    data=target.read_bytes()
    assert result.output_sha256==hashlib.sha256(data).hexdigest()
    assert result.output_bytes==len(data) and result.output_bits==bits
    assert result.source["source_sha256"]==pin
    measurement=inspect_mastering_wav(target,expected_sha256=result.output_sha256)
    assert measurement.source_bits==bits
    assert measurement.measurement.peak.estimated_peak==pytest.approx(result.encoded_measurement.peak.estimated_peak,abs=1e-13)
    assert measurement.measurement.peak.estimated_peak<=settings.ceiling
    assert result.as_dict()["encoded_peak_reverified"]
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
    json.dumps(result.as_dict(),allow_nan=False)


def test_quantization_reserve_bounds_direct_kernel_error():
    rng=np.random.default_rng(21)
    peak=PeakEstimateSettings(factor=8,radius=24)
    for bits in (8,16,24,32):
        for dither in (False,True):
            q=QuantizationSettings(bits=bits,dither=dither,seed=1 if dither else None)
            reserve=quantization_peak_reserve(q,peak)
            bound=(1.5 if dither else 1)/(1 << (bits-1))
            for kernel in interpolation_kernels(peak):
                error=rng.uniform(-bound,bound,size=len(kernel))
                assert abs(float(np.dot(error,kernel)))<=reserve
    assert quantization_peak_reserve(QuantizationSettings(bits=8),peak)>quantization_peak_reserve(QuantizationSettings(bits=16),peak)


def test_encoding_guard_deterministic_second_attempt():
    # One positive sample just below the first 8-bit half-step rounds up.
    # The selected ceiling lies between the original and encoded values.
    level=10.6/128
    ceiling=10.7/128
    x=make_buffer(np.array([level],dtype=np.float32),8000)
    settings=MasteringSettings(target_rms_dbfs=None,
        estimated_peak_ceiling_dbfs=20*np.log10(ceiling))
    q=QuantizationSettings(bits=8)
    with pytest.raises(AudioProcessingError,match="Encoded WAV exceeds"):
        encode_mastered_audio(x,quantization=q,settings=settings,policy="reject")
    a=encode_mastered_audio(x,quantization=q,settings=settings)
    b=encode_mastered_audio(x,quantization=q,settings=settings)
    assert a.attempts==2 and a.attenuation_db<0
    assert a.quantization_peak_reserve>0
    assert a.measurement.peak.estimated_peak<=settings.ceiling
    assert a.wav==b.wav


def test_peak_guard_rejects_non_normalized_audio():
    audio=make_buffer(np.full(32,1.1,dtype=np.float32),8000)
    with pytest.raises(AudioContractError,match="normalized"):
        encode_mastered_audio(audio,quantization=QuantizationSettings(),
                              settings=MasteringSettings())


def test_deterministic_bytes_across_reader_partition(source,tmp_path):
    from dataclasses import replace
    path,pin=source
    cfg=MasteringSettings(max_boost_db=8,packet_frames=29)
    a=master_wav_file(path,tmp_path/"a.wav",options=MasteringWavOptions(cfg,dither_seed=1),
                      expected_sha256=pin)
    b=master_wav_file(path,tmp_path/"b.wav",
        options=MasteringWavOptions(replace(cfg,packet_frames=8192),dither_seed=1),
        expected_sha256=pin,expected_output_sha256=a.output_sha256)
    assert a.output_sha256==b.output_sha256


def test_bad_source_pin_no_destination_side_effect(source,tmp_path):
    path,_=source
    output=tmp_path/"absent"/"result.wav"
    with pytest.raises(AudioIntegrityError):
        master_wav_file(path,output,expected_sha256="0"*64,create_parent=True)
    assert not output.parent.exists()
    with pytest.raises(AudioIntegrityError):
        inspect_mastering_wav(path,expected_sha256="0"*64)


def test_existing_output_and_pin_rollback(source,tmp_path):
    path,pin=source
    output=tmp_path/"existing.wav"
    output.write_bytes(b"existing")
    with pytest.raises(AudioContractError):
        master_wav_file(path,output)
    with pytest.raises(AudioIntegrityError):
        master_wav_file(path,output,overwrite=True,expected_output_sha256="0"*64)
    assert output.read_bytes()==b"existing"
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin


def test_source_path_and_hardlink_alias(source,tmp_path):
    path,pin=source
    with pytest.raises(AudioContractError,match="must differ"):
        master_wav_file(path,path,overwrite=True)
    alias=tmp_path/"alias.wav"
    try:
        os.link(path,alias)
    except OSError:
        pytest.skip("hardlinks unsupported")
    with pytest.raises(AudioContractError,match="file identity"):
        master_wav_file(path,alias,overwrite=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==pin


@pytest.mark.parametrize("kwargs",[
    {"mastering":object()},{"output_bits":12},{"dither_seed":-1},
    {"encoded_peak_policy":"ignore"},
])
def test_invalid_options(kwargs):
    with pytest.raises(AudioContractError):
        MasteringWavOptions(**kwargs)


def test_cli_inspect_and_master(source,tmp_path,capsys):
    path,pin=source
    assert main(["master-inspect",str(path),"--expected-sha256",pin])==0
    inspect=json.loads(capsys.readouterr().out)
    assert inspect["source_sha256"]==pin
    target=tmp_path/"master.wav"
    assert main(["wav-master",str(path),"--output",str(target),"--peak-only",
                 "--peak-ceiling-dbfs","-3","--output-bits","16"])==0
    exported=json.loads(capsys.readouterr().out)
    assert exported["output_bits"]==16
    assert exported["encoded_target_error_db"] is None
    assert target.exists()


def test_cli_rejects_invalid_source_pin(source,tmp_path,capsys):
    path,_=source
    output=tmp_path/"bad.wav"
    assert main(["wav-master",str(path),"--output",str(output),
                 "--expected-sha256","0"*64])==2
    assert "SHA256" in capsys.readouterr().err
    assert not output.exists()


def test_cli_target_options_mutually_exclusive(source,tmp_path):
    path,_=source
    with pytest.raises(SystemExit) as exc:
        main(["wav-master",str(path),"--output",str(tmp_path/"bad.wav"),
              "--peak-only","--target-rms-dbfs","-20"])
    assert exc.value.code==2
