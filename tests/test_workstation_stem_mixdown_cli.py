"""Hash-pinned stem mix recipe, WAV publication and CLI regression tests."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
import types
import wave

import numpy as np
import pytest

from sonumio.midi_workstation.contracts import MidiFormatError,MidiIntegrityError,MidiWorkstationError
from sonumio.midi_workstation.publish import PublicationOptions,publish_bytes
from sonumio.midi_workstation.stem_mixdown import StemMixGain,StemMixSettings
from sonumio.midi_workstation.stem_mixdown_cli import (
    RECIPE_PROTOCOL,encode_mix_recipe,decode_mix_recipe,main,
)
from test_workstation_stem_package import fixture_pack


def setup(tmp_path):
    source=tmp_path/"package.zip"
    content=fixture_pack()
    source.write_bytes(content)
    settings=StemMixSettings((StemMixGain("alpha"),StemMixGain("beta")))
    recipe=tmp_path/"mix.json"
    recipe.write_bytes(encode_mix_recipe(settings,sha256(content).hexdigest()))
    return source,recipe,settings


def call(args,capsys):
    value=main([str(x) for x in args])
    captured=capsys.readouterr()
    return value,captured


@pytest.fixture
def wav_publisher(monkeypatch):
    package=types.ModuleType("sonumio.audio_core")
    package.__path__=[]
    contracts=types.ModuleType("sonumio.audio_core.contracts")
    dither=types.ModuleType("sonumio.audio_core.dither")
    waveio=types.ModuleType("sonumio.audio_core.waveio")
    errors=types.ModuleType("sonumio.audio_core.errors")
    atomic=types.ModuleType("sonumio.atomic_cli_wav")
    for name in ("AudioContractError","AudioIntegrityError","AudioLimitError","AudioProcessingError"):
        setattr(errors,name,type(name,(Exception,),{}))
    @dataclass(frozen=True)
    class QuantizationSettings:
        bits:int
        dither:bool
        seed:int|None
    dither.QuantizationSettings=QuantizationSettings
    @dataclass(frozen=True)
    class Audio:
        samples:np.ndarray
        sample_rate:int
        @property
        def frames(self):return self.samples.shape[1]
        @property
        def channels(self):return self.samples.shape[0]
    contracts.make_buffer=lambda samples,rate:Audio(samples,rate)
    def encode_wav_bytes(audio,*,settings):
        size=settings.bits//8
        # Quantization is irrelevant for CLI routing; encode zeros at the
        # requested exact shape with the same PCM RIFF header contract.
        payload=b"\0"*(audio.channels*audio.frames*size)
        stream=BytesIO()
        with wave.open(stream,"wb") as wav:
            wav.setnchannels(audio.channels)
            wav.setsampwidth(size)
            wav.setframerate(audio.sample_rate)
            wav.writeframes(payload)
        return stream.getvalue()
    waveio.encode_wav_bytes=encode_wav_bytes
    def publish_generated_wav(destination,blob,*,overwrite,create_parent,expected_sha256):
        result=publish_bytes(destination,blob,PublicationOptions(
            overwrite=overwrite,create_parent=create_parent,
            expected_sha256=expected_sha256),expected_extension=".wav")
        return types.SimpleNamespace(wav_sha256=result.output_sha256,
                                     bytes_written=result.bytes_written)
    atomic.publish_generated_wav=publish_generated_wav
    for module in (package,contracts,dither,waveio,errors,atomic):
        monkeypatch.setitem(sys.modules,module.__name__,module)


def test_canonical_recipe_is_digest_pinned_and_reversible(tmp_path):
    source,recipe,settings=setup(tmp_path)
    loaded,pin=decode_mix_recipe(recipe.read_bytes())
    assert loaded==settings
    assert pin==sha256(source.read_bytes()).hexdigest()
    assert encode_mix_recipe(loaded,pin)==recipe.read_bytes()
    assert json.loads(recipe.read_bytes())["protocol"]==RECIPE_PROTOCOL


def test_read_only_mix_plan_has_no_publication(tmp_path,capsys):
    source,recipe,settings=setup(tmp_path)
    value,output=call(["plan",source,"--recipe",recipe],capsys)
    assert value==0,output.err
    receipt=json.loads(output.out)
    assert receipt["mix"]["frames"]==4064
    assert receipt["mix"]["track_count"]==2
    assert receipt["wav_published"] is False
    assert sorted(p.name for p in tmp_path.iterdir())==["mix.json","package.zip"]


def test_default_unity_recipe_can_be_published_and_imported(tmp_path,capsys):
    source,_,_=setup(tmp_path)
    destination=tmp_path/"unity.json"
    code,output=call(["unity-recipe",source,"--output",destination],capsys)
    assert code==0,output.err
    assert destination.exists()
    assert json.loads(output.out)["tracks"]==["alpha","beta"]
    options,pin=decode_mix_recipe(destination.read_bytes())
    assert [g.gain_milli for g in options.gains]==[1000,1000]
    assert pin==sha256(source.read_bytes()).hexdigest()


def test_cli_renders_pcm_and_inspects_by_wave_reader(tmp_path,capsys,wav_publisher):
    source,recipe,settings=setup(tmp_path)
    output=tmp_path/"master.wav"
    code,report=call(["render",source,"--recipe",recipe,"--output",output],capsys)
    assert code==0,report.err
    data=json.loads(report.out)
    assert data["render"]["wav_sha256"]==sha256(output.read_bytes()).hexdigest()
    with wave.open(str(output),"rb") as wav:
        assert (wav.getnframes(),wav.getnchannels(),wav.getframerate())==(4064,2,8000)
        assert wav.getsampwidth()==2
    assert data["render"]["model_inference"] is False


def test_cli_rejects_wrong_input_recipe_digest(tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    output=tmp_path/"master.wav"
    code,result=call(["render",source,"--recipe",recipe,"--output",output,
                       "--expected-recipe-sha256","0"*64],capsys)
    assert code==2
    assert not output.exists()
    assert not result.out


def test_cli_detects_stale_pack_digest(tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    source.write_bytes(source.read_bytes()+b"suffix")
    code,result=call(["render",source,"--recipe",recipe,"--output",tmp_path/"mix.wav"],capsys)
    assert code==2
    assert "does not pin" in result.err
    assert not result.out


def test_cli_output_digest_mismatch_prevents_publication(tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    output=tmp_path/"mix.wav"
    code,result=call(["render",source,"--recipe",recipe,"--output",output,
                       "--expected-output-sha256","0"*64],capsys)
    assert code==2
    assert not output.exists()
    assert not result.out


def test_no_implicit_overwrite_of_existing_mix(tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    destination=tmp_path/"mix.wav"
    args=["render",source,"--recipe",recipe,"--output",destination]
    assert call(args,capsys)[0]==0
    before=destination.read_bytes()
    code,out=call(args,capsys)
    assert code==2
    assert destination.read_bytes()==before
    assert call(args+["--overwrite"],capsys)[0]==0
    assert destination.read_bytes()==before


@pytest.mark.parametrize("file",["package.zip","mix.json"])
def test_source_or_recipe_can_never_be_replaced_by_mixdown(file,tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    destination=tmp_path/file
    original=destination.read_bytes()
    code,out=call(["render",source,"--recipe",recipe,"--output",destination,
                   "--overwrite"],capsys)
    assert code==2
    assert not out.out
    assert destination.read_bytes()==original


def test_missing_directory_requires_explicit_opt_in(tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    output=tmp_path/"missing"/"mix.wav"
    assert call(["render",source,"--recipe",recipe,"--output",output],capsys)[0]==2
    assert not output.parent.exists()
    code,out=call(["render",source,"--recipe",recipe,"--output",output,
                   "--create-parent"],capsys)
    assert code==0,out.err
    assert output.exists()


def test_wrong_recipe_order_fails_closed():
    settings=StemMixSettings((StemMixGain("alpha"),StemMixGain("beta")))
    data=encode_mix_recipe(settings,"f"*64)
    doc=json.loads(data)
    doc["gains"].reverse()
    from sonumio.midi_workstation.contracts import canonical_json
    with pytest.raises(MidiWorkstationError):
        decode_mix_recipe(canonical_json(doc))


@pytest.mark.parametrize("bad",[b"",b"{}",b"null",b"{}\n",b"[{\"x\":1}]",
                                 b"{\"foo\":1}",b"not-json",b"{\"x\":NaN}"])
def test_malformed_recipe_rejected(bad):
    with pytest.raises((MidiFormatError,MidiWorkstationError)):
        decode_mix_recipe(bad)


def test_noncanonical_recipe_bytes_rejected():
    settings=StemMixSettings((StemMixGain("alpha"),))
    encoded=encode_mix_recipe(settings,"e"*64)
    indented=json.dumps(json.loads(encoded),indent=2).encode()
    with pytest.raises(MidiFormatError,match="canonical"):
        decode_mix_recipe(indented)


def test_cli_never_claims_ai_model_or_perceptual_quality(tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    code,out=call(["render",source,"--recipe",recipe,
                   "--output",tmp_path/"mix.wav"],capsys)
    assert code==0,out.err
    result=json.loads(out.out)
    assert result["render"]["mix"]["trained_model_inference"] is False
    assert result["render"]["mix"]["perceptual_quality_established"] is False
    assert result["remote_publication"] is False


def test_symlink_recipe_is_rejected_before_publication(tmp_path,capsys,wav_publisher):
    source,recipe,_=setup(tmp_path)
    link=tmp_path/"alias.json"
    try:link.symlink_to(recipe)
    except (NotImplementedError,OSError):pytest.skip("No symlink privilege")
    output=tmp_path/"mix.wav"
    code,out=call(["render",source,"--recipe",link,"--output",output],capsys)
    assert code==2
    assert not output.exists()
    assert not out.out
