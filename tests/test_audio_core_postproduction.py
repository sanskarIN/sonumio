"""End-to-end local editing chain, exact receipts and verified file output."""
import hashlib
import json
import numpy as np
import pytest

from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.errors import AudioContractError, AudioIntegrityError, AudioLimitError
from sonumio.audio_core.hashing import digest_audio, canonical_json_bytes, checksum_record
from sonumio.audio_core.time_scale import TimeScaleSettings, stretch_audio
from sonumio.audio_core.loop_render import LoopRenderPlan, render_loop
from sonumio.audio_core.loop_analysis import LoopSearchPlan
from sonumio.audio_core.mastering import MasteringSettings, master_audio
from sonumio.audio_core.postproduction import (
    PostProductionPlan, run_postproduction, verify_postproduction_manifest,
)
from sonumio.audio_core.postproduction_io import parse_postproduction_plan, export_postproduction
from sonumio.audio_core.waveio import encode_wav_bytes, read_wav_file
from sonumio.cli import main


def source():
    t=np.arange(8000)/24000
    x=(.15*np.sin(2*np.pi*440*t)).astype(np.float32)
    return make_buffer(np.stack((x,-x)),24000)


def plan():
    return PostProductionPlan(duration_ratio=1.25,
        time_scale=TimeScaleSettings(window_frames=512,search_frames=64),
        loop=LoopRenderPlan(100,4000,64,repeats=2),
        mastering=MasteringSettings(target_rms_dbfs=-25,estimated_peak_ceiling_dbfs=-2))


def test_pipeline_matches_sequential_reference_without_intermediate_quantization():
    audio=source()
    config=plan()
    transformed=stretch_audio(audio,10000,settings=config.time_scale).audio
    looped=render_loop(transformed,config.loop).audio
    expected=master_audio(looped,config.mastering).audio
    result=run_postproduction(audio,config)
    np.testing.assert_array_equal(result.audio.samples,expected.samples)
    assert result.audio.spec.layout==audio.spec.layout
    assert [x.operation for x in result.stages]==["duration","loop","master"]
    assert result.stages[0].input_pcm_sha256==digest_audio(audio)
    assert result.stages[0].output_pcm_sha256==result.stages[1].input_pcm_sha256
    assert result.stages[1].output_pcm_sha256==result.stages[2].input_pcm_sha256
    manifest=result.manifest()
    verified=verify_postproduction_manifest(manifest,expected_payload=digest_audio(expected))
    assert verified["output_frames"]==2*(3900-64)
    assert not verified["intermediate_quantization"]
    json.dumps(manifest,allow_nan=False)


def test_pitch_search_and_master_pipeline():
    audio=source()
    config=PostProductionPlan(pitch_semitones=5,
        time_scale=TimeScaleSettings(method="phase_vocoder",window_frames=512),
        loop_search=LoopSearchPlan(100,4000,start_radius=4,stop_radius=16,
                                  context_frames=64,blend_frames=32),
        search_render_repeats=2,mastering=MasteringSettings(target_rms_dbfs=None))
    result=run_postproduction(audio,config)
    assert [x.operation for x in result.stages]==["pitch","loop","master"]
    assert result.stages[0].output_frames==audio.frames
    found=result.stages[1].details["search"]["candidates"][0]
    assert result.audio.frames==2*found["period_frames"]
    verify_postproduction_manifest(result.manifest())


def test_source_and_output_pin_guards():
    audio=source()
    config=plan()
    result=run_postproduction(audio,config)
    again=run_postproduction(audio,config,expected_source_pcm_sha256=digest_audio(audio),
        expected_output_pcm_sha256=result.output_pcm_sha256)
    assert again.manifest()==result.manifest()
    with pytest.raises(AudioIntegrityError,match="Source PCM"):
        run_postproduction(audio,config,expected_source_pcm_sha256="0"*64)
    with pytest.raises(AudioIntegrityError,match="Output PCM"):
        run_postproduction(audio,config,expected_output_pcm_sha256="0"*64)


def test_rehashed_broken_receipt_chain_rejected():
    result=run_postproduction(source(),plan())
    record=result.manifest()
    payload=record.pop("payload_sha256")
    record.pop("record_sha256")
    record["stages"][1]["input_pcm_sha256"]="0"*64
    tampered=checksum_record(record,payload_sha256=payload)
    with pytest.raises(AudioIntegrityError,match="chain"):
        verify_postproduction_manifest(tampered)


def test_plan_json_roundtrip():
    config=plan()
    loaded=parse_postproduction_plan(canonical_json_bytes(config.as_dict()))
    assert config==loaded
    assert config.sha256==loaded.sha256


@pytest.mark.parametrize("raw",[
    b'{}',b'{"duration_ratio":1,"duration_frames":4}',
    b'{"duration_ratio":1,"duration_ratio":2}',b'{"duration_ratio":NaN}',
    b'{"duration_ratio":1,"unknown":true}',
    b'{"duration_ratio":1,"time_scale":{"unknown":1}}',
    b'{"mastering":{"peak":{"factor":3}}}',
    b'{"loop_search":{"start_frame":0,"stop_frame":10},"search_render_repeats":0}',
])
def test_strict_plan_json(raw):
    with pytest.raises((AudioContractError,TypeError)):
        parse_postproduction_plan(raw)


def test_preflight_rejects_invalid_transformed_geometry():
    audio=source()
    config=PostProductionPlan(duration_ratio=.5,loop=LoopRenderPlan(0,5000))
    with pytest.raises(AudioContractError,match="transformed"):
        run_postproduction(audio,config)


def test_pipeline_intermediate_byte_cap(monkeypatch):
    import sonumio.audio_core.postproduction as module
    monkeypatch.setattr(module,"MAX_PIPELINE_PCM_BYTES",source().spec.byte_count)
    with pytest.raises(AudioLimitError,match="intermediate"):
        run_postproduction(source(),PostProductionPlan(duration_ratio=1.5))


def test_verified_file_export(tmp_path):
    input_path=tmp_path/"input.wav"
    input_path.write_bytes(encode_wav_bytes(source()))
    source_pin=hashlib.sha256(input_path.read_bytes()).hexdigest()
    destination=tmp_path/"output.wav"
    report=export_postproduction(input_path,destination,plan=plan(),output_bits=24,
                                  expected_source_sha256=source_pin,dither_seed=17)
    assert report.output_sha256==hashlib.sha256(destination.read_bytes()).hexdigest()
    assert report.output_bytes==destination.stat().st_size
    assert report.output_bits==24
    assert report.encoded_peak is not None
    assert read_wav_file(destination).frames==report.manifest["output_frames"]
    verify_postproduction_manifest(report.manifest)
    assert not report.as_dict()["intermediate_files_written"]
    assert set(tmp_path.iterdir())=={input_path,destination}


def test_failed_pin_preserves_destination(tmp_path):
    input_path=tmp_path/"input.wav"
    input_path.write_bytes(encode_wav_bytes(source()))
    destination=tmp_path/"output.wav"
    destination.write_bytes(b"existing")
    with pytest.raises(AudioIntegrityError,match="Output PCM"):
        export_postproduction(input_path,destination,plan=plan(),
            expected_output_pcm_sha256="0"*64,overwrite=True)
    assert destination.read_bytes()==b"existing"


def test_cli_plan_and_primary_commands(tmp_path,capsys):
    input_path=tmp_path/"input.wav"
    input_path.write_bytes(encode_wav_bytes(source()))
    plan_path=tmp_path/"plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan().as_dict()))
    destination=tmp_path/"output.wav"
    assert main(["wav-postproduce",str(input_path),"--plan",str(plan_path),
                 "--output",str(destination)])==0
    report=json.loads(capsys.readouterr().out)
    assert report["manifest"]["plan_sha256"]==plan().sha256
    assert main(["master-inspect",str(destination)])==0
    assert json.loads(capsys.readouterr().out)["source_verified"]
    assert main(["loop-search",str(destination),"--start-frame","100",
                 "--stop-frame","1000","--start-radius","0","--stop-radius","0"])==0
    assert json.loads(capsys.readouterr().out)["search"]["coarse_pairs"]==1


def test_cli_invalid_json_does_not_publish(tmp_path,capsys):
    plan_path=tmp_path/"plan.json"
    plan_path.write_text('{"duration_ratio":1,"unknown":0}',encoding="utf-8")
    destination=tmp_path/"output.wav"
    assert main(["wav-postproduce",str(tmp_path/"missing.wav"),"--plan",str(plan_path),
                 "--output",str(destination)])==2
    assert "Unexpected fields" in capsys.readouterr().err
    assert not destination.exists()
