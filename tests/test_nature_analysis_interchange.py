"""Independent numerical oracles for nature descriptors, JSON and controls."""
from dataclasses import replace
import hashlib
import json
import math
import wave
import numpy as np
import pytest

from sonumio.nature_analysis import describe,compare,BANDS
from sonumio.nature_scene import preset,scene_digest
from sonumio.nature_scene_io import encode_scene,decode_scene,MAX_SCENE_BYTES
from sonumio.nature_cli import main
from sonumio.errors import GenerationError
from research.nature_sound_study import NatureStudyPlan,run_study


def tone(hz,amplitude=.2,frames=24000):
    return (amplitude*np.sin(2*np.pi*hz*np.arange(frames)/24000)).astype(np.float32)


@pytest.mark.parametrize("hz,band",[(93.75,0),(375,1),(1500,2),(3000,3),(6000,4),(9000,5)])
def test_spectral_band_oracle(hz,band):
    x=tone(hz)
    result=describe(x)
    assert result["rms"]==pytest.approx(.2/math.sqrt(2),rel=.002)
    assert result["centroid_hz"]==pytest.approx(hz,abs=2)
    assert sum(result["band_energy_fractions"])==pytest.approx(1.)
    assert result["band_energy_fractions"][band]>.95
    assert result["spectral_frames"]==(len(x)-1024)//512+1
    assert result["peak"]<=.20000001
    assert result["overrange_fraction"]==0


def test_silent_metrics_are_explicitly_undefined():
    data=np.zeros(24000,dtype=np.float32)
    value=describe(data)
    assert value["rms"]==value["peak"]==0
    assert value["centroid_hz"] is None
    assert value["crest_factor"] is None
    assert value["spectral_flatness"] is None
    assert value["band_energy_fractions"]==[None]*6
    assert value["active_envelope_fraction"]==0
    assert value["active_envelope_regions"]==0
    assert value["envelope_energy_centroid_seconds"] is None
    c=compare(data,data)
    assert c["waveform_mae"]==0
    assert c["band_distribution_l1"] is None
    assert c["centered_waveform_correlation"] is None
    json.dumps(value,allow_nan=False)


def test_compare_identity_polarity_and_amplitude_oracles():
    a=tone(3000)
    same=compare(a,a)
    opposite=compare(a,-a)
    scaled=compare(a,a*.5)
    assert same["waveform_mae"]==0
    assert same["centered_waveform_correlation"]==pytest.approx(1)
    assert same["band_distribution_l1"]==0
    assert opposite["centered_waveform_correlation"]==pytest.approx(-1)
    assert opposite["band_distribution_l1"]==0
    assert scaled["band_distribution_l1"]<1e-12
    assert scaled["rms_delta"]==pytest.approx(-same["a"]["rms"]/2)


def test_envelope_activity_and_time_center():
    x=np.zeros(48000,dtype=np.float32)
    x[24000:36000]=tone(440,frames=12000)
    result=describe(x)
    assert result["active_envelope_fraction"]==pytest.approx(.25)
    assert result["active_envelope_regions"]==1
    assert result["envelope_energy_centroid_seconds"]==pytest.approx(1.25,abs=.01)


@pytest.mark.parametrize("data",[
    np.zeros(100,dtype=np.float32),np.zeros((2,24000),dtype=np.float32),
    np.zeros(24000,dtype=np.int16),np.full(24000,float("nan"),dtype=np.float32),
    np.full(24000,17,dtype=np.float32),
])
def test_diagnostic_rejects_bad_geometry(data):
    with pytest.raises(GenerationError):
        describe(data)


@pytest.mark.parametrize("name",["forest","storm","coast","creek","campfire"])
def test_scene_interchange_roundtrip(name):
    scene=preset(name,2.,17)
    data=encode_scene(scene)
    assert decode_scene(data,expected_sha256=scene_digest(scene))==scene
    assert encode_scene(decode_scene(data))==data
    with pytest.raises(GenerationError):
        decode_scene(data,expected_sha256="0"*64)


@pytest.mark.parametrize("raw",[
    b"",b"null",b"[]",b'{"a":1,"a":2}',b'{"x":Infinity}',b"\xff",
    b" "*(MAX_SCENE_BYTES+1),
])
def test_reject_malformed_scene_bytes(raw):
    with pytest.raises(GenerationError):
        decode_scene(raw)


@pytest.mark.parametrize("change",[
    lambda v:v.update(format="wrong"),
    lambda v:v.update(sha256="0"*64),
    lambda v:v["scene"].update(extra=True),
    lambda v:v["scene"].update(layers=[]),
    lambda v:v["scene"]["layers"][0]["voice"].update(seed=True),
    lambda v:v["scene"]["layers"][0].update(pan=[]),
])
def test_reject_tampered_scene_fields(change):
    obj=json.loads(encode_scene(preset("storm",2.,17)))
    change(obj)
    with pytest.raises(GenerationError):
        decode_scene(json.dumps(obj).encode())


def test_cli_preset_and_verified_scene_render_agree(tmp_path,capsys):
    scene=tmp_path/"nature.json"
    first=tmp_path/"preset.wav"
    second=tmp_path/"render.wav"
    assert main(["plan","storm","--seconds","2","--seed","17","--output",str(scene)])==0
    report=json.loads(capsys.readouterr().out)
    digest=report["scene_sha256"]
    assert report["output_sha256"]==hashlib.sha256(scene.read_bytes()).hexdigest()
    assert main(["generate","storm","--seconds","2","--seed","17","--output",str(first)])==0
    a=json.loads(capsys.readouterr().out)
    assert main(["render",str(scene),"--expected-scene-sha256",digest,"--output",str(second)])==0
    b=json.loads(capsys.readouterr().out)
    assert first.read_bytes()==second.read_bytes()
    assert a["output_sha256"]==b["output_sha256"]
    with wave.open(str(second),"rb") as wav:
        assert wav.getnframes()==48000 and wav.getnchannels()==2 and wav.getframerate()==24000
    with pytest.raises(SystemExit):
        main(["render",str(scene),"--expected-scene-sha256","0"*64,"--output",str(tmp_path/"bad.wav")])
    assert not (tmp_path/"bad.wav").exists()


@pytest.mark.parametrize("kwargs",[
    {"seeds":(1,)},{"seeds":(1,1)},{"seconds":float("nan")},{"seconds":1},
    {"kinds":("unknown",)},{"kinds":("rain","rain")},{"kinds":[]},
    {"seeds":tuple(range(8)),"seconds":10.},
])
def test_study_budget(kwargs):
    with pytest.raises(ValueError):
        NatureStudyPlan(**kwargs)


def test_paired_nature_control_study():
    plan=NatureStudyPlan((17,29),2.,("rain","birds"))
    report=run_study(plan)
    assert report["all_seeded_replays_exact"]
    assert report["all_zero_intensity_silent"]
    assert report["all_categories_vary_by_seed"]
    assert report["render_frame_budget"]==plan.frames*2*20
    assert not report["baseline_comparison_loudness_matched"]
    for category in report["categories"]:
        assert len(category["runs"])==2
        assert len(category["pairwise_seed_comparisons"])==1
        assert category["distinct_baseline_sha256"]==2
        assert category["intensity_rms_increase_seeds"]==2
    json.dumps(report,allow_nan=False)
