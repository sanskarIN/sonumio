"""Blockwise automation equivalence, peak replay and bounded PCM contracts."""
import hashlib
import io
import json
import wave

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan, TRACKS, arrangement_digest, compose_arrangement
from sonumio.cli import main
from sonumio.errors import GenerationError
from sonumio.music_automation import AutomationPlan, MixKey, MixLane, encode_automation, generate_automated_music
from sonumio.music_blockwise import _encode_replayed_blocks, generate_blockwise_music
from sonumio.music_blockwise_automation import generate_blockwise_automated_music, iter_automated_wet_blocks
from sonumio.music_mix import MixPlan
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(style="rhythmic",bpm=160,bars=4,seed=71,swing=.35))


def automation(mix=None):
    lanes=(
        MixLane("keys","gain",(MixKey(0,.2,"linear"),MixKey(17,1.3,"step"),
                               MixKey(48,.7,"smoothstep"),MixKey(128,1.))),
        MixLane("lead","gain",(MixKey(3,0.,"step"),MixKey(31,2.,"smoothstep"),MixKey(63,.4))),
        MixLane("lead","balance",(MixKey(0,-1.,"linear"),MixKey(29,1.,"step"),MixKey(64,0.))),
        MixLane("hat","balance",(MixKey(0,.25),)),
    )
    return AutomationPlan(lanes,MixPlan() if mix is None else mix)


@pytest.mark.parametrize("block",[256,997,4096,8192,16384])
def test_automation_pcm_matches_full_reference_across_block_sizes(score,block):
    plan=automation()
    reference,expected=generate_automated_music(score,plan)
    actual,report=generate_blockwise_automated_music(score,plan,block_frames=block)
    assert actual==reference
    assert report["wav_sha256"]==expected["wav_sha256"]
    for key in ("score_sha256","automation_sha256","automation_lanes","frames","channels",
                "sample_rate","metrics","pre_master_peak","master_attenuation"):
        assert report[key]==expected[key],key
    for track in TRACKS:
        a,b=report["tracks"][track],expected["tracks"][track]
        assert a.keys()==b.keys()
        for key in a:
            if key=="automated_rms":
                assert a[key]==pytest.approx(b[key],abs=1e-14)
            else:
                assert a[key]==b[key]
    assert report["replay_verified"]
    assert not report["retains_full_float_stems"]
    assert report["retains_final_wav_bytes"]


@pytest.mark.parametrize("mix",[
    MixPlan(mute=TRACKS),
    MixPlan(solo=("lead",),room=0.),
    MixPlan(gains=(2.,)*8,room=1.,peak_ceiling=.05),
    MixPlan(mute=("lead","hat"),room=.3),
])
def test_automation_respects_solo_mute_room_and_peak(score,mix):
    plan=automation(mix)
    reference,expected=generate_automated_music(score,plan)
    actual,report=generate_blockwise_automated_music(score,plan,block_frames=1301)
    assert actual==reference
    assert report["master_attenuation"]==expected["master_attenuation"]
    for track in TRACKS:
        if not mix.active(track):
            assert report["tracks"][track]["automated_peak"]==0.
            assert report["tracks"][track]["automated_rms"]==0.
    with wave.open(io.BytesIO(actual),"rb") as wav:
        pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype="<i2")
    assert np.max(np.abs(pcm.astype(np.int32)))<=round(mix.peak_ceiling*32767)


def test_empty_automation_matches_static_blockwise_renderer(score):
    mix=MixPlan(room=.2,peak_ceiling=.8)
    a,ar=generate_blockwise_music(score,mix,block_frames=997)
    b,br=generate_blockwise_automated_music(score,AutomationPlan((),mix),block_frames=997)
    assert a==b
    assert ar["unscaled_stream_sha256"]==br["unscaled_stream_sha256"]
    assert ar["metrics"]==br["metrics"]


def test_partition_invariance_of_unscaled_automated_stream(score):
    plan=automation()
    outputs=[]
    for block in (997,4096):
        chunks=[]
        cursor=0
        for start,wet in iter_automated_wet_blocks(score,plan,block):
            assert start==cursor
            assert wet.dtype==np.float64 and wet.shape[0]==2
            assert wet.shape[1]<=block
            chunks.append(wet)
            cursor+=wet.shape[1]
        assert cursor==score.plan.frame(score.plan.ticks)
        outputs.append(np.concatenate(chunks,axis=1))
    np.testing.assert_array_equal(*outputs)


def test_invalid_automation_rejected_before_synthesis(score,monkeypatch):
    import sonumio.music_blockwise_automation as module
    def forbidden(*args,**kwargs):
        pytest.fail("Invalid plan must not synthesize")
    monkeypatch.setattr(module,"iter_stem_blocks",forbidden)
    invalid=AutomationPlan((MixLane("lead","gain",(MixKey(score.plan.ticks+1,1.),)),))
    with pytest.raises(GenerationError,match="beyond"):
        generate_blockwise_automated_music(score,invalid)


def test_shared_encoder_rejects_invalid_block_shapes_and_offsets():
    cases=(
        lambda first:iter(((1,np.zeros((2,4),dtype=np.float64)),)),
        lambda first:iter(((0,np.zeros((1,4),dtype=np.float64)),)),
        lambda first:iter(((0,np.zeros((2,4),dtype=np.float32)),)),
        lambda first:iter(((0,np.zeros((2,0),dtype=np.float64)),)),
        lambda first:iter(((0,np.zeros((2,5),dtype=np.float64)),)),
        lambda first:iter(((0,np.full((2,4),np.nan,dtype=np.float64)),)),
    )
    for factory in cases:
        with pytest.raises(GenerationError,match="PCM stream"):
            _encode_replayed_blocks(4,.9,factory)


def test_shared_encoder_detects_truncation_in_both_passes():
    with pytest.raises(GenerationError,match="Incomplete"):
        _encode_replayed_blocks(4,.9,lambda first:iter(((0,np.zeros((2,3),dtype=np.float64)),)))
    def factory(first):
        yield 0,np.zeros((2,4 if first else 3),dtype=np.float64)
    with pytest.raises(GenerationError,match="Incomplete"):
        _encode_replayed_blocks(4,.9,factory)


def test_shared_encoder_detects_second_pass_change():
    def factory(first):
        yield 0,np.zeros((2,4),dtype=np.float64) if first else np.ones((2,4),dtype=np.float64)*.001
    with pytest.raises(GenerationError,match="replay"):
        _encode_replayed_blocks(4,.9,factory)


def test_long_score_does_not_call_full_float_stem_render(monkeypatch):
    import sonumio.music_automation as full
    def forbidden(*args,**kwargs):
        pytest.fail("Full-song floating stems are not allowed")
    monkeypatch.setattr(full,"render_stems",forbidden)
    score=compose_arrangement(ArrangementPlan(bars=24,bpm=120,seed=97))
    plan=AutomationPlan((MixLane("lead","gain",(MixKey(0,.1),MixKey(score.plan.ticks,1.))),))
    wav,report=generate_blockwise_automated_music(score,plan,block_frames=8192)
    assert report["frames"]==48*24000
    assert len(wav)==44+48*24000*4
    assert report["render_passes"]==2


def test_cli_blockwise_flag_preserves_atomic_publication(score,tmp_path,capsys):
    source=tmp_path/"score.json"
    automation_path=tmp_path/"mix.json"
    output=tmp_path/"render.wav"
    plan=automation()
    source.write_bytes(encode_score(score))
    automation_path.write_bytes(encode_automation(plan))
    args=["music-automate",str(source),str(automation_path),"--output",str(output),
          "--expected-score-sha256",arrangement_digest(score),
          "--expected-automation-sha256",plan.sha256,"--block-frames","997"]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    original=output.read_bytes()
    assert report["output_sha256"]==hashlib.sha256(original).hexdigest()
    assert report["renderer_protocol"]=="sonumio-blockwise-automation-v1"
    assert original==generate_automated_music(score,plan)[0]
    assert main(args)==2
    assert main(args+["--overwrite","--expected-output-sha256","0"*64])==2
    assert main(args+["--overwrite","--block-frames","255"])==2
    assert output.read_bytes()==original
