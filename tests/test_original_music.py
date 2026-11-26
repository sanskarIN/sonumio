"""Original symbolic music: timing, harmonic rules, replay and output integrity."""
import hashlib
import io
import json
import wave
import numpy as np
import pytest

from sonumio.music import (
    MusicPlan, Note, Score, compose, render, generate_music, score_digest,
    MODES, STYLES, TRACKS, SAMPLE_RATE,
)
from sonumio.music_cli import main
from sonumio.errors import GenerationError


@pytest.mark.parametrize("bpm",[60,73,100,137,160])
def test_absolute_tick_clock(bpm):
    plan=MusicPlan(bpm=bpm,bars=8)
    frames=[plan.frame(i) for i in range(plan.ticks+1)]
    assert frames[0]==0
    assert frames[-1]==round(8*4*60*SAMPLE_RATE/bpm)
    assert all(b>a for a,b in zip(frames,frames[1:]))
    assert all(abs(x-i*60*SAMPLE_RATE/(bpm*8))<=.5 for i,x in enumerate(frames))


@pytest.mark.parametrize("style",STYLES)
@pytest.mark.parametrize("mode",tuple(MODES))
@pytest.mark.parametrize("root",[48,60,71])
def test_harmonic_and_timing_contract(style,mode,root):
    plan=MusicPlan(style,root,mode,137,4,29)
    score=compose(plan)
    assert score.harmony[-1]==0
    assert score.harmony[-2]==4
    pcs={(root+x)%12 for x in MODES[mode]}
    lead=[]
    for n in score.notes:
        assert 0<=n.start<n.start+n.duration<=plan.ticks
        if n.track in ("pad","bass","lead"):
            assert n.pitch%12 in pcs
        if n.track=="lead":
            lead.append(n)
    assert lead[-1].pitch%12==root%12
    assert score_digest(score)==score_digest(compose(plan))
    assert all(n.start+n.duration<=m.start for n,m in zip(lead,lead[1:]))


@pytest.mark.parametrize("style",STYLES)
def test_stereo_render_determinism_and_health(style):
    plan=MusicPlan(style=style,bars=1,bpm=137,seed=9)
    score=compose(plan)
    a=render(score)
    b=render(score)
    assert a.shape==(2,plan.frame(plan.ticks))
    assert a.dtype==np.float32
    assert np.array_equal(a,b)
    assert np.isfinite(a).all()
    assert 0.01<float(np.sqrt(np.mean(a*a)))<.8
    assert np.max(np.abs(a))<=.900001
    assert np.all(a[:,0]==0)
    assert np.max(np.abs(a[:,-1]))<1e-8
    assert not np.array_equal(a[0],a[1])
    assert score_digest(score)!=score_digest(compose(MusicPlan(style=style,bars=1,bpm=137,seed=10)))


def test_wav_metadata_and_digest():
    data,report=generate_music(MusicPlan(style="rhythmic",bars=2,seed=11))
    with wave.open(io.BytesIO(data),"rb") as handle:
        assert handle.getnchannels()==2
        assert handle.getframerate()==24000
        assert handle.getsampwidth()==2
        assert handle.getnframes()==report["frames"]
    assert report["wav_sha256"]==hashlib.sha256(data).hexdigest()
    assert report["events"]==sum(report["track_counts"].values())
    assert not report["learned_model"]
    assert not report["external_recordings_used"]
    assert not report["perceptual_quality_established"]
    assert report["metrics"]["clipping_fraction"]==0


@pytest.mark.parametrize("kwargs",[
    {"style":"unknown"},{"mode":[]},{"bpm":0},{"bpm":True},{"root":72},
    {"bars":9},{"seed":-1},{"seed":2**32},
])
def test_invalid_plans(kwargs):
    with pytest.raises(GenerationError):
        MusicPlan(**kwargs)


def test_cli_publication_rejects_clobber_and_bad_digest(tmp_path,capsys):
    output=tmp_path/"music.wav"
    args=["--style","calm","--bars","1","--seed","31","--output",str(output)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    old=output.read_bytes()
    assert hashlib.sha256(old).hexdigest()==report["wav_sha256"]
    with pytest.raises(SystemExit) as err:
        main(args)
    assert err.value.code==2
    assert output.read_bytes()==old
    with pytest.raises(SystemExit):
        main(args+["--overwrite","--expected-output-sha256","0"*64])
    assert output.read_bytes()==old
    assert not list(tmp_path.glob("*.partial"))
