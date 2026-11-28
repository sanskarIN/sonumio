"""Score-clock PCM and spectral diagnostics against independent numeric oracles."""
from dataclasses import replace
import hashlib
import json
import math

import numpy as np
import pytest

from sonumio.arrangement import ArrangementPlan,compose_arrangement
from sonumio.audio import SAMPLE_RATE,encode_wav_channels
from sonumio.errors import GenerationError
from sonumio.music_render_analysis import (
    AnalysisSettings,Moments,pcm_view,moments,spectral_bands,analyze_render,BLOCK,
)
from sonumio.music_render_compare import PairMoments,compare_interval,compare_renders
from sonumio.music_render_analysis_cli import main
from sonumio.music_mix import generate_arrangement_audio
from sonumio.music_score_io import encode_score


@pytest.fixture(scope="module")
def score():
    return compose_arrangement(ArrangementPlan(bars=4,bpm=137,swing=.4,seed=17))


@pytest.fixture(scope="module")
def wav(score):
    return generate_arrangement_audio(score)[0]


def test_exact_integer_moments_are_independent_of_block_partition():
    rng=np.random.default_rng(17)
    values=rng.integers(-32768,32768,(BLOCK+17,2),dtype=np.int16)
    direct=moments(values,0,len(values),AnalysisSettings())
    a=moments(values,0,71,AnalysisSettings())
    b=moments(values,71,len(values),AnalysisSettings())
    a.merge(b)
    assert a.report()==direct.report()
    x=values.astype(np.int64)
    report=direct.report()
    assert report["integer"]["sums"]==x.sum(axis=0).tolist()
    assert report["integer"]["sum_squares"]==(x*x).sum(axis=0).tolist()
    assert report["integer"]["cross_product"]==int((x[:,0]*x[:,1]).sum())
    assert report["max_adjacent_step"]==np.abs(np.diff(x,axis=0)).max()/32768.


def test_mono_retention_and_side_energy_have_exact_extremes():
    x=np.array([100,-200,300,-400],dtype=np.int16)
    same=np.stack((x,x),axis=1)
    inverse=np.stack((x,-x),axis=1)
    a=moments(same,0,4,AnalysisSettings()).report()
    b=moments(inverse,0,4,AnalysisSettings()).report()
    assert a["mono_rms_retention"]==1
    assert a["side_energy_fraction"]==0
    assert a["centered_stereo_correlation"]==1
    assert b["mono_rms_retention"]==0
    assert b["side_energy_fraction"]==1
    assert b["centered_stereo_correlation"]==-1


def test_dc_correlation_is_stable_for_large_nearly_constant_pcm():
    values=np.full((20000,2),30000,dtype=np.int16)
    values[7,0]+=1
    values[7,1]-=1
    report=moments(values,0,len(values),AnalysisSettings()).report()
    assert report["centered_stereo_correlation"]==-1


def test_silence_has_defined_counts_and_undefined_ratios():
    values=np.zeros((2048,2),dtype=np.int16)
    report=moments(values,0,len(values),AnalysisSettings()).report()
    assert report["rms"]==0
    assert report["rms_dbfs"] is None
    assert report["mono_rms_retention"] is None
    assert report["centered_stereo_correlation"] is None
    spectrum=spectral_bands(values,0,len(values))
    assert spectrum["spectral_centroid_hz"] is None
    assert all(v is None for v in spectrum["band_fraction"].values())
    json.dumps(report,allow_nan=False)


@pytest.mark.parametrize("frequency,band",[(46.875,"sub"),(187.5,"bass"),(375,"low_mid"),
    (937.5,"mid"),(3750,"presence"),(7500,"high")])
def test_fft_bands_use_channel_energy_not_mono_cancellation(frequency,band):
    t=np.arange(8192)/SAMPLE_RATE
    x=(10000*np.sin(2*np.pi*frequency*t)).round().astype(np.int16)
    values=np.stack((x,-x),axis=1)
    report=spectral_bands(values,0,len(values))
    assert report["band_fraction"][band]>.9
    assert sum(report["band_fraction"].values())==pytest.approx(1.)
    assert report["spectral_centroid_hz"]==pytest.approx(frequency,abs=10)


def test_full_analysis_bars_and_sections_cover_exact_sample_clock(score,wav):
    report=analyze_render(score,wav)
    assert report["wav_sha256"]==hashlib.sha256(wav).hexdigest()
    assert report["frames"]==score.plan.frame(score.plan.ticks)
    assert report["source_provenance_verified"] is False
    assert len(report["bars"])==score.plan.bars
    assert sum(b["signal"]["frames"] for b in report["bars"])==report["frames"]
    for bar,row in enumerate(report["bars"]):
        assert row["start_frame"]==score.plan.frame(bar*32)
        assert row["stop_frame"]==score.plan.frame((bar+1)*32)
        assert row["harmony_degree"]==score.harmony[bar]
    assert sum(s["signal"]["frames"] for s in report["sections"])==report["frames"]
    assert sum(b["note_onsets"] for b in report["bars"])==len(score.events)
    full=moments(pcm_view(score,wav),0,report["frames"],AnalysisSettings()).report()
    assert report["signal"]==full
    assert sum(report["spectrum"]["band_fraction"].values())==pytest.approx(1.)
    assert analyze_render(score,wav)==report


def test_rail_dc_mono_and_activity_findings_are_explicit(score):
    frames=score.plan.frame(score.plan.ticks)
    signal=np.zeros((2,frames),dtype=np.float32)
    signal[0,:1000]=1.
    signal[1,:1000]=-1.
    report=analyze_render(score,encode_wav_channels(signal,SAMPLE_RATE))
    codes={f["code"] for f in report["findings"]}
    assert "pcm_rail_samples" in codes and "low_mono_retention" in codes
    assert any(f["code"]=="low_activity_with_score_events" for row in report["bars"] for f in row["findings"])


def test_pair_gain_fit_is_exact_and_not_applied():
    a=np.array([[100,-200],[300,400],[-500,600]],dtype=np.int16)
    b=(a*2).astype(np.int16)
    report=compare_interval(a,b,0,len(a)).report()
    assert report["least_squares_gain"]==2.
    assert report["gain_fit_residual_mse"]==0.
    assert report["relative_wave_mse"]==1.
    assert report["rms_ratio"]==2.
    assert report["waveform_cosine"]==1.
    assert report["different_samples"]==6


def test_pair_inversion_and_silence_have_distinct_semantics():
    a=np.array([[100,-200],[300,400]],dtype=np.int16)
    b=-a
    report=compare_interval(a,b,0,2).report()
    assert report["least_squares_gain"]==-1
    assert report["relative_wave_mse"]==4
    assert report["waveform_cosine"]==-1
    silent=np.zeros_like(a)
    quiet=compare_interval(silent,a,0,2).report()
    assert quiet["relative_wave_mse"] is None
    assert quiet["least_squares_gain"] is None
    assert quiet["absolute_wave_mse"]>0


def test_pair_merged_partitions_match_direct_oracle():
    rng=np.random.default_rng(42)
    a=rng.integers(-32768,32768,(BLOCK+29,2),dtype=np.int16)
    b=rng.integers(-32768,32768,a.shape,dtype=np.int16)
    left=compare_interval(a,b,0,61)
    left.merge(compare_interval(a,b,61,len(a)))
    direct=compare_interval(a,b,0,len(a)).report()
    assert left.report()==direct
    error=(a.astype(np.int64)-b.astype(np.int64))
    assert direct["integer"]["error_sum_squares"]==int((error*error).sum())
    assert direct["different_samples"]==int((error!=0).sum())


def test_same_wav_comparison_is_exact_identity(score,wav):
    report=compare_renders(score,wav,wav)
    assert report["paired"]["absolute_wave_mse"]==0
    assert report["paired"]["different_samples"]==0
    assert report["paired"]["identical_frames"]==score.plan.frame(score.plan.ticks)
    assert report["paired"]["waveform_cosine"]==1.
    assert report["changes"]["rms_db_delta"]==0
    assert report["alignment_optimized"] is False
    assert all(row["paired"]["different_samples"]==0 for row in report["bars"])


@pytest.mark.parametrize("changes",[
    {"quiet_dbfs":True},{"quiet_dbfs":-121},{"maximum_dc":float("nan")},
    {"minimum_active_fraction":1.1},{"maximum_boundary_step":-1},
])
def test_invalid_settings(changes):
    with pytest.raises(GenerationError):
        AnalysisSettings(**changes)


def test_malformed_wav_or_wrong_score_geometry_rejected(score,wav):
    with pytest.raises(GenerationError):
        pcm_view(score,wav[:-4])
    altered=bytearray(wav)
    altered[22]=1
    with pytest.raises(ValueError):
        pcm_view(score,bytes(altered))
    changed=replace(score,plan=replace(score.plan,bpm=120))
    with pytest.raises(GenerationError):
        pcm_view(changed,wav)


def test_cli_verified_pin_and_comparison(score,wav,tmp_path,capsys):
    source=tmp_path/"score.json"
    audio=tmp_path/"audio.wav"
    source.write_bytes(encode_score(score))
    audio.write_bytes(wav)
    args=[str(source),str(audio)]
    assert main(["inspect",*args,"--expected-source-sha256","0"*64])==2
    assert main(["inspect",*args,"--expected-source-sha256",hashlib.sha256(wav).hexdigest()])==0
    assert json.loads(capsys.readouterr().out)["source_geometry_verified"]
    assert main(["compare",*args,str(audio)])==0
    assert json.loads(capsys.readouterr().out)["paired"]["different_samples"]==0
