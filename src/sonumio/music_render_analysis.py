"""Score-clock signal diagnostics for canonical stereo PCM16 renders.

This module reports measurable signal behavior, not musical preference,
perceived loudness, true peak, originality or provenance. A WAV supplied
with a score is checked for matching geometry only; callers must verify
a render receipt separately if they need content-linked provenance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import numpy as np

from .arrangement import Arrangement, TRACKS, arrangement_digest, TICKS_PER_BAR
from .audio import SAMPLE_RATE
from .audio_core.wave_reader import _decode_canonical_header
from .errors import GenerationError

PROTOCOL = "sonumio-score-clock-audio-diagnostics-v1"
MAX_FRAMES = 120*SAMPLE_RATE
MAX_WAV_BYTES = 44+4*MAX_FRAMES
BLOCK = 16384
FFT = 1024
HOP = 512
BANDS = (("sub",0,80),("bass",80,250),("low_mid",250,500),
         ("mid",500,2000),("presence",2000,5000),("high",5000,12001))


@dataclass(frozen=True,slots=True)
class AnalysisSettings:
    quiet_dbfs: float = -70.
    minimum_active_fraction: float = .001
    maximum_dc: float = .02
    maximum_boundary_step: float = .25
    minimum_mono_retention: float = .2

    def __post_init__(self):
        for name,low,high in (
            ("quiet_dbfs",-120.,-10.),("minimum_active_fraction",0.,1.),
            ("maximum_dc",0.,1.),("maximum_boundary_step",0.,2.),
            ("minimum_mono_retention",0.,1.),
        ):
            value=getattr(self,name)
            if type(value) not in (float,int) or not low<=value<=high:
                raise GenerationError("Invalid audio analysis setting: "+name)
            object.__setattr__(self,name,float(value))


def pcm_view(score,data):
    if not isinstance(score,Arrangement):
        raise GenerationError("Expected validated score for audio analysis")
    frames=score.plan.frame(score.plan.ticks)
    if (not 1<=frames<=MAX_FRAMES or type(data) is not bytes
            or len(data)!=44+4*frames):
        raise GenerationError("Analysis WAV length differs from score clock")
    try:
        geometry=_decode_canonical_header(data[:44],len(data))
    except ValueError as exc:
        raise GenerationError("Invalid canonical analysis WAV header") from exc
    if (geometry.sample_rate,geometry.channels,geometry.bits_per_sample,geometry.frames)!=(SAMPLE_RATE,2,16,frames):
        raise GenerationError("Analysis WAV geometry differs from score")
    return np.frombuffer(data,dtype="<i2",offset=44).reshape(frames,2)


def read_wav(path):
    with open(path,"rb") as stream:
        data=stream.read(MAX_WAV_BYTES+1)
    if not 44<len(data)<=MAX_WAV_BYTES:
        raise GenerationError("Invalid bounded analysis WAV size")
    return data


def _db(amplitude):
    return 20*math.log10(amplitude) if amplitude>0 else None


def _bounds(pcm,start,stop):
    if (not isinstance(pcm,np.ndarray) or pcm.dtype!=np.dtype("<i2")
            or pcm.ndim!=2 or pcm.shape[1]!=2 or not 1<=len(pcm)<=MAX_FRAMES
            or type(start) is not int or type(stop) is not int or not 0<=start<stop<=len(pcm)):
        raise GenerationError("Invalid audio analysis interval")


@dataclass(slots=True)
class Moments:
    frames:int=0
    left_sum:int=0
    right_sum:int=0
    left_squares:int=0
    right_squares:int=0
    cross:int=0
    left_peak:int=0
    right_peak:int=0
    clipped:int=0
    active:int=0
    maximum_step:int=0
    first:tuple[int,int]|None=None
    last:tuple[int,int]|None=None

    def add(self,block,threshold):
        if (not isinstance(block,np.ndarray) or block.dtype!=np.dtype("<i2")
                or block.ndim!=2 or block.shape[1]!=2 or not 1<=len(block)<=BLOCK):
            raise GenerationError("Invalid PCM moment block")
        values=block.astype(np.int64)
        left,right=values.T
        first=tuple(map(int,values[0]))
        last=tuple(map(int,values[-1]))
        if self.first is None:
            self.first=first
        if self.last is not None:
            self.maximum_step=max(self.maximum_step,max(abs(a-b) for a,b in zip(self.last,first)))
        if len(values)>1:
            self.maximum_step=max(self.maximum_step,int(np.max(np.abs(np.diff(values,axis=0)))))
        self.last=last
        self.frames+=len(values)
        self.left_sum+=int(left.sum())
        self.right_sum+=int(right.sum())
        self.left_squares+=int((left*left).sum())
        self.right_squares+=int((right*right).sum())
        self.cross+=int((left*right).sum())
        self.left_peak=max(self.left_peak,int(np.max(np.abs(left))))
        self.right_peak=max(self.right_peak,int(np.max(np.abs(right))))
        self.clipped+=int(((values==-32768)|(values==32767)).sum())
        self.active+=int((np.max(np.abs(values),axis=1)>threshold).sum())

    def merge(self,other):
        if not isinstance(other,Moments) or other.frames<1:
            raise GenerationError("Expected nonempty audio moments")
        if self.first is None:
            self.first=other.first
        if self.last is not None:
            self.maximum_step=max(self.maximum_step,max(abs(a-b) for a,b in zip(self.last,other.first)))
        self.last=other.last
        for name in ("frames","left_sum","right_sum","left_squares","right_squares","cross","clipped","active"):
            setattr(self,name,getattr(self,name)+getattr(other,name))
        for name in ("left_peak","right_peak","maximum_step"):
            setattr(self,name,max(getattr(self,name),getattr(other,name)))

    def report(self):
        if self.frames<1:
            raise GenerationError("Cannot report empty PCM moments")
        n=self.frames
        l=self.left_squares
        r=self.right_squares
        sum_energy=l+r
        # Centered covariance via exact integer numerators: no catastrophic
        # cancellation on large DC-biased, near-constant signals.
        covariance=n*self.cross-self.left_sum*self.right_sum
        left_variance=n*l-self.left_sum**2
        right_variance=n*r-self.right_sum**2
        corr=(max(-1.,min(1.,covariance/math.sqrt(left_variance*right_variance)))
              if left_variance>0 and right_variance>0 else None)
        mid=l+r+2*self.cross
        side=l+r-2*self.cross
        stereo_rms=math.sqrt(sum_energy/(2*n))/32768.
        return {
            "frames":n,"channel_peak":[self.left_peak/32768.,self.right_peak/32768.],
            "channel_rms":[math.sqrt(l/n)/32768.,math.sqrt(r/n)/32768.],
            "channel_dc":[self.left_sum/n/32768.,self.right_sum/n/32768.],
            "peak_dbfs":_db(max(self.left_peak,self.right_peak)/32768.),
            "rms_dbfs":_db(stereo_rms),"rms":stereo_rms,
            "centered_stereo_correlation":corr,
            "mono_rms_retention":math.sqrt(mid/(2*sum_energy)) if sum_energy else None,
            "side_energy_fraction":side/(2*sum_energy) if sum_energy else None,
            "clipped_samples":self.clipped,"active_fraction":self.active/n,
            "max_adjacent_step":self.maximum_step/32768.,
            "integer":{"sums":[self.left_sum,self.right_sum],"sum_squares":[l,r],
                       "cross_product":self.cross,"first":list(self.first),"last":list(self.last)},
        }


def moments(pcm,start,stop,settings):
    _bounds(pcm,start,stop)
    if not isinstance(settings,AnalysisSettings):
        raise GenerationError("Expected AnalysisSettings")
    threshold=32768.*10**(settings.quiet_dbfs/20.)
    result=Moments()
    for cursor in range(start,stop,BLOCK):
        result.add(pcm[cursor:min(cursor+BLOCK,stop)],threshold)
    return result


def spectral_bands(pcm,start,stop):
    """Summed windowed FFT energy, not a perceptual spectral measure.

    Only complete Hann windows are used (except short intervals, where
    one zero-padded window is used). FFT power combines channel energies
    before computing fractions, avoiding mono cancellation.
    """
    _bounds(pcm,start,stop)
    window=np.hanning(FFT)
    frequencies=np.fft.rfftfreq(FFT,1/SAMPLE_RATE)
    masks=[(frequencies>=low)&(frequencies<high) for _,low,high in BANDS]
    energy=np.zeros(len(BANDS),dtype=np.float64)
    centroid_numerator=0.
    total=0.
    windows=0
    starts=range(start,max(start+1,stop-FFT+1),HOP)
    for pos in starts:
        raw=pcm[pos:min(stop,pos+FFT)].astype(np.float64)/32768.
        if len(raw)<FFT:
            padded=np.zeros((FFT,2),dtype=np.float64)
            padded[:len(raw)]=raw
            raw=padded
        spectrum=np.fft.rfft(raw*window[:,None],axis=0)
        power=(spectrum.real**2+spectrum.imag**2).sum(axis=1)
        power[1:-1]*=2
        energy+=np.array([float(power[mask].sum()) for mask in masks])
        centroid_numerator+=float(np.dot(frequencies,power))
        total+=float(power.sum())
        windows+=1
    return {"fft":FFT,"hop":HOP,"windows":windows,
            "band_energy":{name:float(value) for (name,_,_),value in zip(BANDS,energy)},
            "band_fraction":{name:float(value/total) if total else None
                             for (name,_,_),value in zip(BANDS,energy)},
            "spectral_centroid_hz":centroid_numerator/total if total else None,
            "fft_energy":total}


def _merge_spectra(rows):
    total=sum(row["fft_energy"] for row in rows)
    sums={name:sum(row["band_energy"][name] for row in rows) for name,_,_ in BANDS}
    return {"fft":FFT,"hop":HOP,"windows":sum(row["windows"] for row in rows),
            "band_energy":sums,"band_fraction":{name:value/total if total else None for name,value in sums.items()},
            "spectral_centroid_hz":sum((row["spectral_centroid_hz"] or 0.)*row["fft_energy"]
                                        for row in rows)/total if total else None,
            "fft_energy":total}


def _findings(report,settings,*,boundary=None,expected_activity=False):
    findings=[]
    def add(code,measured,limit):
        findings.append({"code":code,"measured":measured,"limit":limit})
    if report["clipped_samples"]:
        add("pcm_rail_samples",report["clipped_samples"],0)
    dc=max(map(abs,report["channel_dc"]))
    if dc>settings.maximum_dc:
        add("dc_offset",dc,settings.maximum_dc)
    if expected_activity and report["active_fraction"]<settings.minimum_active_fraction:
        add("low_activity_with_score_events",report["active_fraction"],settings.minimum_active_fraction)
    retention=report["mono_rms_retention"]
    if retention is not None and retention<settings.minimum_mono_retention:
        add("low_mono_retention",retention,settings.minimum_mono_retention)
    if boundary is not None and boundary>settings.maximum_boundary_step:
        add("bar_boundary_step",boundary,settings.maximum_boundary_step)
    return findings


def analyze_render(score,data,settings=AnalysisSettings()):
    pcm=pcm_view(score,data)
    if not isinstance(settings,AnalysisSettings):
        raise GenerationError("Expected AnalysisSettings")
    bars=[]
    accumulators=[]
    spectra=[]
    for bar in range(score.plan.bars):
        start_tick=bar*TICKS_PER_BAR
        stop_tick=start_tick+TICKS_PER_BAR
        start,stop=score.plan.frame(start_tick),score.plan.frame(stop_tick)
        accum=moments(pcm,start,stop,settings)
        signal=accum.report()
        spectral=spectral_bands(pcm,start,stop)
        crossing=[e for e in score.events if e.tick<stop_tick and e.tick+e.length>start_tick]
        starts=[e for e in crossing if e.tick>=start_tick]
        boundary=(float(np.max(np.abs(pcm[start].astype(np.int64)-pcm[start-1].astype(np.int64))))/32768.
                  if start else float(np.max(np.abs(pcm[0].astype(np.int64))))/32768.)
        bars.append({"bar":bar,"start_frame":start,"stop_frame":stop,
                     "harmony_degree":score.harmony[bar],"note_onsets":len(starts),
                     "overlapping_events":len(crossing),
                     "tracks":[track for track in TRACKS if any(e.track==track for e in crossing)],
                     "boundary_step":boundary,"signal":signal,"spectrum":spectral,
                     "findings":_findings(signal,settings,boundary=boundary,expected_activity=bool(crossing))})
        accumulators.append(accum)
        spectra.append(spectral)
    sections=[]
    overall=Moments()
    for section in score.sections:
        selected=range(section.first_bar,section.first_bar+section.bars)
        acc=Moments()
        for bar in selected:
            acc.merge(accumulators[bar])
        overall.merge(acc)
        report=acc.report()
        sections.append({"kind":section.kind,"first_bar":section.first_bar,"bars":section.bars,
                         "start_frame":bars[section.first_bar]["start_frame"],
                         "stop_frame":bars[section.first_bar+section.bars-1]["stop_frame"],
                         "signal":report,"spectrum":_merge_spectra([spectra[i] for i in selected]),
                         "findings":_findings(report,settings)})
    signal=overall.report()
    return {"protocol":PROTOCOL,"score_sha256":arrangement_digest(score),
            "wav_sha256":hashlib.sha256(data).hexdigest(),"frames":len(pcm),"sample_rate":SAMPLE_RATE,
            "settings":asdict(settings),"signal":signal,"spectrum":_merge_spectra(spectra),
            "bars":bars,"sections":sections,"findings":_findings(signal,settings,expected_activity=True),
            "source_geometry_verified":True,"source_provenance_verified":False,
            "sample_peak_only":True,"lufs_measured":False,"perceptual_quality_established":False,
            "spectra_aligned_to_bars":True}
