"""Original nature sound numeric descriptors and paired evaluation.

All metrics use the same sample clock and waveform. They diagnose
spectral/temporal behavior, not semantic or perceptual realism.
"""
from __future__ import annotations

import math
import numpy as np
from .audio import SAMPLE_RATE
from .errors import GenerationError

FFT=1024
HOP=512
BANDS=(0.,125.,500.,2000.,4500.,8000.,12000.)
POWER_FLOOR=1e-15
MAX_FRAMES=30*SAMPLE_RATE


def _mono(data):
    if (not isinstance(data,np.ndarray) or data.dtype not in (np.dtype("float32"),np.dtype("float64"))
            or data.ndim!=1 or not FFT<=len(data)<=MAX_FRAMES
            or not np.isfinite(data).all() or np.max(np.abs(data))>16):
        raise GenerationError("Invalid bounded nature diagnostic waveform")
    return data.astype(np.float64)


def _spectral(x):
    window=np.hanning(FFT)
    weights=np.ones(FFT//2+1)
    weights[1:-1]=2
    frequency=np.fft.rfftfreq(FFT,1/SAMPLE_RATE)
    total=np.zeros_like(frequency)
    previous=None
    flux=[]
    frame_rms=[]
    for start in range(0,len(x)-FFT+1,HOP):
        frame=x[start:start+FFT]
        frame_rms.append(float(np.sqrt(np.mean(frame*frame))))
        magnitude=np.abs(np.fft.rfft(frame*window))
        power=magnitude*magnitude*weights
        total+=power
        norm=float(np.linalg.norm(magnitude))
        normalized=magnitude/max(norm,POWER_FLOOR)
        if previous is not None:
            flux.append(float(np.linalg.norm(np.maximum(normalized-previous,0))))
        previous=normalized
    energy=float(total.sum())
    fractions=[]
    for lo,hi in zip(BANDS,BANDS[1:]):
        mask=(frequency>=lo)&((frequency<=hi) if hi==BANDS[-1] else (frequency<hi))
        fractions.append(float(total[mask].sum()/energy) if energy>POWER_FLOOR else None)
    if energy>POWER_FLOOR:
        probabilities=total/energy
        centroid=float(np.dot(frequency,probabilities))
        spread=float(np.sqrt(np.dot((frequency-centroid)**2,probabilities)))
        rolloff=float(frequency[min(len(frequency)-1,int(np.searchsorted(np.cumsum(probabilities),.85)))])
        positive=probabilities>0
        entropy=float(-np.sum(probabilities[positive]*np.log2(probabilities[positive])))
        # Flatness with a scale-relative floor. It is diagnostic only.
        scaled=total/max(float(total.max()),POWER_FLOOR)
        flatness=float(np.exp(np.mean(np.log(np.maximum(scaled,1e-12))))/
                       max(float(np.mean(scaled)),POWER_FLOOR))
    else:
        centroid=spread=rolloff=entropy=flatness=None
    return {"spectral_energy":energy,"band_energy_fractions":fractions,
            "centroid_hz":centroid,"spread_hz":spread,"rolloff85_hz":rolloff,
            "spectral_entropy_bits":entropy,"spectral_flatness":flatness,
            "mean_positive_spectral_flux":float(np.mean(flux)) if flux else 0.,
            "spectral_frames":len(frame_rms),
            "window_rms_p10":float(np.quantile(frame_rms,.1)),
            "window_rms_p50":float(np.quantile(frame_rms,.5)),
            "window_rms_p90":float(np.quantile(frame_rms,.9))}


def _envelope_stats(x):
    step=SAMPLE_RATE//20
    rms=np.array([np.sqrt(np.mean(x[i:i+step]**2)) for i in range(0,len(x),step)])
    maximum=float(rms.max())
    mean=float(rms.mean())
    active=rms>max(1e-5,maximum*.1)
    # Count contiguous above-threshold regions; not an event detector.
    starts=np.count_nonzero(active&~np.r_[False,active[:-1]])
    centroid=float(np.dot(np.arange(len(rms))*step+step/2,rms*rms)/np.sum(rms*rms)) if maximum>0 else None
    return {"envelope_windows":len(rms),"active_envelope_fraction":float(np.mean(active)),
            "active_envelope_regions":int(starts),
            "envelope_coefficient_of_variation":float(rms.std()/mean) if mean>POWER_FLOOR else None,
            "envelope_energy_centroid_seconds":None if centroid is None else min(len(x)/SAMPLE_RATE,centroid/SAMPLE_RATE)}


def describe(data):
    x=_mono(data)
    peak=float(np.max(np.abs(x)))
    rms=float(np.sqrt(np.mean(x*x)))
    crossing=np.count_nonzero((x[:-1]<0)!=(x[1:]<0))
    report={"frames":len(x),"sample_rate":SAMPLE_RATE,"rms":rms,"peak":peak,
            "mean_dc":float(x.mean()),"crest_factor":peak/rms if rms>POWER_FLOOR else None,
            "zero_crossings_per_second":float(crossing*SAMPLE_RATE/(len(x)-1)),
            "max_adjacent_step":float(np.max(np.abs(np.diff(x)))),
            "overrange_fraction":float(np.mean(np.abs(x)>1)),
            "first_sample":float(x[0]),"last_sample":float(x[-1])}
    report.update(_spectral(x))
    report.update(_envelope_stats(x))
    return report


def compare(a,b):
    x=_mono(a)
    y=_mono(b)
    if x.shape!=y.shape:
        raise GenerationError("Nature comparison sample geometry mismatch")
    a=describe(x)
    b=describe(y)
    xc=x-x.mean()
    yc=y-y.mean()
    norm=float(np.linalg.norm(xc)*np.linalg.norm(yc))
    correlation=float(np.dot(xc,yc)/norm) if norm>POWER_FLOOR else None
    bands_a=a["band_energy_fractions"]
    bands_b=b["band_energy_fractions"]
    distance=sum(abs(i-j) for i,j in zip(bands_a,bands_b,strict=True)) if all(i is not None for i in (*bands_a,*bands_b)) else None
    return {"waveform_mae":float(np.mean(np.abs(x-y))),
            "centered_waveform_correlation":None if correlation is None else max(-1.,min(1.,correlation)),
            "band_distribution_l1":distance,
            "rms_delta":b["rms"]-a["rms"],
            "centroid_delta_hz":None if a["centroid_hz"] is None or b["centroid_hz"] is None else b["centroid_hz"]-a["centroid_hz"],
            "envelope_activity_delta":b["active_envelope_fraction"]-a["active_envelope_fraction"],
            "a":a,"b":b,"perceptual_quality_established":False}
