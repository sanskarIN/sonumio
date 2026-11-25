"""Transparent numerical reference features for the 8-kHz waveform baseline.

No pretrained classifier, semantic embedding, or listening-quality score.
Reference centroids are fitted only on eligible training clips, with equal
weight per recording group within each category. Validation checks whether
the reference probe has any discriminative value before interpreting samples.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from research.archive_readiness import ArchiveReadinessResult,eligible_fingerprint
from research.audio_dataset import _cropped_clip_digest
from research.corpus_archive import CorpusArchiveReader
from research.corpus_archive_format import ArchiveError,canonical_json
from research.wave_diffusion import CLIP_FRAMES,SAMPLE_RATE,CATEGORIES

PROTOCOL="sonumio-archive-spectral-reference-v1"
WINDOW=256
HOP=128
BAND_EDGES=(0,125,250,500,1000,2000,3000,4001)
BINS=np.searchsorted(np.fft.rfftfreq(WINDOW,1/SAMPLE_RATE),BAND_EDGES)
TAPER=np.hanning(WINDOW+1)[:-1]
FEATURE_NAMES=tuple(
    [f"band_{lo}_{hi}_mean_db" for lo,hi in zip(BAND_EDGES[:-1],BAND_EDGES[1:],strict=True)]
    +[f"band_{lo}_{hi}_std_db" for lo,hi in zip(BAND_EDGES[:-1],BAND_EDGES[1:],strict=True)]
    +["frame_energy_std_db","centroid_over_nyquist","zero_crossing_fraction","spectral_flatness"]
)
MIN_RMS=1e-5
SCALE_FLOOR=.05
MAX_SIGNAL_MAGNITUDE=1e6


def _wave(value):
    if (not isinstance(value,np.ndarray) or value.dtype!=np.float32
            or value.shape!=(CLIP_FRAMES,) or not np.isfinite(value).all()):
        raise ArchiveError("Expected finite float32 waveform clip")
    signal=value.astype(np.float64)
    if np.max(np.abs(signal))>MAX_SIGNAL_MAGNITUDE:
        raise ArchiveError("Generated waveform exceeds numerical diagnostic bound")
    return signal


def signal_statistics(value):
    signal=_wave(value)
    centered=signal-signal.mean()
    rms=float(np.sqrt(np.mean(signal**2)))
    crms=float(np.sqrt(np.mean(centered**2)))
    peak=float(np.max(np.abs(signal)))
    return {
        "rms":rms,"centered_rms":crms,"peak":peak,"absolute_dc":abs(float(signal.mean())),
        "overrange_fraction":float(np.mean(np.abs(signal)>1)),
        "near_clip_fraction":float(np.mean(np.abs(signal)>=.999)),
        "crest_factor":peak/rms if rms else None,
        "silent_or_constant":crms<MIN_RMS,
    }


def spectral_features(value):
    """Relative-power features are invariant to positive gain and DC shift.

    Silence/constant inputs are explicitly unscorable, not treated as a
    category. No waveform clipping or normalization is performed for the
    amplitude statistics; only this feature vector is centered and scaled.
    """
    signal=_wave(value)
    signal-=signal.mean()
    rms=float(np.sqrt(np.mean(signal**2)))
    if rms<MIN_RMS:
        return None
    signal/=rms
    frames=np.lib.stride_tricks.sliding_window_view(signal,WINDOW)[::HOP]
    power=np.abs(np.fft.rfft(frames*TAPER,axis=-1))**2
    total=power.sum(axis=1)
    active=total>total.max()*1e-8
    fractions=np.add.reduceat(power[active],BINS[:-1],axis=1)/total[active,None]
    band_db=10*np.log10(np.maximum(fractions,1e-8))
    relative=10*np.log10(np.maximum(total/total.max(),1e-8))
    freq=np.fft.rfftfreq(WINDOW,1/SAMPLE_RATE)/(SAMPLE_RATE/2)
    centroid=float(np.mean((power[active]@freq)/total[active]))
    # Compute flatness on normalized relative bin powers; floor avoids log(0).
    bin_fraction=power[active]/total[active,None]
    flatness=float(np.mean(np.exp(np.mean(np.log(np.maximum(bin_fraction,1e-12)),axis=1))
                          *power.shape[1]))
    crossing=float(np.mean(np.signbit(signal[1:])!=np.signbit(signal[:-1])))
    result=np.concatenate((band_db.mean(axis=0),band_db.std(axis=0),
                           [relative.std(),centroid,crossing,flatness]))
    if result.shape!=(len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ArchiveError("Nonfinite spectral reference features")
    return result


@dataclass(frozen=True,slots=True,eq=False)
class SpectralReference:
    centers:np.ndarray
    scale:np.ndarray
    fingerprint:str
    eligible_training_sha256:str
    group_counts:tuple[int,...]

    def distances(self,features):
        if (not isinstance(features,np.ndarray) or features.shape!=(len(FEATURE_NAMES),)
                or not np.isfinite(features).all()):
            raise ArchiveError("Invalid spectral probe input")
        return np.mean(((self.centers-features)/self.scale)**2,axis=1)


def _audited_rows(reader,ready,split):
    rows=ready.eligible[split]
    fingerprint=eligible_fingerprint(rows)
    if (fingerprint!=ready.eligible_fingerprints[split]
            or fingerprint!=ready.report["eligible_split_fingerprints"][split]):
        raise ArchiveError("Spectral probe eligibility fingerprint changed")
    expected={item.row.index:item for item in rows}
    seen=set()
    for shard in reader.iter_shards(split):
        for row,clip in zip(shard.rows,shard.clips,strict=True):
            item=expected.get(row.index)
            if item is None:
                continue
            if row.index in seen or item.row!=row or _cropped_clip_digest(clip[0])!=item.pcm_sha256:
                raise ArchiveError("Spectral probe input differs from audited row")
            seen.add(row.index)
            yield row,clip[0]
    if seen!=set(expected):
        raise ArchiveError("Spectral probe missed eligible rows")


def _require_ready(reader,ready):
    if not isinstance(reader,CorpusArchiveReader) or not isinstance(ready,ArchiveReadinessResult):
        raise ArchiveError("Expected verified archive and readiness")
    if (reader.report()["archive_sha256"]!=ready.archive_sha256
            or ready.report["readiness_gate_passed"] is not True):
        raise ArchiveError("Spectral probe requires matching ready archive")


def fit_reference(reader,ready):
    _require_ready(reader,ready)
    sums={}
    counts=defaultdict(int)
    for row,clip in _audited_rows(reader,ready,"train"):
        feature=spectral_features(clip)
        if feature is None:
            continue
        key=(row.label,row.group_id)
        if key not in sums:
            sums[key]=np.zeros(len(FEATURE_NAMES),dtype=np.float64)
        sums[key]+=feature
        counts[key]+=1
    groups=[[] for _ in CATEGORIES]
    for key,value in sorted(sums.items()):
        groups[key[0]].append(value/counts[key])
    if any(not group for group in groups):
        raise ArchiveError("Spectral reference lacks a training category")
    centers=np.stack([np.mean(group,axis=0) for group in groups])
    # Category-balanced scale: within- and between-group variation about
    # the equally weighted category center. No held-out data sets scaling.
    global_center=centers.mean(axis=0)
    scale=np.sqrt(np.mean([np.mean((np.stack(group)-global_center)**2,axis=0)
                          for group in groups],axis=0))
    scale=np.maximum(scale,SCALE_FLOOR)
    group_counts=tuple(map(len,groups))
    hasher=hashlib.sha256(canonical_json({
        "protocol":PROTOCOL,"features":list(FEATURE_NAMES),
        "eligible_training_sha256":ready.eligible_fingerprints["train"],
        "group_counts":list(group_counts),"scale_floor":SCALE_FLOOR,
    }))
    hasher.update(np.asarray(centers,dtype="<f8").tobytes())
    hasher.update(np.asarray(scale,dtype="<f8").tobytes())
    # Immutable bytes-backed arrays protect the fitted fingerprint.
    centers=np.frombuffer(centers.tobytes(),dtype=np.float64).reshape(centers.shape)
    scale=np.frombuffer(scale.tobytes(),dtype=np.float64)
    return SpectralReference(centers,scale,hasher.hexdigest(),
                             ready.eligible_fingerprints["train"],group_counts)


def classify(reference,features,target):
    if type(target) is not int or not 0<=target<len(CATEGORIES):
        raise ArchiveError("Invalid reference target category")
    if features is None:
        return {"predicted":None,"target_rank":None,"target_distance":None,
                "nearest_wrong_minus_target":None,"strict_win":False,"tied_best":False}
    distances=reference.distances(features)
    target_distance=float(distances[target])
    wrong=np.delete(distances,target)
    less=int(np.count_nonzero(wrong<target_distance-1e-10))
    ties=int(np.count_nonzero(np.abs(wrong-target_distance)<=1e-10))
    nearest=np.flatnonzero(np.abs(distances-distances.min())<=1e-10)
    return {
        "predicted":int(nearest[0]) if len(nearest)==1 else None,
        "target_rank":1+less+.5*ties,"target_distance":target_distance,
        "nearest_wrong_minus_target":float(wrong.min()-target_distance),
        "strict_win":less==0 and ties==0,"tied_best":less==0 and ties>0,
    }


def validation_reference_report(reader,ready,reference):
    _require_ready(reader,ready)
    if reference.eligible_training_sha256!=ready.eligible_fingerprints["train"]:
        raise ArchiveError("Reference fitted on different eligible training data")
    groups=defaultdict(list)
    confusion=np.zeros((len(CATEGORIES),len(CATEGORIES)+1),dtype=np.int64)
    clips=0
    for row,clip in _audited_rows(reader,ready,"validation"):
        decision=classify(reference,spectral_features(clip),row.label)
        pred=decision["predicted"]
        confusion[row.label,pred if pred is not None else len(CATEGORIES)]+=1
        groups[(row.label,row.group_id)].append(float(decision["strict_win"]))
        clips+=1
    group_scores=[float(np.mean(values)) for values in groups.values()]
    categories=[]
    for label in range(len(CATEGORIES)):
        scores=[float(np.mean(values)) for (category,_),values in groups.items() if category==label]
        if not scores:
            raise ArchiveError("Reference validation lacks a recording category")
        categories.append(float(np.mean(scores)))
    return {
        "protocol":PROTOCOL,"reference_sha256":reference.fingerprint,
        "feature_names":list(FEATURE_NAMES),"training_groups":dict(zip(CATEGORIES,reference.group_counts,strict=True)),
        "validation_clips":clips,"validation_group_category_pairs":len(group_scores),
        "clip_accuracy":float(np.trace(confusion[:,:len(CATEGORIES)])/clips),
        "group_macro_accuracy":float(np.mean(group_scores)),
        "category_balanced_group_accuracy":float(np.mean(categories)),
        "per_category_group_accuracy":dict(zip(CATEGORIES,categories,strict=True)),
        "uniform_guess_reference":1/len(CATEGORIES),
        "confusion_columns":[*CATEGORIES,"tie_or_unscorable"],
        "confusion_rows":list(CATEGORIES),"confusion_counts":confusion.tolist(),
        "heldout_validation_used_for_fitting":False,"test_split_scored":False,
        "semantic_classifier_validated":False,
    }
