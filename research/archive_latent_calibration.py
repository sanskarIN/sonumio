"""Streaming calibration with explicit recording hierarchy weighting.

The distribution is uniform by category, recording group, asset, then crop,
with uniform time within each latent channel. This mirrors the intended
balanced sampler distribution without random draws. The plan uses only
eligible training provenance; no identifier is included in the report.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import struct

import numpy as np
import torch

from research.archive_latent_access import AuditedCropAccess
from research.corpus_archive_format import ArchiveError, canonical_json
from research.crop_codec import CropAudioCodec
from research.crop_latent import (
    CropLatentCalibration, checked_latent, encode_native, MAX_CALIBRATION_CLIPS,
)
from research.crop_codec_packet import LATENT_CHANNELS, check_frames, check_bits
from research.wave_diffusion import CATEGORIES

PROTOCOL = "sonumio-private-balanced-latent-calibration-v1"
MODES = ("group_balanced", "clip_weighted")


def calibration_weights(access: AuditedCropAccess, mode: str):
    """Return private row-index weights and public aggregate coverage only."""
    if not isinstance(access, AuditedCropAccess) or type(mode) is not str or mode not in MODES:
        raise ArchiveError("Invalid private latent calibration weighting mode")
    hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for item in access.ready.eligible["train"]:
        row = item.row
        hierarchy[row.label][row.group_id][row.asset_id].append(row.index)
    if set(hierarchy) != set(range(len(CATEGORIES))):
        raise ArchiveError("Private latent calibration requires all categories")
    weights = {}
    by_category = {}
    total = access.count("train")
    for label, groups in sorted(hierarchy.items()):
        for assets in groups.values():
            for crops in assets.values():
                weight = (1.0/total if mode == "clip_weighted" else
                          1.0/(len(CATEGORIES)*len(groups)*len(assets)*len(crops)))
                for index in crops:
                    if index in weights:
                        raise ArchiveError("Duplicate private latent calibration index")
                    weights[index] = weight
        indices = [index for assets in groups.values() for crops in assets.values() for index in crops]
        by_category[CATEGORIES[label]] = {
            "recording_groups": len(groups),
            "assets": sum(len(assets) for assets in groups.values()),
            "eligible_clips": len(indices),
            "weight_mass": math.fsum(weights[index] for index in indices),
        }
    if len(weights) != total or not math.isclose(math.fsum(weights.values()),1.,rel_tol=0,abs_tol=1e-12):
        raise ArchiveError("Private latent calibration weights do not sum to one")
    trace = hashlib.sha256(canonical_json({
        "protocol":PROTOCOL,"mode":mode,
        "eligible_training_sha256":access.ready.eligible_fingerprints["train"],
    }))
    for index, weight in sorted(weights.items()):
        trace.update(struct.pack("<Id",index,weight))
    return weights,{
        "protocol":PROTOCOL,"mode":mode,"weight_trace_sha256":trace.hexdigest(),
        "eligible_training_sha256":access.ready.eligible_fingerprints["train"],
        "eligible_clips":total,"total_weight":math.fsum(weights.values()),
        "effective_weighted_clip_count":1/math.fsum(w*w for w in weights.values()),
        "by_category":by_category,"recording_identifiers_exported":False,
    }


class WeightedChannelMoments:
    """Merge per-clip channel distributions with normalized row weights."""
    def __init__(self, frames):
        self.frames = check_frames(frames)
        self.clips = 0
        self.mass = 0.
        self.mean = np.zeros(LATENT_CHANNELS,dtype=np.float64)
        self.m2 = np.zeros_like(self.mean)

    def add(self, latent, weight):
        checked_latent(latent,self.frames,native=True)
        if (len(latent)!=1 or latent.requires_grad or type(weight) not in (int,float)
                or not math.isfinite(weight) or not 0<weight<=1):
            raise ArchiveError("Expected one detached weighted native latent clip")
        if self.clips>=MAX_CALIBRATION_CLIPS:
            raise ArchiveError("Weighted calibration exceeds clip budget")
        data = latent[0].numpy().astype(np.float64)
        mean = data.mean(axis=1)
        variance = ((data-mean[:,None])**2).mean(axis=1)
        combined = self.mass+weight
        delta = mean-self.mean
        self.m2 += weight*variance+delta**2*self.mass*weight/combined
        self.mean += delta*weight/combined
        self.mass = combined
        self.clips += 1

    def finish(self):
        if self.clips<2 or not math.isclose(self.mass,1.,rel_tol=0,abs_tol=1e-10):
            raise ArchiveError("Incomplete weighted latent calibration")
        std = np.sqrt(np.maximum(self.m2/self.mass,0))
        return CropLatentCalibration(self.frames,tuple(self.mean.tolist()),tuple(std.tolist()),self.clips)


@torch.no_grad()
def fit_archive_calibration(access, codec, *, bits=None, mode="group_balanced"):
    if (not isinstance(codec,CropAudioCodec) or codec.training
            or any(p.requires_grad for p in codec.parameters())):
        raise ArchiveError("Private calibration requires frozen evaluation-mode codec")
    if bits is not None:
        check_bits(bits)
    weights, report = calibration_weights(access,mode)
    moments = WeightedChannelMoments(codec.frames)
    for rows, wave in access.scan("train",1):
        row, = rows
        if row.index not in weights:
            raise ArchiveError("Private calibration row not in weight plan")
        moments.add(encode_native(codec,wave,bits),weights.pop(row.index))
    if weights or moments.clips!=access.count("train"):
        raise ArchiveError("Private calibration weight plan not fully consumed")
    calibration = moments.finish()
    return calibration,{**report,"calibration_sha256":calibration.fingerprint(),
                        "heldout_used_for_fitting":False}
