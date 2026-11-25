"""Aggregate, recording-group-aware validation for private crop latent research."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import struct

import torch

from research.archive_latent_access import AuditedCropAccess, counter_generator, hash_tensor
from research.crop_codec import reconstruction_metrics
from research.crop_latent import encode_native, restore_for_decoder
from research.corpus_archive_format import ArchiveError
from research.velocity_prediction import velocity_target, decode_velocity
from research.wave_diffusion import CATEGORIES, CLIP_FRAMES

PROTOCOL = "sonumio-private-archive-latent-validation-v1"
LEVELS = ("early", "middle", "late")
METRICS = (
    "matched_velocity_mse", "null_velocity_mse", "wrong_velocity_mse",
    "zero_velocity_mse", "wrong_minus_matched", "null_minus_matched",
    "codec_waveform_l1", "decoded_waveform_l1", "decoded_minus_codec_l1",
    "codec_waveform_mse", "decoded_waveform_mse",
    "codec_snr_db", "decoded_snr_db", "native_outside_bound_fraction",
    "native_peak_before_projection", "clamp_mse", "requantization_mse",
)


class _Mean:
    def __init__(self):
        self.count = 0
        self.sums = dict.fromkeys(METRICS, 0.0)

    def add(self, values):
        if (set(values) != set(METRICS)
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in values.values())):
            raise ArchiveError("Invalid private latent validation metrics")
        self.count += 1
        for key, value in values.items():
            self.sums[key] += value

    def report(self):
        return {"cases": self.count, **{
            key: value/self.count if self.count else None for key, value in self.sums.items()
        }}


def _macro(groups):
    values = [group.report() for group in groups.values()]
    return {"recording_groups": len(values), **{
        metric: math.fsum(row[metric] for row in values)/len(values) if values else None
        for metric in METRICS
    }}


@torch.no_grad()
def validate_latent(access: AuditedCropAccess, codec, model, schedule, calibration,
                    *, bits, seed, repeats, batch_size):
    if (not isinstance(access, AuditedCropAccess)
            or type(repeats) is not int or not 1 <= repeats <= 4):
        raise ArchiveError("Invalid private latent validation setup")
    if codec.training or any(parameter.requires_grad for parameter in codec.parameters()):
        raise ArchiveError("Private latent validation requires frozen codec")
    panel = (0, (schedule.steps-1)//2, schedule.steps-1)
    overall = _Mean()
    by_level = {level: _Mean() for level in LEVELS}
    by_category = {category: _Mean() for category in CATEGORIES}
    groups = defaultdict(_Mean)
    level_groups = {level: defaultdict(_Mean) for level in LEVELS}
    category_groups = {category: defaultdict(_Mean) for category in CATEGORIES}
    trace = hashlib.sha256(b"sonumio-private-latent-validation-noise-v1\0")
    mode = model.training
    model.eval()
    try:
        for rows, source_batch in access.scan("validation", batch_size):
            # One clip at a time fixes numerical evaluation geometry as well
            # as noise identity, independent of archive shard and scan batch.
            for row, wave in zip(rows, source_batch, strict=True):
                source = wave[None]
                native = encode_native(codec, source, bits)
                clean = calibration.normalize(native)
                codec_metrics = reconstruction_metrics(source, codec.decode(native), frames=CLIP_FRAMES)
                for level, t in zip(LEVELS, panel, strict=True):
                    for repeat in range(repeats):
                        counter = row.index*12 + panel.index(t)*4 + repeat
                        noise = torch.randn(clean.shape, generator=counter_generator(seed,b"validation",counter),
                                            dtype=torch.float32)
                        trace.update(struct.pack("<III", row.index, t, repeat))
                        hash_tensor(trace, noise, "<f4")
                        timestep = torch.tensor([t], dtype=torch.long)
                        label = torch.tensor([row.label], dtype=torch.long)
                        target = velocity_target(clean, noise, timestep, schedule)
                        noisy = schedule.corrupt(clean, timestep, noise)
                        prediction = model(noisy, timestep, label)
                        null = model(noisy, timestep, torch.tensor([model.classes]))
                        # Score every wrong class, not just a chosen rotation.
                        wrong = [float((model(noisy, timestep, torch.tensor([other]))-target).double().square().mean())
                                 for other in range(model.classes) if other != row.label]
                        matched_mse = float((prediction-target).double().square().mean())
                        null_mse = float((null-target).double().square().mean())
                        wrong_mse = min(wrong)
                        estimated, _ = decode_velocity(noisy, prediction, timestep, schedule)
                        projected = restore_for_decoder(estimated, calibration, bits=bits)
                        decoded = reconstruction_metrics(source, codec.decode(projected.latent), frames=CLIP_FRAMES)
                        values = {
                            "matched_velocity_mse": matched_mse,
                            "null_velocity_mse": null_mse,
                            "wrong_velocity_mse": wrong_mse,
                            "zero_velocity_mse": float(target.double().square().mean()),
                            "wrong_minus_matched": wrong_mse-matched_mse,
                            "null_minus_matched": null_mse-matched_mse,
                            "codec_waveform_l1": float(codec_metrics["waveform_l1"][0]),
                            "decoded_waveform_l1": float(decoded["waveform_l1"][0]),
                            "decoded_minus_codec_l1": float(decoded["waveform_l1"][0]-codec_metrics["waveform_l1"][0]),
                            "codec_waveform_mse": float(codec_metrics["waveform_mse"][0]),
                            "decoded_waveform_mse": float(decoded["waveform_mse"][0]),
                            "codec_snr_db": float(codec_metrics["snr_db"][0]),
                            "decoded_snr_db": float(decoded["snr_db"][0]),
                            **{key: projected.diagnostics[key] for key in (
                                "native_outside_bound_fraction", "native_peak_before_projection",
                                "clamp_mse", "requantization_mse")},
                        }
                        category = CATEGORIES[row.label]
                        for summary in (overall, by_level[level], by_category[category],
                                        groups[row.group_id], level_groups[level][row.group_id],
                                        category_groups[category][row.group_id]):
                            summary.add(values)
    finally:
        model.train(mode)
    if overall.count != access.count("validation")*len(panel)*repeats:
        raise ArchiveError("Private latent validation coverage incomplete")
    return {
        "protocol": PROTOCOL, "timesteps": dict(zip(LEVELS,panel,strict=True)),
        "seed": seed, "repeats": repeats, "noise_trace_sha256": trace.hexdigest(),
        "eligible_validation_sha256": access.ready.eligible_fingerprints["validation"],
        "eligible_validation_clips": access.count("validation"),
        "clip_weighted": overall.report(), "group_macro": _macro(groups),
        "by_timestep": {name: {"clip_weighted": by_level[name].report(),
                              "group_macro": _macro(level_groups[name])} for name in LEVELS},
        "by_category": {name: {"clip_weighted": by_category[name].report(),
                              "group_macro": _macro(category_groups[name])} for name in CATEGORIES},
        "wrong_condition_control": "lowest_mse_across_all_wrong_labels",
        "teacher_forced_not_generated_audio": True, "test_split_scored": False,
        "recording_identifiers_exported": False,
    }
