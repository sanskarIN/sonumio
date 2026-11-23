"""Score sampled waveforms against isolated first-party synthetic references."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from platform import python_version
from statistics import mean

import numpy as np
import torch

from research.architecture_compare import _audio_hashes
from research.generation_audit import _wave_array
from research.latent_audio import (
    FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS, render_seeded_waveform, synthetic_bilingual_waves,
)
from research.text_audio import _synth_prompts
from research.velocity_ddim import sample_velocity_ddim
from research.velocity_ddim_study import _model_state_sha256, _train_models
from research.wave_diffusion import CATEGORIES, DiffusionSchedule, SAMPLE_RATE, _downsample_24k_to_8k

PROTOCOL = "sonumio-generated-synthetic-source-grounding-v1"
REFERENCE_SEEDS = {
    "fit": (701, 702, 703, 704),
    "calibration": (711, 712),
    "holdout": (731, 732, 733),
}
LANGUAGES = ("en", "hi")
ARMS = ("raw", "latent")
MIN_REFERENCE_ACCURACY = 2 / 3
WINDOW = 512
HOP = 256
BANDS_HZ = (0, 63, 125, 250, 500, 1000, 2000, 3000, 3500, 3750, 4001)
BAND_BINS = np.searchsorted(np.fft.rfftfreq(WINDOW, 1 / SAMPLE_RATE), BANDS_HZ)
TAPER = np.hanning(WINDOW + 1)[:-1]


def spectral_signature(waveform: np.ndarray | torch.Tensor) -> np.ndarray | None:
    """Describe relative band power and its variation across short frames."""
    signal = _wave_array(waveform).astype(np.float64)
    signal -= signal.mean()
    if np.sqrt(np.mean(signal**2)) < 1e-5:
        return None

    frames = np.lib.stride_tricks.sliding_window_view(signal, WINDOW)[::HOP]
    power = np.abs(np.fft.rfft(frames * TAPER, axis=-1)) ** 2
    band_power = np.add.reduceat(power, BAND_BINS[:-1], axis=-1)
    total = band_power.sum(axis=-1)
    # Silent frames carry no spectral shape. The envelope features below
    # still reflect their duration and relative energy.
    active = total > total.max() * 1e-8
    fractions = band_power[active] / total[active, None]
    band_db = 10 * np.log10(np.maximum(fractions, 1e-8))
    envelope_db = 10 * np.log10(np.maximum(total / total.max(), 1e-8))
    return np.concatenate((
        np.mean(band_db, axis=0),
        np.std(band_db, axis=0),
        [np.mean(envelope_db), np.std(envelope_db)],
    ))


@dataclass(frozen=True)
class Decision:
    distances: tuple[float, ...]
    order: tuple[int, ...]
    accepted: bool

    def as_dict(self, target: int | None = None) -> dict:
        closest = self.order[0] if self.order else None
        row = {
            "closest": CATEGORIES[closest] if closest is not None else None,
            "distance": self.distances[closest] if closest is not None else None,
            "within_reference_radius": self.accepted,
        }
        if target is not None:
            row["target_rank"] = self.order.index(target) + 1 if self.order else None
            if self.order:
                alternative = next(index for index in self.order if index != target)
                row["target_distance_margin"] = self.distances[alternative] - self.distances[target]
            else:
                row["target_distance_margin"] = None
        return row


@dataclass(frozen=True)
class SpectralSourceProbe:
    centers: np.ndarray
    scale: np.ndarray
    radii: np.ndarray

    def score(self, waveform: np.ndarray | torch.Tensor) -> Decision:
        features = spectral_signature(waveform)
        if features is None:
            return Decision((), (), False)
        distances = _distances(features, self.centers, self.scale)
        order = tuple(int(index) for index in np.argsort(distances, kind="stable"))
        closest = order[0]
        return Decision(
            tuple(map(float, distances)),
            order,
            bool(distances[closest] <= self.radii[closest]),
        )


def _distances(features: np.ndarray, centers: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(((features[..., None, :] - centers) / scale) ** 2, axis=-1))


def _wave_hash(waveform: np.ndarray) -> str:
    return hashlib.sha256(waveform.tobytes()).hexdigest()


def _reference_data() -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    waves = {
        split: np.stack([
            [render_seeded_waveform((label,), seed) for seed in seeds]
            for label in range(len(CATEGORIES))
        ])
        for split, seeds in REFERENCE_SEEDS.items()
    }
    hashes = {
        split: tuple(_wave_hash(wave) for wave in panel.reshape(-1, FRAMES))
        for split, panel in waves.items()
    }
    all_hashes = [digest for panel in hashes.values() for digest in panel]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("Duplicate recordings in or across probe splits")
    return waves, hashes


def _panel_features(panel: np.ndarray) -> np.ndarray:
    signatures = [spectral_signature(wave) for wave in panel.reshape(-1, FRAMES)]
    if any(features is None for features in signatures):
        raise ValueError("Reference panel contains a silent recording")
    return np.stack(signatures).reshape(*panel.shape[:2], -1)


def _fit(fit: np.ndarray, calibration: np.ndarray) -> SpectralSourceProbe:
    train = _panel_features(fit)
    centers = train.mean(axis=1)
    scale = np.maximum(np.sqrt(np.mean((train - centers[:, None, :]) ** 2, axis=(0, 1))), 2)
    distances = _distances(_panel_features(calibration), centers, scale)
    own_class = distances[np.arange(len(CATEGORIES)), :, np.arange(len(CATEGORIES))]
    radii = np.maximum(1.5 * own_class.max(axis=1), 0.75)
    return SpectralSourceProbe(centers, scale, radii)


def _summary(rows: list[dict], *, noise_confounded: frozenset[str] = frozenset()) -> dict:
    ranks = [row["target_rank"] for row in rows]
    margins = [row["target_distance_margin"] for row in rows if row["target_rank"] is not None]
    accepted = [row for row in rows if row["within_reference_radius"]]
    correct_accepted = [row for row in accepted if row["target_rank"] == 1]
    return {
        "cases": len(rows),
        "cases_ranked": sum(rank is not None for rank in ranks),
        "rank_one": ranks.count(1),
        "accepted": len(accepted),
        "accepted_rank_one": len(correct_accepted),
        "accepted_rank_one_excluding_noise_confounds": sum(
            row["closest"] not in noise_confounded for row in correct_accepted
        ),
        "mean_reciprocal_rank": mean(1 / rank if rank is not None else 0 for rank in ranks),
        "mean_target_distance_margin": mean(margins) if margins else None,
    }


def _reference_controls(probe: SpectralSourceProbe, holdout: np.ndarray) -> dict:
    rows = [
        {"category": CATEGORIES[label], "recording_seed": seed, **probe.score(wave).as_dict(label)}
        for label, waves in enumerate(holdout)
        for seed, wave in zip(REFERENCE_SEEDS["holdout"], waves, strict=True)
    ]
    silence = probe.score(np.zeros(FRAMES, dtype=np.float32))
    if silence.accepted or silence.order:
        raise ValueError("Silence must abstain from source classification")
    noise = {"white_8k": [], "filtered_to_8k": []}
    for seed in REFERENCE_SEEDS["holdout"]:
        rng = np.random.default_rng(seed)
        white = rng.standard_normal(FRAMES).astype(np.float32)
        filtered = _downsample_24k_to_8k(rng.standard_normal(3 * (FRAMES + 32)))[16:-16]
        for kind, wave in (("white_8k", white), ("filtered_to_8k", filtered)):
            wave /= np.abs(wave).max()
            noise[kind].append({"seed": seed, **probe.score(wave).as_dict()})
    confounded = frozenset(
        row["closest"] for cases in noise.values() for row in cases if row["within_reference_radius"]
    )
    summary = _summary(rows, noise_confounded=confounded)
    passed = (
        summary["accepted_rank_one"] / summary["cases"] >= MIN_REFERENCE_ACCURACY
        and all(any(row["category"] == category and row["target_rank"] == 1
                    and row["within_reference_radius"] for row in rows) for category in CATEGORIES)
    )
    return {
        "heldout": summary,
        "heldout_cases": rows,
        "minimum_heldout_accepted_accuracy": MIN_REFERENCE_ACCURACY,
        "at_least_one_correct_acceptance_per_category_required": True,
        "supports_interpreting_generated_scores": passed,
        "silence_abstains": True,
        "noise_controls": noise,
        "categories_accepted_for_noise": sorted(confounded),
        "acceptance_radii": dict(zip(CATEGORIES, map(float, probe.radii), strict=True)),
    }


def _sample_assay(
    probe: SpectralSourceProbe, models: dict, codec, schedule: DiffusionSchedule,
    *, sampling_seeds: tuple[int, ...], evaluations: int, guidance: float,
) -> tuple[list[dict], list[dict]]:
    cases, null_checks = [], []
    for arm in ARMS:
        for seed in sampling_seeds:
            settings = {
                "seed": seed, "evaluations": evaluations, "latent": arm == "latent",
                "codec": codec if arm == "latent" else None,
            }
            first = sample_velocity_ddim(
                models[arm], schedule, _synth_prompts((0,), "en", seed), guidance=0, **settings,
            )
            second = sample_velocity_ddim(
                models[arm], schedule, _synth_prompts((1,), "hi", seed), guidance=0, **settings,
            )
            if not torch.equal(first, second):
                raise ValueError("Zero-guidance waveform changed with caption")
            null = probe.score(first.clamp(-1, 1))
            null_checks.append({
                "arm": arm, "sampling_seed": seed,
                "zero_guidance_waveform_sha256": _wave_hash(first.numpy()),
                "caption_invariant": True,
            })
            for language in LANGUAGES:
                for label, category in enumerate(CATEGORIES):
                    caption = _synth_prompts((label,), language, seed)
                    wave = sample_velocity_ddim(models[arm], schedule, caption, guidance=guidance, **settings)
                    audio = wave.clamp(-1, 1)
                    decision = probe.score(audio)
                    rotated = (label + 1) % len(CATEGORIES)
                    cases.append({
                        "arm": arm, "sampling_seed": seed, "language": language,
                        "requested": category, "prompt": caption, "rotated_label": CATEGORIES[rotated],
                        "matched": decision.as_dict(label),
                        "rotated": decision.as_dict(rotated),
                        "null_guidance": null.as_dict(label),
                        "waveform_sha256": _wave_hash(audio.numpy()),
                        "preclamp_out_of_range_fraction": float((wave.abs() > 1).double().mean()),
                    })
    return cases, null_checks


def run_study(
    *, model_seed: int = 17, train_steps: int = 16, codec_steps: int = 8,
    evaluations: int = 8, guidance: float = 1.0, sampling_seeds: tuple[int, ...] = (301, 302),
) -> dict:
    """Train and sample both original v-DDIM arms, then compare fixed controls."""
    if (type(model_seed) is not int or not 0 <= model_seed < 2**32 - 8192
            or type(train_steps) is not int or not 1 <= train_steps <= 64
            or type(codec_steps) is not int or not 1 <= codec_steps <= 64
            or type(evaluations) is not int or not 2 <= evaluations <= 8
            or type(guidance) not in (int, float) or not math.isfinite(guidance) or not 0 < guidance <= 4
            or type(sampling_seeds) is not tuple or not 1 <= len(sampling_seeds) <= 4
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in sampling_seeds)
            or len(set(sampling_seeds)) != len(sampling_seeds)):
        raise ValueError("Invalid bounded source-grounding study configuration")

    all_recording_seeds = tuple(seed for seeds in REFERENCE_SEEDS.values() for seed in seeds)
    all_recording_seeds += TRAIN_SEEDS + VALIDATION_SEEDS
    if len(set(all_recording_seeds)) != len(all_recording_seeds):
        raise ValueError("Reference and generator seed splits overlap")
    waves, hashes = _reference_data()
    generator_train = synthetic_bilingual_waves(TRAIN_SEEDS)
    generator_heldout = synthetic_bilingual_waves(VALIDATION_SEEDS)
    generator_hashes = _audio_hashes(generator_train.waves) | _audio_hashes(generator_heldout.waves)
    generator_fingerprints = (generator_train.fingerprint, generator_heldout.fingerprint)
    if generator_hashes.intersection(digest for split in hashes.values() for digest in split):
        raise ValueError("Exact recordings overlap between reference and generator data")
    probe = _fit(waves["fit"], waves["calibration"])
    reference = _reference_controls(probe, waves["holdout"])
    reference["splits"] = {
        split: {
            "recording_seeds": list(seeds),
            "recording_count": len(hashes[split]),
            "sha256": hashlib.sha256("".join(hashes[split]).encode("ascii")).hexdigest(),
        }
        for split, seeds in REFERENCE_SEEDS.items()
    }
    reference["overlap_with_generator_data"] = False
    del waves, generator_train, generator_heldout

    training, fingerprint, cases, null_checks = None, None, [], []
    if reference["supports_interpreting_generated_scores"]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            schedule = DiffusionSchedule(8, profile="cosine_zero_terminal_v1")
            models, codec, training = _train_models(
                seed=model_seed, train_steps=train_steps, codec_steps=codec_steps,
                schedule=schedule, channels=16,
            )
            if (training["training_fingerprint"], training["validation_fingerprint"]) != generator_fingerprints:
                raise ValueError("Generator data changed after probe overlap verification")
            fingerprint = _model_state_sha256(models, codec)
            cases, null_checks = _sample_assay(
                probe, models, codec, schedule, sampling_seeds=sampling_seeds,
                evaluations=evaluations, guidance=guidance,
            )
            if _model_state_sha256(models, codec) != fingerprint:
                raise ValueError("Sampling mutated the trained generator state")
    summaries = {}
    if cases:
        noise_confounded = frozenset(reference["categories_accepted_for_noise"])
        for arm in ARMS:
            summaries[arm] = {}
            for language in LANGUAGES:
                panel = [case for case in cases if case["arm"] == arm and case["language"] == language]
                summaries[arm][language] = {
                    key: _summary([case[key] for case in panel], noise_confounded=noise_confounded)
                    for key in ("matched", "rotated", "null_guidance")
                }
    return {
        "protocol": PROTOCOL,
        "runtime": {"python": python_version(), "numpy": np.__version__, "torch": torch.__version__},
        "configuration": {
            "sample_rate": SAMPLE_RATE, "frames": FRAMES, "model_seed": model_seed,
            "train_steps": train_steps, "codec_steps": codec_steps, "evaluations": evaluations,
            "guidance": guidance, "sampling_seeds": list(sampling_seeds), "categories": list(CATEGORIES),
        },
        "status": "scored" if cases else "reference_gate_failed",
        "reference_probe": reference,
        "uniform_rank_one_chance": 1 / len(CATEGORIES),
        "training": training,
        "trained_model_sha256": fingerprint,
        "by_arm_and_language": summaries,
        "cases": cases,
        "null_guidance_checks": null_checks,
        "weight_or_audio_files_saved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--codec-steps", type=int, default=8)
    parser.add_argument("--evaluations", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--sampling-seeds", type=int, nargs="+", default=[301, 302])
    args = parser.parse_args()
    report = run_study(
        model_seed=args.model_seed, train_steps=args.train_steps, codec_steps=args.codec_steps,
        evaluations=args.evaluations, guidance=args.guidance, sampling_seeds=tuple(args.sampling_seeds),
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
