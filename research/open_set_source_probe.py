"""Open-set synthetic audio probe with explicit, isolated noise controls.

A rejection decision is not evidence of sound identity; this probe measures
similarity to first-party DSP recipes only. Noise families are artificial
counterexamples, not an exhaustive model of real-world out-of-distribution audio.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from research.generated_source_grounding import (
    Decision, REFERENCE_SEEDS, spectral_signature, _reference_data, _fit,
)
from research.generation_audit import _wave_array
from research.latent_audio import FRAMES, TRAIN_SEEDS, VALIDATION_SEEDS
from research.wave_diffusion import CATEGORIES, _downsample_24k_to_8k

PROTOCOL = "sonumio-open-set-synthetic-probe-v1"
NOISE_SEEDS = {
    "fit": (1801, 1802, 1803, 1804),
    "calibration": (1811, 1812, 1813),
    "holdout": (1831, 1832, 1833, 1834),
}
NOISE_FAMILIES = (
    "white", "downsampled_white", "pink", "brown", "smoothed_short",
    "smoothed_long", "amplitude_modulated",
)
LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
FLOOR = 1e-12


def _normalized_audio(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.shape != (FRAMES,) or not np.isfinite(data).all():
        raise ValueError("Invalid artificial control audio")
    data = data - data.mean()
    peak = np.abs(data).max()
    if peak <= FLOOR:
        raise ValueError("Degenerate artificial control audio")
    return (0.9 * data / peak).astype(np.float32)


def noise_wave(family: str, seed: int) -> np.ndarray:
    """Deterministic controls with different spectral and temporal structure."""
    if family not in NOISE_FAMILIES or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid noise family or seed")
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(FRAMES)
    if family == "white":
        signal = white
    elif family == "downsampled_white":
        signal = _downsample_24k_to_8k(rng.standard_normal(3 * (FRAMES + 32)))[16:-16]
    elif family in ("pink", "brown"):
        spectrum = np.fft.rfft(white)
        frequency = np.arange(len(spectrum), dtype=np.float64)
        scale = np.maximum(frequency, 1) ** (-0.5 if family == "pink" else -1.0)
        spectrum *= scale
        spectrum[0] = 0
        signal = np.fft.irfft(spectrum, n=FRAMES)
    elif family in ("smoothed_short", "smoothed_long"):
        width = 4 if family == "smoothed_short" else 20
        signal = np.convolve(white, np.ones(width) / width, mode="same")
    else:
        # Nonstationary noise, a deliberately harder negative than white noise.
        phase = rng.uniform(0, 2 * np.pi)
        t = np.arange(FRAMES, dtype=np.float64) / FRAMES
        envelope = 0.15 + 0.85 * (0.5 + 0.5 * np.sin(2 * np.pi * t + phase)) ** 2
        signal = white * envelope
    return _normalized_audio(signal)


def phase_surrogate(waveform: np.ndarray, seed: int) -> np.ndarray:
    """Destroy phase relationships while preserving relative full-clip power.

    These are ambiguous challenge controls, not automatically invalid audio.
    In particular, phase-scrambled stationary tonal sounds may retain identity.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid surrogate seed")
    data = _wave_array(waveform).astype(np.float64)
    spectrum = np.fft.rfft(data - data.mean())
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, len(spectrum))
    scrambled = np.abs(spectrum) * np.exp(1j * phase)
    scrambled[0] = 0
    scrambled[-1] = abs(spectrum[-1]) * rng.choice((-1, 1))
    return _normalized_audio(np.fft.irfft(scrambled, n=FRAMES))


def extended_signature(waveform: np.ndarray | torch.Tensor) -> np.ndarray | None:
    """Scale-invariant spectral and temporal features; no learned weights."""
    spectral = spectral_signature(waveform)
    if spectral is None:
        return None
    signal = _wave_array(waveform).astype(np.float64)
    signal -= signal.mean()
    energy = np.mean(signal ** 2)
    signal /= np.sqrt(energy)
    absolute = np.abs(signal)
    # The log transform bounds the leverage of extremely impulsive controls.
    amplitude = np.log1p([
        *np.quantile(absolute, (0.25, 0.5, 0.75, 0.9, 0.99)),
        np.mean(signal ** 4), absolute.max(),
    ])
    correlations = np.array([
        np.mean(signal[:-lag] * signal[lag:]) for lag in LAGS
    ])
    frames = np.lib.stride_tricks.sliding_window_view(signal, 256)[::128]
    frame_energy = np.mean(frames ** 2, axis=1)
    envelope = np.log10(np.maximum(frame_energy, 1e-8))
    increments = np.diff(envelope)
    power = np.abs(np.fft.rfft(frames * np.hanning(257)[:-1], axis=-1)) ** 2
    distributions = power / np.maximum(power.sum(axis=1, keepdims=True), FLOOR)
    flux = np.abs(np.diff(distributions, axis=0)).sum(axis=1)
    flatness = np.exp(np.log(np.maximum(distributions, FLOOR)).mean(axis=1))
    flatness *= distributions.shape[1]
    temporal = np.array([
        np.std(envelope), np.quantile(envelope, 0.1), np.quantile(envelope, 0.9),
        np.mean(np.abs(increments)), np.std(increments),
        np.mean(flux), np.std(flux), np.mean(flatness), np.std(flatness),
    ])
    # Spectral terms are in dB; express the other feature families on a
    # comparable nominal scale before applying pooled within-group scaling.
    result = np.concatenate((spectral, 10 * amplitude, 10 * correlations, 10 * temporal))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite open-set feature vector")
    return result


def _features(panel: np.ndarray) -> np.ndarray:
    rows = [extended_signature(wave) for wave in panel.reshape(-1, FRAMES)]
    if any(row is None for row in rows):
        raise ValueError("Unexpected silence in calibration or fit data")
    return np.stack(rows).reshape(*panel.shape[:2], -1)


def _distances(feature: np.ndarray, centers: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(((feature[..., None, :] - centers) / scale) ** 2, axis=-1))


@dataclass(frozen=True)
class OpenSetDecision:
    source: Decision
    closest_noise: str | None
    noise_distance: float | None
    noise_margin: float | None
    radius_passed: bool
    noise_margin_passed: bool

    def as_dict(self, target: int | None = None) -> dict:
        return {
            **self.source.as_dict(target),
            "closest_noise_family": self.closest_noise,
            "noise_distance": self.noise_distance,
            "noise_margin": self.noise_margin,
            "source_radius_passed": self.radius_passed,
            "noise_margin_passed": self.noise_margin_passed,
        }


@dataclass(frozen=True)
class OpenSetSourceProbe:
    source_centers: np.ndarray
    noise_centers: np.ndarray
    scale: np.ndarray
    radii: np.ndarray
    margin_thresholds: np.ndarray

    def detailed_score(self, waveform: np.ndarray | torch.Tensor) -> OpenSetDecision:
        feature = extended_signature(waveform)
        if feature is None:
            return OpenSetDecision(Decision((), (), False), None, None, None, False, False)
        source_distances = _distances(feature, self.source_centers, self.scale)
        noise_distances = _distances(feature, self.noise_centers, self.scale)
        order = tuple(int(i) for i in np.argsort(source_distances, kind="stable"))
        closest = order[0]
        noise_index = int(np.argmin(noise_distances))
        noise_distance = float(noise_distances[noise_index])
        margin = noise_distance - float(source_distances[closest])
        radius_ok = bool(source_distances[closest] <= self.radii[closest])
        margin_ok = bool(margin > self.margin_thresholds[closest])
        return OpenSetDecision(
            Decision(tuple(map(float, source_distances)), order, radius_ok and margin_ok),
            NOISE_FAMILIES[noise_index], noise_distance, margin, radius_ok, margin_ok,
        )

    def score(self, waveform: np.ndarray | torch.Tensor) -> Decision:
        """Compatible with the existing DDIM generated-waveform assay."""
        return self.detailed_score(waveform).source


def fit_probe(
    source_fit: np.ndarray, source_calibration: np.ndarray,
    noise_fit: np.ndarray, noise_calibration: np.ndarray,
) -> OpenSetSourceProbe:
    """Fit centroids, calibrate radii and strict negative margins, no holdout."""
    for panel, count in (
        (source_fit, len(CATEGORIES)), (source_calibration, len(CATEGORIES)),
        (noise_fit, len(NOISE_FAMILIES)), (noise_calibration, len(NOISE_FAMILIES)),
    ):
        if (not isinstance(panel, np.ndarray) or panel.ndim != 3
                or panel.shape[0] != count or panel.shape[2] != FRAMES
                or not 2 <= panel.shape[1] <= 8):
            raise ValueError("Invalid bounded source/noise reference panel")
    sf, sc, nf, nc = map(_features, (source_fit, source_calibration, noise_fit, noise_calibration))
    sources, noises = sf.mean(axis=1), nf.mean(axis=1)
    residuals = np.concatenate((
        (sf - sources[:, None]).reshape(-1, sf.shape[-1]),
        (nf - noises[:, None]).reshape(-1, nf.shape[-1]),
    ))
    scale = np.maximum(np.sqrt(np.mean(residuals ** 2, axis=0)), 2.0)
    calibrated = _distances(sc, sources, scale)
    own = calibrated[np.arange(len(CATEGORIES)), :, np.arange(len(CATEGORIES))]
    radii = np.maximum(1.5 * own.max(axis=1), 0.75)
    negatives = nc.reshape(-1, nc.shape[-1])
    ds = _distances(negatives, sources, scale)
    dn = _distances(negatives, noises, scale).min(axis=1)
    nearest = ds.argmin(axis=1)
    margins = dn - ds[np.arange(len(negatives)), nearest]
    thresholds = np.zeros(len(CATEGORIES), dtype=np.float64)
    # Strict > rejects every observed calibration negative even at a tie.
    # No positive holdout labels or generated model outputs enter calibration.
    for label in range(len(CATEGORIES)):
        selected = margins[nearest == label]
        if len(selected):
            thresholds[label] = max(0.0, float(selected.max()))
    for array in (sources, noises, scale, radii, thresholds):
        array.setflags(write=False)
    return OpenSetSourceProbe(sources, noises, scale, radii, thresholds)


def reference_panels() -> tuple[dict, dict, dict]:
    """Return disjoint source/noise panels with exact waveform hash checks."""
    seeds = TRAIN_SEEDS + VALIDATION_SEEDS + tuple(
        seed for split in (REFERENCE_SEEDS, NOISE_SEEDS) for group in split.values() for seed in group
    )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Source, noise or generator seed overlap")
    sources, source_hashes = _reference_data()
    noises = {
        split: np.stack([[noise_wave(family, seed) for seed in group]
                         for family in NOISE_FAMILIES])
        for split, group in NOISE_SEEDS.items()
    }
    noise_hashes = {
        split: tuple(hashlib.sha256(wave.tobytes()).hexdigest()
                     for wave in panel.reshape(-1, FRAMES))
        for split, panel in noises.items()
    }
    all_hashes = [digest for groups in (source_hashes, noise_hashes)
                  for panel in groups.values() for digest in panel]
    if len(set(all_hashes)) != len(all_hashes):
        raise ValueError("Exact waveform overlap in source or noise references")
    return sources, noises, {"source": source_hashes, "noise": noise_hashes}


def evaluate_probe(probe: OpenSetSourceProbe, sources: np.ndarray, noises: np.ndarray) -> dict:
    """Evaluate only the untouched holdout; report per-category coverage."""
    source_rows = [
        {"category": category, "seed": seed, **probe.detailed_score(wave).as_dict(label)}
        for label, category in enumerate(CATEGORIES)
        for seed, wave in zip(REFERENCE_SEEDS["holdout"], sources[label], strict=True)
    ]
    noise_rows = [
        {"family": family, "seed": seed, **probe.detailed_score(wave).as_dict()}
        for label, family in enumerate(NOISE_FAMILIES)
        for seed, wave in zip(NOISE_SEEDS["holdout"], noises[label], strict=True)
    ]
    by_category = {
        category: {
            "cases": len(REFERENCE_SEEDS["holdout"]),
            "correct_accepted": sum(
                row["category"] == category and row["target_rank"] == 1
                and row["within_reference_radius"] for row in source_rows
            ),
        }
        for category in CATEGORIES
    }
    by_noise_family = {
        family: {
            "cases": len(NOISE_SEEDS["holdout"]),
            "false_acceptances": sum(
                row["family"] == family and row["within_reference_radius"] for row in noise_rows
            ),
        }
        for family in NOISE_FAMILIES
    }
    correct = sum(row["correct_accepted"] for row in by_category.values())
    false_acceptances = sum(row["false_acceptances"] for row in by_noise_family.values())
    silence = probe.detailed_score(np.zeros(FRAMES, dtype=np.float32))
    if silence.source.accepted or silence.source.order:
        raise ValueError("Silent audio must abstain")
    gate = (
        correct >= 12 and all(row["correct_accepted"] >= 1 for row in by_category.values())
        and false_acceptances == 0
    )
    return {
        "source_cases": len(source_rows), "correct_accepted_sources": correct,
        "noise_cases": len(noise_rows), "false_accepted_noise": false_acceptances,
        "by_source_category": by_category, "by_noise_family": by_noise_family,
        "source_decisions": source_rows, "noise_decisions": noise_rows,
        "silence_abstains": True,
        "gate": {
            "minimum_correct_accepted_sources": 12,
            "minimum_per_category": 1, "maximum_noise_false_acceptances": 0,
            "passed": gate,
        },
    }


def baseline_comparison(sources: dict, noises: dict) -> dict:
    """Evaluate the old probe on exactly the same extended held-out controls."""
    old = _fit(sources["fit"], sources["calibration"])
    correct = sum(
        (d := old.score(wave)).accepted and d.order[0] == label
        for label, rows in enumerate(sources["holdout"]) for wave in rows
    )
    by_noise = {
        family: sum(old.score(wave).accepted for wave in rows)
        for family, rows in zip(NOISE_FAMILIES, noises["holdout"], strict=True)
    }
    return {
        "correct_accepted_sources": correct,
        "false_accepted_noise": sum(by_noise.values()),
        "false_acceptances_by_noise_family": by_noise,
    }
