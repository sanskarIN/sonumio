"""Bounded, phase-preserving transform representations for audio research.

These are floating-point reconstruction baselines, not production bitstream
codecs. No neural weights, quantization or entropy coding are implied.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from research.latent_audio import FRAMES

BLOCK = 64
MAX_BATCH = 128
BUDGETS = (8, 16, 32, 64)
METHODS = ("dct_low", "dct_ranked", "pca", "mdct_low", "mdct_ranked")


def _waves(waves: np.ndarray) -> np.ndarray:
    if (not isinstance(waves, np.ndarray) or waves.ndim != 2
            or waves.shape[1] != FRAMES or not 1 <= len(waves) <= MAX_BATCH
            or waves.dtype not in (np.float32, np.float64)
            or not np.isfinite(waves).all()):
        raise ValueError("Expected bounded finite mono waveform batch")
    return waves.astype(np.float64, copy=False)


@lru_cache(maxsize=1)
def dct_basis() -> np.ndarray:
    n = np.arange(BLOCK, dtype=np.float64)[:, None]
    k = np.arange(BLOCK, dtype=np.float64)[None, :]
    basis = np.sqrt(2.0 / BLOCK) * np.cos(np.pi / BLOCK * (n + 0.5) * k)
    basis[:, 0] /= np.sqrt(2.0)
    basis.setflags(write=False)
    return basis


@lru_cache(maxsize=1)
def mdct_basis() -> tuple[np.ndarray, np.ndarray]:
    n = np.arange(2 * BLOCK, dtype=np.float64)[:, None]
    k = np.arange(BLOCK, dtype=np.float64)[None, :]
    matrix = np.sqrt(2.0 / BLOCK) * np.cos(
        np.pi / BLOCK * (n + 0.5 + BLOCK / 2) * (k + 0.5)
    )
    window = np.sin(np.pi / (2 * BLOCK) * (n[:, 0] + 0.5))
    matrix.setflags(write=False)
    window.setflags(write=False)
    return matrix, window


def analysis(waves: np.ndarray, family: str) -> np.ndarray:
    """Analyze contiguous blocks or 50%-overlapping sine-windowed blocks."""
    wave = _waves(waves)
    if family == "dct":
        return wave.reshape(len(wave), -1, BLOCK) @ dct_basis()
    if family != "mdct":
        raise ValueError("Unknown transform family")
    matrix, window = mdct_basis()
    padded = np.pad(wave, ((0, 0), (BLOCK, BLOCK)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 2 * BLOCK, axis=-1)[:, ::BLOCK]
    return (frames * window) @ matrix


def synthesis(coefficients: np.ndarray, family: str) -> np.ndarray:
    frames = FRAMES // BLOCK + (1 if family == "mdct" else 0)
    if (family not in ("dct", "mdct") or not isinstance(coefficients, np.ndarray)
            or coefficients.ndim != 3 or coefficients.shape[1:] != (frames, BLOCK)
            or not 1 <= len(coefficients) <= MAX_BATCH
            or coefficients.dtype not in (np.float32, np.float64)
            or not np.isfinite(coefficients).all()):
        raise ValueError("Invalid transform coefficient layout")
    if family == "dct":
        return (coefficients @ dct_basis().T).reshape(len(coefficients), FRAMES)
    matrix, window = mdct_basis()
    blocks = (coefficients @ matrix.T) * window
    result = np.zeros((len(blocks), FRAMES + 2 * BLOCK), dtype=np.float64)
    for index in range(frames):
        offset = index * BLOCK
        result[:, offset:offset + 2 * BLOCK] += blocks[:, index]
    return result[:, BLOCK:-BLOCK]


def _balanced_covariance(blocks: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal category weight irrespective of recording or block count."""
    means, second = [], []
    for category in sorted(set(labels.tolist())):
        values = blocks[labels == category].reshape(-1, BLOCK)
        means.append(values.mean(axis=0))
        second.append(values.T @ values / len(values))
    center = np.mean(means, axis=0)
    covariance = np.mean(second, axis=0) - np.outer(center, center)
    covariance = (covariance + covariance.T) * 0.5
    return center, covariance


def _ranked_indices(coefficients: np.ndarray, labels: np.ndarray) -> tuple[int, ...]:
    energies = np.stack([
        np.mean(coefficients[labels == category] ** 2, axis=(0, 1))
        for category in sorted(set(labels.tolist()))
    ])
    # Normalize by category energy so loud stationary sources cannot
    # monopolize the retained coordinates. No per-recording side information.
    total = energies.sum(axis=1, keepdims=True)
    score = (energies / np.maximum(total, 1e-20)).mean(axis=0)
    return tuple(map(int, np.argsort(-score, kind="stable")))


@dataclass(frozen=True)
class TransformFit:
    dct_order: tuple[int, ...]
    mdct_order: tuple[int, ...]
    pca_center: np.ndarray
    pca_basis: np.ndarray
    pca_eigenvalues: np.ndarray
    training_recordings: int

    @classmethod
    def fit(cls, waves: np.ndarray, labels: np.ndarray) -> TransformFit:
        values = _waves(waves)
        if (not isinstance(labels, np.ndarray) or labels.shape != (len(values),)
                or labels.dtype.kind not in "iu" or set(labels.tolist()) != set(range(6))):
            raise ValueError("Fit requires all six source categories")
        if any(np.count_nonzero(labels == label) < 2 for label in range(6)):
            raise ValueError("At least two training recordings per category required")
        center, covariance = _balanced_covariance(values.reshape(len(values), -1, BLOCK), labels)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(-eigenvalues, kind="stable")
        basis = eigenvectors[:, order].copy()
        # Resolve the arbitrary sign of each eigenvector for reproducibility.
        for col in range(BLOCK):
            pivot = np.argmax(np.abs(basis[:, col]))
            if basis[pivot, col] < 0:
                basis[:, col] *= -1
        eigenvalues = np.maximum(eigenvalues[order], 0)
        for array in (center, basis, eigenvalues):
            array.setflags(write=False)
        return cls(
            _ranked_indices(analysis(values, "dct"), labels),
            _ranked_indices(analysis(values, "mdct"), labels),
            center, basis, eigenvalues, len(values),
        )

    def indices(self, method: str, budget: int) -> tuple[int, ...]:
        if method not in METHODS or type(budget) is not int or budget not in BUDGETS:
            raise ValueError("Unsupported method or coefficient budget")
        if method == "dct_ranked":
            return self.dct_order[:budget]
        if method == "mdct_ranked":
            return self.mdct_order[:budget]
        return tuple(range(budget))

    def encode(self, waves: np.ndarray, method: str, budget: int) -> np.ndarray:
        indices = self.indices(method, budget)
        values = _waves(waves)
        if method == "pca":
            blocks = values.reshape(len(values), -1, BLOCK) - self.pca_center
            return blocks @ self.pca_basis[:, :budget]
        family = "mdct" if method.startswith("mdct") else "dct"
        return analysis(values, family)[..., list(indices)]

    def decode(self, values: np.ndarray, method: str, budget: int) -> np.ndarray:
        indices = self.indices(method, budget)
        frames = FRAMES // BLOCK + int(method.startswith("mdct"))
        if (not isinstance(values, np.ndarray) or values.ndim != 3
                or values.shape[1:] != (frames, budget)
                or not 1 <= len(values) <= MAX_BATCH
                or values.dtype not in (np.float32, np.float64)
                or not np.isfinite(values).all()):
            raise ValueError("Invalid retained coefficient tensor")
        if method == "pca":
            blocks = values @ self.pca_basis[:, :budget].T + self.pca_center
            return blocks.reshape(len(values), FRAMES)
        full = np.zeros((len(values), frames, BLOCK), dtype=np.float64)
        full[..., list(indices)] = values
        return synthesis(full, "mdct" if method.startswith("mdct") else "dct")

    def describe(self, method: str, budget: int) -> dict:
        indices = self.indices(method, budget)
        frames = FRAMES // BLOCK + int(method.startswith("mdct"))
        return {
            "method": method, "coefficients_per_frame": budget,
            "coefficient_frames": frames, "retained_scalar_values": frames * budget,
            "scalar_compression_ratio": FRAMES / (frames * budget),
            "retained_indices": list(indices),
            "fit_recordings": self.training_recordings,
            "bitrate_established": False,
        }


def reconstruction_metrics(reference: np.ndarray, reconstructed: np.ndarray) -> list[dict]:
    ref, decoded = _waves(reference), _waves(reconstructed)
    if ref.shape != decoded.shape:
        raise ValueError("Mismatched reconstruction batch")
    error = decoded - ref
    results = []
    for a, b, delta in zip(ref, decoded, error, strict=True):
        power = float(np.mean(a ** 2))
        mse = float(np.mean(delta ** 2))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        # A finite floor is disclosed; perfect reconstruction is not infinite JSON.
        snr = 10 * math.log10(power / max(mse, 1e-20)) if power > 1e-20 else None
        results.append({
            "mae": float(np.mean(np.abs(delta))),
            "mse": mse, "relative_squared_error": mse / power if power > 1e-20 else None,
            "snr_db_with_error_floor": snr,
            "waveform_cosine": float(np.clip(np.dot(a, b) / norm, -1, 1)) if norm > 1e-20 else None,
            "out_of_range_fraction": float(np.mean(np.abs(b) > 1)),
            "decoded_rms": float(np.sqrt(np.mean(b ** 2))),
            "decoded_dc": float(b.mean()),
        })
    return results
