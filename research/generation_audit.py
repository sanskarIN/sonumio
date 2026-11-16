"""Bounded in-memory audio-generation audit for the experimental Sonumio DDPMs.

This checks *actual generated waveforms* after training, not merely denoising
loss. It deliberately avoids claims of perceptual quality, prompt correctness,
copyright compliance, originality, FAD, CLAP, speech or Sonumio-1 readiness.
No WAVs, neural parameters, gradients, embeddings or private data are saved.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch

from research.latent_audio import (
    FRAMES, LatentResearchModel, sample_latent, render_seeded_waveform,
)
from research.text_audio import ByteTextTokenizer, TextWaveDenoiser
from research.wave_diffusion import CATEGORIES, DiffusionSchedule, SAMPLE_RATE

PROTOCOL = 'in-memory-generated-waveform-numerical-audit-v1'
# Fixed first-party DSP audio names; never add external voice or copyrighted audio.
PAIRS = (('rain', 'बारिश', 'rain'),
         ('wind', 'हवा', 'wind'))
AUDIT_SEEDS = (51, 52)
MAX_AUDIT_SEEDS = 4


@dataclass(frozen=True)
class WaveStatistics:
    sha256: str
    rms: float
    peak: float
    clipping_fraction: float
    near_silence_fraction: float
    spectral_centroid_hz: float
    low_frequency_energy_fraction: float
    spectral_flatness: float

    def to_dict(self) -> dict:
        return {
            'sha256': self.sha256,
            'rms': self.rms,
            'peak': self.peak,
            'clipping_fraction': self.clipping_fraction,
            'near_silence_fraction': self.near_silence_fraction,
            'spectral_centroid_hz': self.spectral_centroid_hz,
            'low_frequency_energy_fraction': self.low_frequency_energy_fraction,
            'spectral_flatness': self.spectral_flatness,
        }


def _wave_array(waveform: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(waveform, torch.Tensor):
        if waveform.device.type != 'cpu' or waveform.requires_grad:
            raise ValueError('Audit input must be detached CPU audio')
        waveform = waveform.numpy()
    if (not isinstance(waveform, np.ndarray)
            or waveform.shape != (FRAMES,)
            or waveform.dtype.kind != 'f'
            or not np.isfinite(waveform).all()
            or (np.abs(waveform) > 1.0001).any()):
        raise ValueError('Audio must be finite 8-kHz mono 8192-frame float waveform')
    return np.asarray(waveform, dtype=np.float32)


def waveform_statistics(waveform: torch.Tensor | np.ndarray) -> dict:
    """Deterministic waveform/spectral diagnostics; NOT a listener rating."""
    raw = _wave_array(waveform)
    data = raw.astype(np.float64)
    rms = float(np.sqrt(np.square(data).mean()))
    peak = float(np.abs(data).max())
    window = np.hanning(FRAMES + 1)[:-1]
    spectrum = np.abs(np.fft.rfft(data * window))
    power = np.square(spectrum)
    frequencies = np.fft.rfftfreq(FRAMES, d=1 / SAMPLE_RATE)
    total = float(power.sum())
    centroid = float((frequencies * power).sum() / total) if total > 1e-14 else 0.0
    low_fraction = float(power[frequencies < 300].sum() / total) if total > 1e-14 else 0.0
    # Spectral flatness 0 for strict silence; geometric/arithmetic mean.
    flatness = float(np.exp(np.log(power + 1e-12).mean()) /
                     (power.mean() + 1e-12)) if total > 1e-14 else 0.0
    metrics = WaveStatistics(
        sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
        rms=rms, peak=peak,
        clipping_fraction=float((np.abs(data) >= 0.999).mean()),
        near_silence_fraction=float((np.abs(data) < 1e-5).mean()),
        spectral_centroid_hz=centroid,
        low_frequency_energy_fraction=low_fraction,
        spectral_flatness=flatness,
    ).to_dict()
    if not all(math.isfinite(v) for k, v in metrics.items() if k != 'sha256'):
        raise ValueError('Nonfinite generated waveform statistics')
    return metrics


def _compare_waveforms(left: torch.Tensor, right: torch.Tensor) -> dict:
    """Same seed, different prompt: describe a waveform change, not meaning."""
    a, b = _wave_array(left).astype(np.float64), _wave_array(right).astype(np.float64)
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {
        'mean_absolute_difference': float(np.abs(a - b).mean()),
        'waveform_cosine': max(-1.0, min(1.0, dot / norm)) if norm > 1e-12 else 0.0,
        'exact_same_waveform': bool(np.array_equal(a, b)),
    }


@torch.no_grad()
def sample_raw_full(raw: TextWaveDenoiser, schedule: DiffusionSchedule,
                    caption: str, *, seed: int, guidance: float = 1.0) -> torch.Tensor:
    """Sample the raw model trained on 8192-frame clips (short CLI uses 2048).

    An explicit torch.Generator makes the noise path seed-reproducible without
    changing global RNG state. Restore caller's train/eval mode.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Sampling seed must be an unsigned 32-bit integer')
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or not 0 <= guidance <= 4:
        raise ValueError('Guidance must be finite within [0, 4]')
    if type(schedule.steps) is not int or not 8 <= schedule.steps <= 256:
        raise ValueError('Invalid bounded DDPM sampling schedule')
    device = next(raw.parameters()).device
    if device.type != 'cpu' or schedule.betas.device.type != 'cpu':
        raise ValueError('This generation audit supports CPU only')
    tokens = ByteTextTokenizer().encode([caption])
    blank = torch.zeros_like(tokens)
    generator = torch.Generator().manual_seed(seed)
    wave = torch.randn((1, 1, FRAMES), generator=generator)
    training = raw.training
    raw.eval()
    try:
        for index in range(schedule.steps - 1, -1, -1):
            step = torch.tensor([index], dtype=torch.long)
            predicted = raw(wave, step, tokens)
            if guidance != 1.0:
                null = raw(wave, step, blank)
                predicted = null + guidance * (predicted - null)
            a, b, bar = schedule.alphas[index], schedule.betas[index], schedule.alpha_bars[index]
            mean = (wave - b * predicted / (1 - bar).sqrt()) / a.sqrt()
            if index:
                variance = b * (1 - schedule.alpha_bars[index - 1]) / (1 - bar)
                wave = mean + variance.sqrt() * torch.randn(
                    wave.shape, generator=generator)
            else:
                wave = mean
    finally:
        raw.train(training)
    return wave.clamp(-1, 1)[0, 0].cpu()


@torch.no_grad()
def generation_audit(raw: TextWaveDenoiser, latent: LatentResearchModel,
                     schedule: DiffusionSchedule, *,
                     seeds: tuple[int, ...] = AUDIT_SEEDS,
                     guidance: float = 1.0) -> dict:
    """Numerically inspect actual raw/latent/DDPM outputs in RAM only.

    Both arms are evaluated for the same fixed bilingual prompt pairs and
    sampling seeds. References are independent held-out first-party DSP.
    Audio is never written, uploaded or serialized to JSON.
    """
    if (type(seeds) is not tuple or not 1 <= len(seeds) <= MAX_AUDIT_SEEDS
            or len(set(seeds)) != len(seeds) or any(type(seed) is not int
                or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError('Provide 1..4 unique unsigned sampling seeds')
    if type(guidance) not in (int, float) or not math.isfinite(guidance) or not 0 <= guidance <= 4:
        raise ValueError('Guidance must be finite within [0, 4]')
    if any(next(network.parameters()).device.type != 'cpu'
           for network in (raw, latent)):
        raise ValueError('Generation audit requires CPU models')
    if schedule.betas.device.type != 'cpu':
        raise ValueError('Generation audit requires CPU diffusion schedule')
    original_mode = latent.training
    gathered = {'raw': [], 'latent': []}
    contrasts = {'raw': [], 'latent': []}
    references = {}
    try:
        for en, hi, category in PAIRS:
            members = (CATEGORIES.index(category),)
            # Reuse fixed held-out DSP recordings as numerical feature references.
            reference = render_seeded_waveform(members, 101)
            references[category] = waveform_statistics(reference)
        for seed in seeds:
            for arm in ('raw', 'latent'):
                output = {}
                for english, hindi, category in PAIRS:
                    for caption, language in ((english, 'en'), (hindi, 'hi')):
                        if arm == 'raw':
                            waveform = sample_raw_full(raw, schedule, caption,
                                                       seed=seed, guidance=guidance)
                        else:
                            latent.eval()
                            waveform = sample_latent(latent, schedule, caption,
                                                     seed=seed, guidance=guidance)
                        statistics = waveform_statistics(waveform)
                        gathered[arm].append({
                            'seed': seed, 'category': category,
                            'language': language, 'stats': statistics,
                        })
                        output[(category, language)] = waveform
                for language in ('en', 'hi'):
                    values = _compare_waveforms(output[('rain', language)],
                                                output[('wind', language)])
                    contrasts[arm].append({'seed': seed, 'language': language, **values})
                for category in ('rain', 'wind'):
                    values = _compare_waveforms(output[(category, 'en')],
                                                output[(category, 'hi')])
                    contrasts[arm].append({
                        'seed': seed, 'category': category,
                        'language': 'en-vs-hi', **values,
                    })
    finally:
        latent.train(original_mode)
    return {
        'protocol': PROTOCOL,
        'sample_rate': SAMPLE_RATE, 'duration_seconds': FRAMES / SAMPLE_RATE,
        'seeds': list(seeds), 'guidance': float(guidance),
        'prompts_by_category': {
            category: {'en': en, 'hi': hi}
            for en, hi, category in PAIRS
        },
        'reference_dsp_heldout_seed': 101,
        'reference_dsp_metrics': references,
        'generated': gathered,
        'paired_prompt_contrasts': contrasts,
        'audio_files_saved': False, 'neural_weights_saved': False,
        'interpretation': [
            'Generated wave health and prompt sensitivity, not prompt accuracy',
            'A changed waveform does not prove the requested sound is audible',
            'Same-source bilingual caption differences do not prove language understanding',
            'DSP numerical reference features do not measure distributional fidelity',
            'No FAD/CLAP, copyright, memorization, listener or voice safety scores',
        ],
    }
