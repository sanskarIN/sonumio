"""Differentiable, conditioned additive synthesizer for pitched note research.

The neural network predicts bounded harmonic amplitudes, two oscillator-bank
frequency controls and smooth note-edge envelopes. Waveform synthesis itself
uses differentiable mathematics. This is an independent CPU experiment, not
a pretrained model or an open-vocabulary music generation system.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from research.instrument_note_data import (
    INSTRUMENTS, MAX_BATCH, MIN_PITCH, MAX_PITCH, MIN_VELOCITY, MAX_VELOCITY,
    MIN_FRAMES, MAX_FRAMES,
)
from sonumio.audio import SAMPLE_RATE

PROTOCOL = "sonumio-learned-harmonic-instruments-v1"
WIDTHS = (32, 48, 64)
BANKS = 2
PARTIALS = 8
ANCHORS = 16
VOICES = BANKS * PARTIALS
CONTROL_COUNT = 7
NYQUIST_GUARD = SAMPLE_RATE * .46
MAX_BATCH_SAMPLES = MAX_BATCH * MAX_FRAMES


def _controls(instruments, pitches, velocities, frames):
    if type(frames) is not int or not MIN_FRAMES <= frames <= MAX_FRAMES:
        raise ValueError("Invalid learned instrument frame count")
    if (not isinstance(instruments, torch.Tensor) or instruments.device.type != "cpu"
            or instruments.dtype != torch.long or instruments.ndim != 1
            or not 1 <= len(instruments) <= MAX_BATCH
            or len(instruments) * frames > MAX_BATCH_SAMPLES):
        raise ValueError("Invalid learned instrument batch")
    for value, low, high, label in (
        (instruments, 0, len(INSTRUMENTS) - 1, "identity"),
        (pitches, MIN_PITCH, MAX_PITCH, "pitch"),
        (velocities, MIN_VELOCITY, MAX_VELOCITY, "velocity"),
    ):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != torch.long or value.shape != instruments.shape
                or bool(((value < low) | (value > high)).any())):
            raise ValueError("Invalid learned instrument " + label)
    return len(instruments)


def _normalized_clock(frames):
    return torch.arange(frames, dtype=torch.float32) / (frames - 1)


def _edge_envelope(clock, attack, release):
    """Continuous differentiable ramps on a normalized inclusive note clock."""
    rising = torch.clamp(clock[None, :] / attack[:, None], 0., 1.)
    falling = torch.clamp((1. - clock[None, :]) / release[:, None], 0., 1.)
    return torch.sin(rising * (math.pi / 2)).square() * torch.sin(
        falling * (math.pi / 2)).square()


@dataclass(frozen=True)
class HarmonicParameters:
    amplitudes: torch.Tensor
    detune_cents: torch.Tensor
    inharmonicity: torch.Tensor
    vibrato_depth: torch.Tensor
    attack_fraction: torch.Tensor
    release_fraction: torch.Tensor


class HarmonicInstrument(nn.Module):
    """Predict oscillator-bank parameters from instrument and note controls."""

    def __init__(self, width=48):
        super().__init__()
        if type(width) is not int or width not in WIDTHS:
            raise ValueError("Unsupported harmonic instrument width")
        self.width = width
        self.embedding = nn.Embedding(len(INSTRUMENTS), width)
        self.network = nn.Sequential(
            nn.Linear(width + 3, 2 * width),
            nn.SiLU(),
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, 2 * width),
            nn.SiLU(),
        )
        self.amplitudes = nn.Linear(2 * width, (VOICES + 1) * ANCHORS)
        self.controls = nn.Linear(2 * width, CONTROL_COUNT)
        self.register_buffer("harmonics", torch.arange(1, PARTIALS + 1, dtype=torch.float32))

    def predict_parameters(self, instruments, pitches, velocities, frames):
        count = _controls(instruments, pitches, velocities, frames)
        numeric = torch.stack((
            (pitches.to(torch.float32) - 62.) / 26.,
            velocities.to(torch.float32) / 1000.,
            torch.full((count,), math.log2(frames / MIN_FRAMES) / 2., dtype=torch.float32),
        ), dim=1)
        hidden = self.network(torch.cat((self.embedding(instruments), numeric), dim=1))
        logits = self.amplitudes(hidden).reshape(count, VOICES + 1, ANCHORS)
        # One silent channel permits arbitrary energy attenuation. Softmax
        # guarantees the sum of sounding amplitudes is at most one.
        gains = torch.softmax(logits, dim=1)[:, :VOICES, :]
        control = self.controls(hidden)
        detune = torch.stack((torch.zeros_like(control[:, 0]),
                              8. * torch.tanh(control[:, 0])), dim=1)
        stretch = .0003 * torch.sigmoid(control[:, 1:3])
        vibrato = .15 * torch.sigmoid(control[:, 3:5])
        attack = .0005 + .499 * torch.sigmoid(control[:, 5])
        release = .0005 + .499 * torch.sigmoid(control[:, 6])
        result = HarmonicParameters(gains, detune, stretch, vibrato, attack, release)
        if any(not bool(torch.isfinite(value).all()) for value in (
            result.amplitudes, result.detune_cents, result.inharmonicity,
            result.vibrato_depth, result.attack_fraction, result.release_fraction,
        )):
            raise ValueError("Nonfinite learned harmonic parameters")
        return result

    def forward(self, instruments, pitches, velocities, frames):
        parameters = self.predict_parameters(instruments, pitches, velocities, frames)
        count = len(instruments)
        clock = torch.arange(frames, dtype=torch.float32) / SAMPLE_RATE
        normalized = _normalized_clock(frames)
        fundamental = 440. * torch.exp2((pitches.to(torch.float32) - 69.) / 12.)
        bank_frequency = fundamental[:, None] * torch.exp2(parameters.detune_cents / 1200.)
        harmonic = self.harmonics[None, None, :]
        frequency = bank_frequency[:, :, None] * harmonic * torch.sqrt(
            1. + parameters.inharmonicity[:, :, None] * harmonic.square()
        )
        # Conservative oscillator cutoff. Bounded phase modulation can have
        # small sidebands; this is not a proof of strict alias-free synthesis.
        valid = frequency < NYQUIST_GUARD
        modulation = torch.sin(2. * math.pi * 5.1 * clock)[None, None, None, :]
        phase = (2. * math.pi * frequency[:, :, :, None] * clock[None, None, None, :]
                 + harmonic[:, :, :, None] * parameters.vibrato_depth[:, :, None, None] * modulation)
        waves = torch.sin(phase) * valid[:, :, :, None]
        envelope = F.interpolate(parameters.amplitudes, size=frames, mode="linear",
                                 align_corners=True).reshape(count, BANKS, PARTIALS, frames)
        mixed = (waves * envelope).sum(dim=(1, 2))
        edges = _edge_envelope(normalized, parameters.attack_fraction, parameters.release_fraction)
        waveform = mixed * edges * (velocities.to(torch.float32) / 1000.)[:, None]
        if waveform.shape != (count, frames) or not bool(torch.isfinite(waveform).all()):
            raise ValueError("Nonfinite learned instrument waveform")
        return waveform


def initialize_model(width, seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid learned instrument initialization seed")
    # Do not change the caller's global RNG trajectory.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return HarmonicInstrument(width)


def fixed_sine(instruments, pitches, velocities, frames):
    """Untrained pitch/velocity/duration control, identical for all timbres."""
    count = _controls(instruments, pitches, velocities, frames)
    clock = torch.arange(frames, dtype=torch.float32) / SAMPLE_RATE
    frequency = 440. * torch.exp2((pitches.to(torch.float32) - 69.) / 12.)
    attack = torch.full((count,), min(.45, 160. / frames), dtype=torch.float32)
    release = torch.full((count,), min(.45, 1200. / frames), dtype=torch.float32)
    edge = _edge_envelope(_normalized_clock(frames), attack, release)
    return (.6 * velocities.to(torch.float32) / 1000.)[:, None] * edge * torch.sin(
        2. * math.pi * frequency[:, None] * clock[None, :]
    )


def _wave_pair(predicted, target):
    if (not isinstance(predicted, torch.Tensor) or predicted.device.type != "cpu"
            or predicted.dtype != torch.float32 or predicted.ndim != 2
            or not 1 <= predicted.shape[0] <= MAX_BATCH
            or not MIN_FRAMES <= predicted.shape[1] <= MAX_FRAMES
            or not isinstance(target, torch.Tensor) or target.shape != predicted.shape
            or target.device.type != "cpu" or target.dtype != torch.float32
            or not bool(torch.isfinite(predicted).all())
            or not bool(torch.isfinite(target).all())):
        raise ValueError("Invalid instrument waveform comparison")
    return predicted, target


def _relative_wave_mse(predicted, target):
    error = (predicted - target).square().mean(dim=1)
    power = target.square().mean(dim=1).clamp_min(1e-8)
    return error / power


def _spectral_error(predicted, target):
    """Average relative compressed-magnitude error at two STFT resolutions."""
    rows = []
    for fft in (256, 1024):
        window = torch.hann_window(fft, dtype=torch.float32)
        kwargs = dict(n_fft=fft, hop_length=fft // 4, win_length=fft,
                      window=window, center=False, normalized=True, return_complex=True)
        actual = torch.log1p(20. * torch.stft(predicted, **kwargs).abs())
        reference = torch.log1p(20. * torch.stft(target, **kwargs).abs())
        numerator = (actual - reference).square().mean(dim=(1, 2))
        denominator = reference.square().mean(dim=(1, 2)).clamp_min(1e-8)
        rows.append(numerator / denominator)
    return torch.stack(rows).mean(dim=0)


def instrument_loss(predicted, target):
    predicted, target = _wave_pair(predicted, target)
    waveform = _relative_wave_mse(predicted, target).mean()
    spectral = _spectral_error(predicted, target).mean()
    result = waveform + .25 * spectral
    if not bool(torch.isfinite(result)):
        raise ValueError("Nonfinite learned instrument loss")
    return result


@torch.no_grad()
def note_metrics(predicted, target):
    predicted, target = _wave_pair(predicted, target)
    power = target.square().mean(dim=1).clamp_min(1e-8)
    predicted_power = predicted.square().mean(dim=1)
    cosine_denom = torch.sqrt(predicted_power * power).clamp_min(1e-8)
    metrics = {
        "relative_wave_mse": _relative_wave_mse(predicted, target),
        "relative_spectral_error": _spectral_error(predicted, target),
        "rms_ratio": torch.sqrt(predicted_power / power),
        "waveform_cosine": (predicted * target).mean(dim=1) / cosine_denom,
        "peak": predicted.abs().amax(dim=1),
        "clipping_fraction": (predicted.abs() >= 1.).to(torch.float32).mean(dim=1),
    }
    if any(not bool(torch.isfinite(value).all()) for value in metrics.values()):
        raise ValueError("Nonfinite learned instrument diagnostic")
    return metrics
