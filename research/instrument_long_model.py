"""Windowed differentiable harmonic synthesis for sustained musical notes.

The original short-note architecture is preserved. The extended renderer
conditions on the complete note duration and uses absolute phase and
envelope coordinates in every bounded window, rather than restarting
oscillators or release ramps at block boundaries.
"""
from __future__ import annotations

import math
import torch

from research.instrument_harmonic_model import (
    HarmonicInstrument, ANCHORS, BANKS, PARTIALS, NYQUIST_GUARD,
    _controls, _edge_envelope,
)
from research.instrument_note_data import MIN_FRAMES, MAX_BATCH
from sonumio.audio import SAMPLE_RATE

PROTOCOL = "sonumio-windowed-harmonic-instrument-v1"
MAX_NOTE_FRAMES = 4 * SAMPLE_RATE
MAX_WINDOW = 8192
MAX_WINDOW_SAMPLES = MAX_BATCH * MAX_WINDOW


def validate_window(instruments, pitches, velocities, frames, start, count):
    size = _controls(instruments, pitches, velocities, frames,
                     max_frames=MAX_NOTE_FRAMES,
                     max_batch_samples=MAX_BATCH * MAX_NOTE_FRAMES)
    if (type(start) is not int or type(count) is not int
            or not 0 <= start < frames or not 1 <= count <= MAX_WINDOW
            or start + count > frames or size * count > MAX_WINDOW_SAMPLES):
        raise ValueError("Invalid sustained note window")
    return size


def _window_clock(frames, start, count):
    absolute = torch.arange(start, start + count, dtype=torch.float32)
    return absolute / SAMPLE_RATE, absolute / (frames - 1)


def interpolate_anchors(anchors, normalized):
    """Evaluate linear control anchors at absolute normalized note coordinates."""
    if (not isinstance(anchors, torch.Tensor) or anchors.ndim != 3
            or anchors.shape[-1] != ANCHORS or anchors.dtype != torch.float32
            or anchors.device.type != "cpu"
            or not isinstance(normalized, torch.Tensor) or normalized.ndim != 1
            or normalized.dtype != torch.float32 or normalized.device.type != "cpu"
            or not 1 <= normalized.numel() <= MAX_WINDOW
            or not bool(torch.isfinite(anchors).all())
            or not bool(torch.isfinite(normalized).all())
            or bool(((normalized < 0.) | (normalized > 1.)).any())):
        raise ValueError("Invalid sustained note anchor coordinates")
    position = normalized * (ANCHORS - 1)
    left = position.floor().to(torch.long).clamp(0, ANCHORS - 2)
    fraction = position - left
    return anchors[:, :, left] * (1. - fraction) + anchors[:, :, left + 1] * fraction


class WindowedHarmonicInstrument(HarmonicInstrument):
    """Same learnable architecture, explicit full-duration window rendering."""

    def predict_parameters(self, instruments, pitches, velocities, frames):
        size = _controls(instruments, pitches, velocities, frames,
                         max_frames=MAX_NOTE_FRAMES,
                         max_batch_samples=MAX_BATCH * MAX_NOTE_FRAMES)
        return self._condition_parameters(instruments, pitches, velocities, frames, size)

    def render_window(self, instruments, pitches, velocities, frames, start, count):
        size = validate_window(instruments, pitches, velocities, frames, start, count)
        parameters = self.predict_parameters(instruments, pitches, velocities, frames)
        clock, normalized = _window_clock(frames, start, count)
        fundamental = 440. * torch.exp2((pitches.to(torch.float32) - 69.) / 12.)
        bank_frequency = fundamental[:, None] * torch.exp2(parameters.detune_cents / 1200.)
        harmonic = self.harmonics[None, None, :]
        frequency = bank_frequency[:, :, None] * harmonic * torch.sqrt(
            1. + parameters.inharmonicity[:, :, None] * harmonic.square()
        )
        valid = frequency < NYQUIST_GUARD
        modulation = torch.sin(2. * math.pi * 5.1 * clock)[None, None, None, :]
        phase = (2. * math.pi * frequency[:, :, :, None] * clock[None, None, None, :]
                 + harmonic[:, :, :, None] * parameters.vibrato_depth[:, :, None, None] * modulation)
        waves = torch.sin(phase) * valid[:, :, :, None]
        envelope = interpolate_anchors(parameters.amplitudes, normalized).reshape(
            size, BANKS, PARTIALS, count)
        mixed = (waves * envelope).sum(dim=(1, 2))
        edge = _edge_envelope(normalized, parameters.attack_fraction, parameters.release_fraction)
        result = mixed * edge * (velocities.to(torch.float32) / 1000.)[:, None]
        if result.shape != (size, count) or not bool(torch.isfinite(result).all()):
            raise ValueError("Invalid sustained note output")
        return result

    def forward(self, instruments, pitches, velocities, frames):
        # Avoid accidentally materializing a four-second autograd graph.
        if type(frames) is not int or not MIN_FRAMES <= frames <= MAX_WINDOW:
            raise ValueError("Long notes require explicit render_window")
        return self.render_window(instruments, pitches, velocities, frames, 0, frames)


def initialize_windowed(width, seed):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid sustained note initialization seed")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return WindowedHarmonicInstrument(width)


def sine_window(instruments, pitches, velocities, frames, start, count):
    size = validate_window(instruments, pitches, velocities, frames, start, count)
    clock, normalized = _window_clock(frames, start, count)
    frequency = 440. * torch.exp2((pitches.to(torch.float32) - 69.) / 12.)
    attack = torch.full((size,), min(.45, 160. / frames), dtype=torch.float32)
    release = torch.full((size,), min(.45, 1200. / frames), dtype=torch.float32)
    edge = _edge_envelope(normalized, attack, release)
    return (.6 * velocities.to(torch.float32) / 1000.)[:, None] * edge * torch.sin(
        2. * math.pi * frequency[:, None] * clock[None, :])


@torch.no_grad()
def render_complete(model, instruments, pitches, velocities, frames, *, block=4096):
    """Bounded no-grad iterator; callers can consume or encode one block at a time."""
    if not isinstance(model, WindowedHarmonicInstrument):
        raise ValueError("Expected windowed harmonic model")
    if type(block) is not int or not 1 <= block <= MAX_WINDOW:
        raise ValueError("Invalid sustained synthesis block length")
    validate_window(instruments, pitches, velocities, frames, 0, min(block, frames))
    for start in range(0, frames, block):
        yield start, model.render_window(instruments, pitches, velocities, frames,
                                         start, min(block, frames - start))
