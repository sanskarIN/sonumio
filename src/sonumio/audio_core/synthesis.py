"""Original deterministic offline musical-tone and noise reference synthesis.

This is explicitly PROCEDURAL, never a trained Sonumio-1 model. It creates
short local demonstration tones for unit tests and effect evaluation, not
voices, copyrighted melodies or commercial song reproduction.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from .automation import Tempo, envelope_adsr
from .contracts import AudioBuffer, AudioLayout, check_rate, finite_float, make_buffer
from .errors import AudioContractError, AudioLimitError

WAVEFORMS = ("sine", "triangle", "square", "saw", "pulse")
NOISE_COLORS = ("white", "pink_approx", "brown")


def oscillator(frequency_hz: float | np.ndarray, sample_rate: int,
               frames: int, *, waveform: str = "sine", phase: float = 0,
               antialias: bool = True, duty_cycle: float = 0.5
               ) -> np.ndarray:
    """Waveforms through additive bandlimited harmonic expansion.

    High-frequency aliasing is reduced by excluding partials above Nyquist.
    The exact timbre differs from an ideal discontinuous saw/square.
    No hardware RNG is touched. Input frequency modulation is permitted
    as a finite vector of one frequency per output sample.
    """
    check_rate(sample_rate)
    if type(frames) is not int or not 1 <= frames <= 2_000_000:
        raise AudioContractError("Invalid oscillator frame limit")
    initial = finite_float(phase, "oscillator initial phase",
                           minimum=-1e6, maximum=1e6)
    duty = finite_float(duty_cycle, "pulse duty cycle", minimum=.01, maximum=.99)
    if waveform not in WAVEFORMS:
        raise AudioContractError("Unsupported oscillator waveform")
    if type(antialias) is not bool:
        raise AudioContractError("antialias flag must be boolean")
    if isinstance(frequency_hz, np.ndarray):
        if (frequency_hz.shape != (frames,) or frequency_hz.dtype not in
                (np.float32, np.float64) or not np.isfinite(frequency_hz).all()):
            raise AudioContractError("Frequency automation must be finite frame vector")
        freqs = frequency_hz.astype(np.float64)
    else:
        value = finite_float(frequency_hz, "oscillator frequency", minimum=1,
                             maximum=sample_rate / 2 - 1)
        freqs = np.full(frames, value, dtype=np.float64)
    if np.any((freqs < 1) | (freqs >= sample_rate / 2)):
        raise AudioContractError("Oscillator frequency must lie below Nyquist")
    cycles = np.cumsum(freqs / sample_rate, dtype=np.float64)
    cycles -= freqs / sample_rate
    angle = 2 * math.pi * cycles + initial
    if waveform == "sine":
        return np.sin(angle).astype(np.float32)
    if not antialias:
        fraction = np.mod(angle / (2 * math.pi), 1)
        if waveform == "saw":
            values = 2 * fraction - 1
        elif waveform == "triangle":
            values = 4 * np.abs(fraction - .5) - 1
        elif waveform == "square":
            values = np.where(fraction < .5, 1, -1)
        else:
            values = np.where(fraction < duty, 1, -1)
        return values.astype(np.float32)
    # Variable-frequency bandlimiting: mask harmonic contribution above
    # current Nyquist. Cap at 128 partials to keep CPU reference bounded.
    nyquist = sample_rate / 2
    max_partial = min(128, int(nyquist / float(np.min(freqs))))
    output = np.zeros(frames, dtype=np.float64)
    if waveform == "triangle":
        for harmonic in range(1, max_partial + 1, 2):
            amplitude = 8 / (math.pi ** 2 * harmonic ** 2)
            if harmonic % 4 == 3:
                amplitude *= -1
            output += amplitude * np.sin(angle * harmonic) * (
                freqs * harmonic < nyquist
            )
    elif waveform == "square":
        for harmonic in range(1, max_partial + 1, 2):
            output += 4 / (math.pi * harmonic) * np.sin(angle * harmonic) * (
                freqs * harmonic < nyquist
            )
    elif waveform == "saw":
        for harmonic in range(1, max_partial + 1):
            output += (-2 / (math.pi * harmonic)) * np.sin(
                angle * harmonic
            ) * (freqs * harmonic < nyquist)
    else:
        for harmonic in range(1, max_partial + 1):
            output += (2 / (math.pi * harmonic)) * math.sin(
                math.pi * harmonic * duty
            ) * np.cos(angle * harmonic) * (freqs * harmonic < nyquist)
    return np.clip(output, -1, 1).astype(np.float32)


def deterministic_noise(frames: int, channels: int = 1, *,
                        seed: int, color: str = "white",
                        gain: float = 1) -> np.ndarray:
    if (type(frames) is not int or not 1 <= frames <= 2_000_000
            or type(channels) is not int or not 1 <= channels <= 8
            or type(seed) is not int or not 0 <= seed < 2**32
            or color not in NOISE_COLORS):
        raise AudioContractError("Invalid deterministic noise specification")
    amplitude = finite_float(gain, "noise gain", minimum=0, maximum=10)
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((channels, frames), dtype=np.float64)
    if color == "brown":
        # Controlled random walk, remove DC then normalize peak.
        data = np.cumsum(data, axis=1)
        data -= np.mean(data, axis=1, keepdims=True)
    elif color == "pink_approx":
        # Three lowpass one-pole banks: explicitly an approximation.
        output = np.zeros_like(data)
        for coefficient, weight in ((.995, .45), (.9, .3), (.6, .25)):
            previous = np.zeros(channels, dtype=np.float64)
            stage = np.empty_like(data)
            for frame in range(frames):
                previous = coefficient * previous + (1 - coefficient) * data[:, frame]
                stage[:, frame] = previous
            output += weight * stage
        data = output
    peaks = np.max(np.abs(data), axis=1)
    data /= np.maximum(peaks[:, None], 1e-12)
    return (data * amplitude).astype(np.float32)


def midi_to_frequency(note: int, *, concert_a: float = 440) -> float:
    if type(note) is not int or not 0 <= note <= 127:
        raise AudioContractError("MIDI note must be integer 0..127")
    reference = finite_float(concert_a, "concert pitch", minimum=400, maximum=480)
    return reference * 2 ** ((note - 69) / 12)


@dataclass(frozen=True, slots=True)
class SynthNote:
    """A single procedural tone event, not a learned/generated song."""

    pitch: int
    start_frame: int
    duration_frames: int
    velocity: float = 0.7
    waveform: str = "sine"
    attack_frames: int = 64
    decay_frames: int = 128
    sustain: float = 0.8
    release_frames: int = 256
    pan: float = 0

    def __post_init__(self):
        midi_to_frequency(self.pitch)
        if (type(self.start_frame) is not int or self.start_frame < 0
                or type(self.duration_frames) is not int or self.duration_frames < 1):
            raise AudioContractError("Invalid procedural note frame placement")
        if self.waveform not in WAVEFORMS:
            raise AudioContractError("Invalid procedural note waveform")
        for name, n in (("attack", self.attack_frames), ("decay", self.decay_frames),
                        ("release", self.release_frames)):
            if type(n) is not int or n < 0:
                raise AudioContractError(name + " frames must be nonnegative")
        if self.attack_frames + self.decay_frames + self.release_frames > self.duration_frames:
            raise AudioContractError("Procedural ADSR exceeds note duration")
        finite_float(self.velocity, "note velocity", minimum=0, maximum=1)
        finite_float(self.sustain, "note sustain", minimum=0, maximum=1)
        finite_float(self.pan, "note pan", minimum=-1, maximum=1)


def render_synth_score(events: Sequence[SynthNote], *,
                       sample_rate: int = 24000, frames: int = 24000,
                       channels: int = 2, max_polyphony: int = 16,
                       master_gain: float = 0.5) -> AudioBuffer:
    check_rate(sample_rate)
    if (type(frames) is not int or not 1 <= frames <= sample_rate * 120
            or type(channels) is not int or channels not in (1, 2)
            or type(max_polyphony) is not int or not 1 <= max_polyphony <= 64
            or not isinstance(events, (list, tuple)) or len(events) > 512):
        raise AudioContractError("Invalid bounded procedural score")
    master = finite_float(master_gain, "synth score gain", minimum=0, maximum=2)
    if any(not isinstance(x, SynthNote) for x in events):
        raise AudioContractError("Score must contain SynthNote entries")
    for note in events:
        if note.start_frame + note.duration_frames > frames:
            raise AudioContractError("Synth note exceeds score duration")
        if midi_to_frequency(note.pitch) >= sample_rate / 2:
            raise AudioContractError("Synth note pitch beyond Nyquist")
    # Sweep event boundaries to enforce maximum simultaneous note count.
    boundaries = []
    for note in events:
        boundaries.extend(((note.start_frame, 1),
                           (note.start_frame + note.duration_frames, -1)))
    active = 0
    for _, change in sorted(boundaries):
        active += change
        if active > max_polyphony:
            raise AudioLimitError("Score exceeds configured simultaneous polyphony")
    result = np.zeros((channels, frames), dtype=np.float64)
    for note in events:
        wave = oscillator(midi_to_frequency(note.pitch), sample_rate,
                          note.duration_frames, waveform=note.waveform)
        env = envelope_adsr(note.duration_frames,
                            attack=note.attack_frames,
                            decay=note.decay_frames,
                            sustain=note.sustain,
                            release=note.release_frames)
        sample = wave.astype(np.float64) * env * note.velocity
        start = note.start_frame
        stop = start + note.duration_frames
        if channels == 1:
            result[0, start:stop] += sample
        else:
            angle = (note.pan + 1) * math.pi / 4
            result[0, start:stop] += sample * math.cos(angle)
            result[1, start:stop] += sample * math.sin(angle)
    result *= master
    return make_buffer(result.astype(np.float32), sample_rate,
                       source_note="original procedural non-AI synthesizer")


def synth_score_report(events: Sequence[SynthNote], audio: AudioBuffer
                       ) -> dict:
    if not isinstance(audio, AudioBuffer):
        raise AudioContractError("Expected AudioBuffer")
    from .numerics import stable_peak, stable_rms
    return {
        "events": len(events),
        "channels": audio.channels,
        "sample_rate": audio.sample_rate,
        "frames": audio.frames,
        "peak": stable_peak(audio.samples),
        "rms": stable_rms(audio.samples),
        "generator_type": "deterministic procedural synthesis, not learned model",
        "third_party_model_weights": False,
        "audio_files_saved": False,
    }
