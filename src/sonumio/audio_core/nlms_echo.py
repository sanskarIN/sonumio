"""Explicit-reference normalized LMS (NLMS) echo suppression research.

A known mono far-end reference and aligned multichannel microphone PCM
are used to estimate causal short FIR echo paths. Error output is the
microphone minus the predicted reference echo. This is a deterministic
offline model and not a full double-talk-robust teleconferencing AEC.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, MAX_BYTES, MAX_SECONDS,
    check_audio, check_rate, finite_float, make_buffer,
)
from .errors import AudioContractError, AudioLimitError

MAX_ADAPT_TAPS = 4096
MAX_ADAPT_PRODUCTS = 180_000_000
MAX_ADAPT_WORK_BYTES = 192 * 1024 * 1024
MAX_ABS_PCM = 1_000_000.


def _integer(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise AudioContractError(f"{name} must be integer in [{lo}, {hi}]")
    return value


@dataclass(frozen=True, slots=True)
class NlmsSettings:
    taps: int = 128
    step_size: float = .5
    epsilon: float = 1e-6
    leakage: float = 0.
    maximum_weight_l2: float = 100.
    enabled: bool = True

    def __post_init__(self):
        _integer(self.taps, "adaptive taps", 1, MAX_ADAPT_TAPS)
        finite_float(self.step_size, "NLMS step size", minimum=0.001, maximum=1.)
        finite_float(self.epsilon, "NLMS denominator epsilon",
                     minimum=1e-12, maximum=1.)
        finite_float(self.leakage, "NLMS leakage", minimum=0., maximum=.05)
        finite_float(self.maximum_weight_l2, "NLMS weight bound",
                     minimum=.01, maximum=10000.)
        if type(self.enabled) is not bool:
            raise AudioContractError("NLMS adaptation enabled must be boolean")


@dataclass(frozen=True, slots=True)
class NlmsGeometry:
    channels: int
    frames: int
    taps: int
    sample_rate: int

    def __post_init__(self):
        _integer(self.channels, "microphone channels", 1, 8)
        _integer(self.frames, "input frames", 1, 120 * 192000)
        _integer(self.taps, "adaptive taps", 1, MAX_ADAPT_TAPS)
        check_rate(self.sample_rate)
        if self.frames > int(MAX_SECONDS * self.sample_rate):
            raise AudioLimitError("Adaptive echo input exceeds duration budget")
        if self.frames * self.channels * self.taps > MAX_ADAPT_PRODUCTS:
            raise AudioLimitError("Adaptive echo input exceeds MAC work budget")
        if self.estimated_bytes > MAX_ADAPT_WORK_BYTES:
            raise AudioLimitError("Adaptive echo exceeds working memory budget")

    @property
    def estimated_bytes(self):
        return (
            self.frames * self.channels * 8 * 7
            + self.frames * 16 + self.taps * self.channels * 8 * 5
        )


@dataclass(frozen=True, slots=True)
class EchoSummary:
    sample_rate: int
    frames: int
    microphone_channels: int
    taps: int
    adapted_frames: int
    frozen_frames: int
    reference_rms: float
    microphone_rms: float
    residual_rms: float
    echo_reduction_db: float
    maximum_weight_l2: float
    protocol: str = "sonumio-explicit-reference-nlms-v1"

    def as_dict(self):
        obj = asdict(self)
        obj["model_inference"] = False
        obj["double_talk_handling_automatic"] = False
        obj["echo_cancellation_guaranteed"] = False
        obj["clock_alignment_automatic"] = False
        return obj


@dataclass(frozen=True, slots=True, eq=False)
class EchoResult:
    residual: AudioBuffer
    prediction: AudioBuffer
    weights: np.ndarray
    summary: EchoSummary

    def __post_init__(self):
        data = self.weights
        if (not isinstance(data, np.ndarray) or data.ndim != 2
                or data.dtype != np.float64 or not np.isfinite(data).all()
                or data.shape[0] != self.residual.channels):
            raise AudioContractError("Invalid adaptive echo coefficient matrix")
        frozen = np.frombuffer(data.tobytes(), dtype=np.float64)
        object.__setattr__(self, "weights", frozen.reshape(data.shape))


class NlmsEchoStream:
    """One-shot, fixed-clock streaming adaptive FIR.

    push receives equal-duration reference and microphone buffers.
    Adaptation gates are caller-provided booleans; an automatic voice
    activity detector or double-talk classifier is intentionally absent.
    """

    def __init__(self, sample_rate: int, microphone_channels: int,
                 settings: NlmsSettings = NlmsSettings()):
        check_rate(sample_rate)
        if not isinstance(settings, NlmsSettings):
            raise AudioContractError("Expected NlmsSettings")
        _integer(microphone_channels, "microphone_channels", 1, 8)
        self._rate = sample_rate
        self._channels = microphone_channels
        self._settings = settings
        self._hist = np.zeros(settings.taps, dtype=np.float64)
        self._weights = np.zeros((microphone_channels, settings.taps),
                                 dtype=np.float64)
        self._frames = 0
        self._adapted = 0
        self._frozen = 0
        self._ref_energy = 0.
        self._mic_energy = 0.
        self._err_energy = 0.
        self._finished = False

    @property
    def input_frames(self):
        return self._frames

    @property
    def adapted_frames(self):
        return self._adapted

    @property
    def frozen_frames(self):
        return self._frozen

    @property
    def finished(self):
        return self._finished

    def push(self, reference: np.ndarray, microphone: np.ndarray, *,
             freeze: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self._finished:
            raise AudioContractError("NLMS stream already finished")
        check_audio(reference)
        check_audio(microphone)
        source = reference.reshape(1, -1) if reference.ndim == 1 else reference
        mic = microphone.reshape(1, -1) if microphone.ndim == 1 else microphone
        if (source.shape[0] != 1 or mic.shape[0] != self._channels
                or source.shape[1] != mic.shape[1]):
            raise AudioContractError("NLMS requires aligned mono reference and microphones")
        if (float(np.max(np.abs(source))) > MAX_ABS_PCM
                or float(np.max(np.abs(mic))) > MAX_ABS_PCM):
            raise AudioContractError("NLMS PCM amplitude exceeds processing bound")
        frames = source.shape[1]
        mask = np.zeros(frames, dtype=np.bool_) if freeze is None else freeze
        if (not isinstance(mask, np.ndarray) or mask.ndim != 1
                or mask.shape[0] != frames or mask.dtype != np.bool_):
            raise AudioContractError("NLMS freeze mask must be boolean per audio frame")
        NlmsGeometry(self._channels, self._frames + frames,
                     self._settings.taps, self._rate)
        if frames * self._channels * 16 > MAX_BYTES:
            raise AudioLimitError("NLMS push output exceeds budget")
        predicted = np.empty((self._channels, frames), dtype=np.float64)
        residual = np.empty_like(predicted)
        # The fixed reduction and update order ensure chunk-boundary parity.
        for frame in range(frames):
            if self._settings.taps > 1:
                self._hist[1:] = self._hist[:-1]
            self._hist[0] = float(source[0, frame])
            y = self._weights @ self._hist
            e = mic[:, frame].astype(np.float64) - y
            if not np.isfinite(e).all():
                raise AudioContractError("NLMS prediction generated nonfinite residual")
            predicted[:, frame] = y
            residual[:, frame] = e
            if mask[frame] or not self._settings.enabled:
                self._frozen += 1
            else:
                norm = float(np.dot(self._hist, self._hist))
                if norm > 0:
                    gain = self._settings.step_size / (norm + self._settings.epsilon)
                    self._weights *= (1 - self._settings.leakage)
                    self._weights += (
                        gain * e[:, None] * self._hist[None, :]
                    )
                    self._adapted += 1
                else:
                    self._frozen += 1
            self._ref_energy += float(source[0, frame]) ** 2
            self._mic_energy += float(np.sum(mic[:, frame].astype(np.float64) ** 2))
            self._err_energy += float(np.sum(e * e))
        max_l2 = float(np.max(np.sqrt(np.sum(self._weights ** 2, axis=1))))
        if max_l2 > self._settings.maximum_weight_l2:
            raise AudioLimitError("NLMS estimated echo path exceeds coefficient norm")
        self._frames += frames
        return residual, predicted

    def finish(self) -> tuple[np.ndarray, EchoSummary]:
        if self._finished:
            raise AudioContractError("NLMS stream already finished")
        if not self._frames:
            raise AudioContractError("Cannot finish an empty NLMS stream")
        channels = self._channels
        frames = self._frames
        ref_rms = math.sqrt(self._ref_energy / frames)
        mic_rms = math.sqrt(self._mic_energy / (frames * channels))
        err_rms = math.sqrt(self._err_energy / (frames * channels))
        reduction = 20 * math.log10(
            max(mic_rms, 1e-12) / max(err_rms, 1e-12)
        )
        weight_l2 = float(np.max(
            np.sqrt(np.sum(self._weights * self._weights, axis=1))
        ))
        self._finished = True
        summary = EchoSummary(
            self._rate, frames, channels, self._settings.taps,
            self._adapted, self._frozen, ref_rms, mic_rms,
            err_rms, reduction, weight_l2,
        )
        return self._weights.copy(), summary


def suppress_reference_echo(
    reference: AudioBuffer,
    microphone: AudioBuffer,
    settings: NlmsSettings = NlmsSettings(),
    *,
    freeze_mask: np.ndarray | None = None,
    chunk_frames: int = 8192,
) -> EchoResult:
    """Return predicted echo, residual and finite coefficient report."""
    if (not isinstance(reference, AudioBuffer)
            or not isinstance(microphone, AudioBuffer)
            or not isinstance(settings, NlmsSettings)):
        raise AudioContractError("Expected PCM reference, microphones and NlmsSettings")
    if reference.channels != 1 or (
        reference.sample_rate != microphone.sample_rate
        or reference.frames != microphone.frames
    ):
        raise AudioContractError("Echo reference and microphone clocks must align")
    _integer(chunk_frames, "chunk_frames", 1, 65536)
    NlmsGeometry(microphone.channels, reference.frames,
                 settings.taps, reference.sample_rate)
    if freeze_mask is not None and (
        not isinstance(freeze_mask, np.ndarray)
        or freeze_mask.dtype != np.bool_
        or freeze_mask.shape != (reference.frames,)
    ):
        raise AudioContractError("Echo freeze mask must be boolean per sample")
    stream = NlmsEchoStream(reference.sample_rate, microphone.channels, settings)
    residue = np.empty(microphone.samples.shape, dtype=np.float32)
    prediction = np.empty_like(residue)
    for start in range(0, reference.frames, chunk_frames):
        stop = min(reference.frames, start + chunk_frames)
        frozen = freeze_mask[start:stop] if freeze_mask is not None else None
        out, echoed = stream.push(
            reference.samples[:, start:stop],
            microphone.samples[:, start:stop],
            freeze=frozen,
        )
        if (float(np.max(np.abs(out))) > float(np.finfo(np.float32).max)
                or float(np.max(np.abs(echoed))) > float(np.finfo(np.float32).max)):
            raise AudioLimitError("Adaptive echo output exceeds finite float32 range")
        residue[:, start:stop] = out
        prediction[:, start:stop] = echoed
    weights, report = stream.finish()
    return EchoResult(
        microphone.copy_with(residue, source_note="explicit reference NLMS residual"),
        microphone.copy_with(prediction, source_note="estimated reference echo"),
        weights, report,
    )
