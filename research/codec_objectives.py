"""Original differentiable, bounded multiresolution neural audio-codec losses.

These are explicit *signal-domain* reconstruction criteria, NOT perceptual
quality, FAD, CLAP, ASR, or human listening scores. Each component is a
nonnegative differentiable scalar computed from exact, sample-aligned
first-party 8192-frame 8-kHz mono waveforms.

The training objective intentionally distinguishes transient timing,
high-frequency magnitude and spectral convergence from sample L1;
it does not guarantee that any weighted combination sounds better.
No pretrained dependencies, files, audio exports or model releases.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
from torch.nn import functional as F

from research.latent_audio import FRAMES
from research.wave_diffusion import SAMPLE_RATE

PROTOCOL = "sonumio-codec-multiresolution-objective-v1"
FFT_SIZES = (256, 512, 1024, 2048)
ENVELOPE_WIDTHS = (64, 256, 1024)
COMPONENT_NAMES = (
    "waveform_l1", "preemphasis_l1", "envelope_rms_l1",
    "log_magnitude_l1", "linear_magnitude_l1",
    "spectral_convergence", "spectral_flux_l1",
    "highband_energy_ratio_l1",
)
WEIGHT_NAMES = (
    "waveform", "preemphasis", "envelope", "log_magnitude",
    "linear_magnitude", "spectral_convergence", "spectral_flux",
    "highband_energy_ratio",
)
MAX_BATCH = 8
EPS = 1e-7
SAMPLE_LIMIT = 1.0001


@dataclass(frozen=True, slots=True)
class CodecObjectiveWeights:
    """Explicit finite nonnegative weights; no hidden adaptive tradeoffs."""

    waveform: float = 1.0
    preemphasis: float = 0.2
    envelope: float = 0.25
    log_magnitude: float = 0.75
    linear_magnitude: float = 0.1
    spectral_convergence: float = 0.15
    spectral_flux: float = 0.15
    highband_energy_ratio: float = 0.1

    def __post_init__(self) -> None:
        for name in WEIGHT_NAMES:
            value = getattr(self, name)
            if (type(value) not in (float, int)
                    or not math.isfinite(value)
                    or not 0 <= value <= 10):
                raise ValueError("Codec objective weights must be finite in [0,10]")
        if sum(getattr(self, name) for name in WEIGHT_NAMES) <= 0:
            raise ValueError("Codec objective needs a nonzero supervised component")

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in WEIGHT_NAMES}


@dataclass(frozen=True, slots=True)
class CodecSpectralConfig:
    """Bounded FFT/hop and high-band configuration for 8-kHz waveforms."""

    fft_sizes: tuple[int, ...] = FFT_SIZES
    highband_hz: int = 1500
    high_frequency_boost: float = 0.5
    preemphasis_coefficient: float = 0.95

    def __post_init__(self) -> None:
        if (type(self.fft_sizes) is not tuple
                or not 1 <= len(self.fft_sizes) <= 4
                or len(set(self.fft_sizes)) != len(self.fft_sizes)
                or tuple(sorted(self.fft_sizes)) != self.fft_sizes
                or any(type(f) is not int or f not in FFT_SIZES
                       for f in self.fft_sizes)
                or type(self.highband_hz) is not int
                or not 500 <= self.highband_hz < SAMPLE_RATE // 2
                or type(self.high_frequency_boost) not in (float, int)
                or not math.isfinite(self.high_frequency_boost)
                or not 0 <= self.high_frequency_boost <= 2
                or type(self.preemphasis_coefficient) not in (float, int)
                or not math.isfinite(self.preemphasis_coefficient)
                or not 0 <= self.preemphasis_coefficient <= 1):
            raise ValueError("Invalid bounded multiresolution spectral settings")

    def as_dict(self) -> dict:
        return {
            "fft_sizes": list(self.fft_sizes),
            "highband_hz": self.highband_hz,
            "high_frequency_boost": float(self.high_frequency_boost),
            "preemphasis_coefficient": float(
                self.preemphasis_coefficient
            ),
        }


def _validated_wave(
    reference: torch.Tensor, restored: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (not isinstance(reference, torch.Tensor)
            or not isinstance(restored, torch.Tensor)
            or reference.shape != restored.shape
            or reference.ndim != 3
            or reference.shape[1:] != (1, FRAMES)
            or not 1 <= len(reference) <= MAX_BATCH
            or reference.dtype != torch.float32
            or restored.dtype != torch.float32
            or reference.device.type != "cpu"
            or restored.device.type != "cpu"
            or not torch.isfinite(reference).all()
            or not torch.isfinite(restored).all()
            or reference.abs().max().item() > SAMPLE_LIMIT
            or restored.abs().max().item() > SAMPLE_LIMIT):
        raise ValueError(
            "Reconstruction objective needs aligned finite CPU float32 "
            "mono clips within [-1,1]"
        )
    return reference[:, 0], restored[:, 0]


def _spectrogram(
    signal: torch.Tensor, n_fft: int,
) -> torch.Tensor:
    """Hann STFT magnitude with stable gradients and no stored transform."""
    if type(n_fft) is not int or n_fft not in FFT_SIZES:
        raise ValueError("Unsupported experimental codec spectral window")
    window = torch.hann_window(n_fft, dtype=torch.float32)
    transform = torch.stft(
        signal, n_fft, hop_length=n_fft // 4,
        window=window, center=True,
        normalized=False, return_complex=True,
    )
    magnitude = transform.abs()
    if not torch.isfinite(magnitude).all():
        raise ValueError("Nonfinite multiresolution codec spectrum")
    return magnitude


def _log_spectral_flux(magnitude: torch.Tensor) -> torch.Tensor:
    compressed = torch.log1p(magnitude)
    return compressed[..., 1:] - compressed[..., :-1]


def _rms_envelope(signal: torch.Tensor, width: int) -> torch.Tensor:
    if type(width) is not int or width not in ENVELOPE_WIDTHS:
        raise ValueError("Unsupported deterministic RMS envelope width")
    squared = signal.double().reshape(
        signal.shape[0], FRAMES // width, width,
    ).square().mean(dim=-1)
    return (squared + 1e-14).sqrt().float()


def _highband_ratio(
    magnitude: torch.Tensor, fft: int, cutoff: int,
) -> torch.Tensor:
    frequencies = torch.fft.rfftfreq(
        fft, d=1 / SAMPLE_RATE
    )
    high = frequencies >= cutoff
    if not high.any() or high.all():
        raise ValueError("Invalid cutoff for 8-kHz experimental spectrum")
    power = magnitude.double().square()
    high_power = power[:, high, :].sum(dim=(1, 2))
    all_power = power.sum(dim=(1, 2))
    return (high_power / all_power.clamp_min(EPS)).float()


def _scalar_finite(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 0
            or not tensor.is_floating_point() or not torch.isfinite(tensor)
            or tensor.item() < 0):
        raise ValueError("Nonfinite/negative codec signal metric " + name)
    return tensor


def codec_components(
    reference: torch.Tensor, restored: torch.Tensor, *,
    config: CodecSpectralConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Return separate differentiable scalar metrics with explicit units.

    Magnitude objectives compare sample-aligned STFT. Spectral
    convergence normalizes an error norm by the source spectrum norm
    (with a numerical guard near silence), so it must not be treated
    as a perceptual or frequency-invariant accuracy score.
    """
    opts = config if config is not None else CodecSpectralConfig()
    if not isinstance(opts, CodecSpectralConfig):
        raise ValueError("Expected an explicit codec spectral configuration")
    original, reconstructed = _validated_wave(reference, restored)
    results: dict[str, torch.Tensor] = {}

    results["waveform_l1"] = F.l1_loss(reconstructed, original)

    k = float(opts.preemphasis_coefficient)
    high_original = original[:, 1:] - k * original[:, :-1]
    high_reconstructed = reconstructed[:, 1:] - k * reconstructed[:, :-1]
    results["preemphasis_l1"] = F.l1_loss(
        high_reconstructed, high_original,
    )

    envelopes = [
        F.l1_loss(
            _rms_envelope(reconstructed, width),
            _rms_envelope(original, width),
        )
        for width in ENVELOPE_WIDTHS
    ]
    results["envelope_rms_l1"] = torch.stack(envelopes).mean()

    magnitude_l1 = []
    log_l1 = []
    convergence = []
    flux_l1 = []
    band_ratio_l1 = []
    for n_fft in opts.fft_sizes:
        source_mag = _spectrogram(original, n_fft)
        model_mag = _spectrogram(reconstructed, n_fft)
        frequency_hz = torch.fft.rfftfreq(
            n_fft, d=1 / SAMPLE_RATE,
        )
        frequency_weight = (
            1 + float(opts.high_frequency_boost)
            * (frequency_hz / (SAMPLE_RATE / 2)).square()
        ).view(1, -1, 1)
        magnitude_l1.append(
            ((model_mag - source_mag).abs() * frequency_weight).mean()
        )
        compressed_original = torch.log1p(source_mag)
        compressed_model = torch.log1p(model_mag)
        log_l1.append(
            ((compressed_model - compressed_original).abs()
             * frequency_weight).mean()
        )
        numerator = torch.linalg.vector_norm(
            (model_mag - source_mag).flatten(1), dim=1,
        )
        denominator = torch.linalg.vector_norm(
            source_mag.flatten(1), dim=1,
        ).clamp_min(EPS)
        convergence.append((numerator / denominator).mean())
        flux_l1.append(
            F.l1_loss(
                _log_spectral_flux(model_mag),
                _log_spectral_flux(source_mag),
            )
        )
        band_ratio_l1.append(F.l1_loss(
            _highband_ratio(model_mag, n_fft, opts.highband_hz),
            _highband_ratio(source_mag, n_fft, opts.highband_hz),
        ))
    results["log_magnitude_l1"] = torch.stack(log_l1).mean()
    results["linear_magnitude_l1"] = torch.stack(magnitude_l1).mean()
    results["spectral_convergence"] = torch.stack(convergence).mean()
    results["spectral_flux_l1"] = torch.stack(flux_l1).mean()
    results["highband_energy_ratio_l1"] = torch.stack(
        band_ratio_l1
    ).mean()

    if set(results) != set(COMPONENT_NAMES):
        raise ValueError("Incomplete multiresolution objective decomposition")
    for name, value in results.items():
        _scalar_finite(name, value)
    return results


def weighted_codec_loss(
    reference: torch.Tensor, restored: torch.Tensor, *,
    weights: CodecObjectiveWeights | None = None,
    config: CodecSpectralConfig | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted sum with no learned or hidden adaptive weighting.

    Does NOT compare codec model architectures on perceptual quality.
    A held-out study is required to determine whether it is worthwhile.
    """
    opts = weights if weights is not None else CodecObjectiveWeights()
    if not isinstance(opts, CodecObjectiveWeights):
        raise ValueError("Expected explicit bounded codec objective weights")
    if type(return_components) is not bool:
        raise ValueError("return_components must be boolean")
    components = codec_components(reference, restored, config=config)
    contributions = [
        components[key] * float(getattr(opts, weight))
        for key, weight in zip(COMPONENT_NAMES, WEIGHT_NAMES)
    ]
    total = torch.stack(contributions).sum()
    _scalar_finite("total", total)
    return (total, components) if return_components else total


@torch.no_grad()
def describe_codec_error(
    reference: torch.Tensor, restored: torch.Tensor, *,
    config: CodecSpectralConfig | None = None,
) -> dict:
    """Numeric-only, explainable, sample-aligned codec diagnostics."""
    source, produced = _validated_wave(reference, restored)
    error = (source.double() - produced.double())
    source_energy = float(source.double().square().sum())
    residual_energy = float(error.square().sum())
    signal_to_noise_db = 10 * math.log10(
        (source_energy + 1e-10) / (residual_energy + 1e-10)
    )
    errors = codec_components(reference, restored, config=config)
    stats = {name: float(value) for name, value in errors.items()}
    if not math.isfinite(signal_to_noise_db):
        raise ValueError("Nonfinite reconstruction SNR")
    return {
        "protocol": PROTOCOL,
        "components": stats,
        "sample_rate": SAMPLE_RATE,
        "frames": FRAMES,
        "clips": len(reference),
        "sample_aligned_mae": float(error.abs().mean()),
        "sample_aligned_rmse": float(error.square().mean().sqrt()),
        "snr_db": signal_to_noise_db,
        "zero_error_uses_finite_epsilon": True,
        "listenability_or_semantics_measured": False,
        "external_perceptual_model_used": False,
        "audio_or_model_weights_written": False,
    }
