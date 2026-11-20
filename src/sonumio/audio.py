"""Audio transport, metrics, and PCM-WAV encoding."""
from dataclasses import dataclass
import io
import wave
import numpy as np

SAMPLE_RATE = 24_000

@dataclass(frozen=True)
class AudioResult:
    samples: np.ndarray
    sample_rate: int
    engine_id: str
    seed: int

    @property
    def duration_seconds(self) -> float:
        return self.samples.shape[-1] / self.sample_rate

def to_mono(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 2:
        data = data.mean(axis=0)
    if data.ndim != 1 or not data.size or not np.isfinite(data).all():
        raise ValueError("Expected a nonempty, finite, mono PCM audio array")
    return data

def encode_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    if sample_rate < 8_000 or sample_rate > 192_000:
        raise ValueError("Unsupported sample rate")
    mono = to_mono(samples)
    pcm = (np.clip(mono, -1.0, 1.0) * 32767.0).round().astype("<i2")
    destination = io.BytesIO()
    with wave.open(destination, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return destination.getvalue()


def _validate_channels(samples: np.ndarray) -> np.ndarray:
    """Validate bounded mono/stereo channel-first PCM without downmixing."""
    data = np.asarray(samples, dtype=np.float32)
    if (data.ndim != 2 or data.shape[0] not in (1, 2)
            or data.shape[1] == 0 or not np.isfinite(data).all()):
        raise ValueError("Expected nonempty finite 1- or 2-channel PCM (channels, frames)")
    return data


def encode_wav_channels(samples: np.ndarray, sample_rate: int) -> bytes:
    """Export 1- or 2-channel PCM16 WAV, preserving stereo separation.

    Channel-first input (channels, frames) becomes interleaved WAV frames.
    This function is intentionally distinct from the historic mono-downmixing
    encode_wav to preserve the published v0.0.1 mono output contract.
    """
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int)
            or not 8_000 <= sample_rate <= 192_000):
        raise ValueError("Unsupported sample rate")
    data = _validate_channels(samples)
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).round().astype("<i2")
    destination = io.BytesIO()
    with wave.open(destination, "wb") as wav:
        wav.setnchannels(data.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.T.tobytes(order="C"))
    return destination.getvalue()


def metrics_channels(samples: np.ndarray) -> dict[str, float]:
    """Measure actual channel samples, not a mono sum that masks peaks."""
    data = _validate_channels(samples).astype(np.float64)
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    return {
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 3),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
        "clipping_fraction": float(np.mean(np.abs(data) >= 1.0)),
    }

def metrics(samples: np.ndarray) -> dict[str, float]:
    mono = to_mono(samples).astype(np.float64)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    return {
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 3),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
        "clipping_fraction": float(np.mean(np.abs(mono) >= 1.0)),
    }
