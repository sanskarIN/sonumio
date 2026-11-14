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

def metrics(samples: np.ndarray) -> dict[str, float]:
    mono = to_mono(samples).astype(np.float64)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono ** 2)))
    return {
        "peak_dbfs": round(20 * np.log10(max(peak, 1e-12)), 3),
        "rms_dbfs": round(20 * np.log10(max(rms, 1e-12)), 3),
        "clipping_fraction": float(np.mean(np.abs(mono) >= 1.0)),
    }
