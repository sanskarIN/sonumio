"""Local latency microbenchmark of procedural DSP; not a neural benchmark."""
import json
import statistics
from sonumio.service import generate_audio

def benchmark(prompt: str = "rain and thunder in a forest", seconds: float = 3,
              repeats: int = 10, seed: int = 7) -> dict:
    if not 1 <= repeats <= 200:
        raise ValueError("repeats must be between 1 and 200")
    results = [generate_audio(prompt, seconds, seed=seed + i) for i in range(repeats)]
    samples = sorted(row.metrics["generation_wall_seconds"] for row in results)
    p95 = samples[min(len(samples) - 1, int(0.95 * (len(samples) - 1)))]
    return {
        "backend": "procedural-sfx-v0 (CPU DSP; not neural AI)",
        "prompt": prompt, "audio_duration_seconds": seconds,
        "repeats": repeats, "median_generation_seconds": statistics.median(samples),
        "p95_generation_seconds": p95,
        "median_real_time_factor": statistics.median(samples) / seconds,
        "note": "Local environment specific; benchmark neural models separately",
    }

if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
