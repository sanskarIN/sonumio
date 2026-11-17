"""Trace original ambient/drone DSP expression across NumPy versions.

This does not change `_drone`, the generator or any neural data. It
recomputes exactly the original array expressions and verifies bytewise
equality against the unchanged implementation in the SAME process.
No waveform arrays, WAV files or model weights are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from research.data_reproducibility import _environment, _hash_ok
from research.dsp_pipeline_trace import digest_array
from sonumio.engines import procedural as dsp

PROTOCOL = "sonumio-original-ambient-expression-stage-v1"
FRAMES = round(dsp.SAMPLE_RATE * 1.5)


def _record(samples: np.ndarray) -> dict:
    if (not isinstance(samples, np.ndarray) or samples.ndim != 1
            or len(samples) != FRAMES or not np.isfinite(samples).all()):
        raise ValueError("Expected finite one-dimensional 36k-frame diagnostic")
    result = digest_array(samples)
    return {
        "dtype": str(samples.dtype),
        "sha256": result["sha256"],
    }


def build_probe() -> dict:
    t = np.arange(FRAMES, dtype=np.float32) / dsp.SAMPLE_RATE
    stages: dict[str, dict] = {}
    stages["time_grid"] = _record(t)

    phase_110 = 2 * np.pi * 110 * t
    phase_164 = 2 * np.pi * 164.81 * t
    phase_220 = 2 * np.pi * 220.0 * t
    phase_lfo = 2 * np.pi * 0.4 * t
    stages["phase_110"] = _record(phase_110)
    stages["phase_164_81"] = _record(phase_164)
    stages["phase_220"] = _record(phase_220)
    stages["phase_lfo"] = _record(phase_lfo)

    sin_110 = np.sin(phase_110)
    sin_164 = np.sin(phase_164)
    sin_220 = np.sin(phase_220)
    sin_lfo = np.sin(phase_lfo)
    stages["sin_110"] = _record(sin_110)
    stages["sin_164_81"] = _record(sin_164)
    stages["sin_220"] = _record(sin_220)
    stages["sin_lfo"] = _record(sin_lfo)

    scaled_164 = 0.55 * sin_164
    scaled_220 = 0.25 * sin_220
    wave = sin_110 + scaled_164 + scaled_220
    lfo = 0.38 + 0.08 * sin_lfo
    output = (wave * lfo).astype(np.float32)

    stages["scaled_sin_164_81"] = _record(scaled_164)
    stages["scaled_sin_220"] = _record(scaled_220)
    stages["wave"] = _record(wave)
    stages["envelope"] = _record(lfo)
    stages["original_drone_output"] = _record(output)

    canonical = dsp._drone(t)
    if not np.array_equal(output, canonical):
        raise ValueError("Diagnostic does not reproduce unchanged procedural drone")
    result = {
        "protocol": PROTOCOL,
        "sample_rate": dsp.SAMPLE_RATE,
        "frames": FRAMES,
        "stages": stages,
        "environment": _environment(),
        "matches_original_drone_exactly": True,
        "source_generator_modified": False,
        "audio_files_saved": False,
        "model_weights_saved": False,
    }
    validate_probe(result)
    return result


def validate_probe(value: dict) -> None:
    if (not isinstance(value, dict)
            or value.get("protocol") != PROTOCOL
            or value.get("sample_rate") != dsp.SAMPLE_RATE
            or value.get("frames") != FRAMES
            or value.get("matches_original_drone_exactly") is not True
            or value.get("source_generator_modified") is not False
            or value.get("audio_files_saved") is not False
            or value.get("model_weights_saved") is not False):
        raise ValueError("Invalid ambient expression experiment metadata")
    stages = value.get("stages")
    if not isinstance(stages, dict) or list(stages) != [
        "time_grid", "phase_110", "phase_164_81",
        "phase_220", "phase_lfo", "sin_110", "sin_164_81",
        "sin_220", "sin_lfo", "scaled_sin_164_81",
        "scaled_sin_220", "wave", "envelope",
        "original_drone_output",
    ]:
        raise ValueError("Missing ambient expression stages or wrong order")
    for name, record in stages.items():
        if (not isinstance(record, dict)
                or record.get("dtype") not in ("float32", "float64")
                or not _hash_ok(record.get("sha256"))):
            raise ValueError("Invalid ambient expression stage " + name)


def compare_probes(first: dict, second: dict) -> dict:
    validate_probe(first)
    validate_probe(second)
    mismatches = [
        name for name in first["stages"]
        if first["stages"][name] != second["stages"][name]
    ]
    return {
        "protocol": "sonumio-ambient-expression-version-comparison-v1",
        "first_divergent_stage": mismatches[0] if mismatches else None,
        "different_stages": mismatches,
        "same_final_drone_bytes": (
            first["stages"]["original_drone_output"]["sha256"]
            == second["stages"]["original_drone_output"]["sha256"]
        ),
        "same_cpu_model": (
            first["environment"]["cpu_model"]
            == second["environment"]["cpu_model"]
        ),
        "numpy_versions": (
            first["environment"]["numpy"],
            second["environment"]["numpy"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    if args.compare:
        values = [json.loads(p.read_text(encoding="utf-8")) for p in args.compare]
        report = compare_probes(values[0], values[1])
    else:
        report = build_probe()
    serialized = json.dumps(report, sort_keys=True, indent=2,
                            ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
