"""Isolated long-form verified WAV scanner resource benchmark.

Generates only deterministic scratch PCM inside TemporaryDirectory. Each scan
runs in an independent Python process so peak RSS readings cannot inherit a
previous packet-size trial. These are observational *process* measurements,
not latency guarantees, neural inference, or listening-quality results.

Example:
    python -m benchmarks.wav_quality_profile --seconds 121 --timeline
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

from sonumio.audio_core.stream_quality import inspect_wav_quality
from sonumio.audio_core.streaming import FramePacket
from sonumio.audio_core.wave_stream import WavPacketSpec, WavPacketWriter

PROTOCOL = "sonumio-verified-wav-resource-profile-v1"
SAMPLE_RATE = 8_000


def _rss_highwater_kib() -> int | None:
    if sys.platform not in ("linux", "darwin"):
        return None
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError):
        return None
    return int(value / 1024) if sys.platform == "darwin" else int(value)


def _validate(seconds: int, sizes: tuple[int, ...], timeline: bool) -> None:
    if type(seconds) is not int or not 1 <= seconds <= 600:
        raise ValueError("seconds must be an integer in [1,600]")
    if (not isinstance(sizes, tuple) or not 1 <= len(sizes) <= 5
            or any(type(n) is not int or not 1 <= n <= 65536 for n in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError("Supply 1..5 distinct packet sizes in [1,65536]")
    if type(timeline) is not bool:
        raise ValueError("timeline must be boolean")


def _make_source(path: Path, seconds: int) -> str:
    count = seconds * SAMPLE_RATE
    spec = WavPacketSpec(SAMPLE_RATE, 1, count)
    with WavPacketWriter(path, spec) as writer:
        for begin in range(0, count, 8192):
            stop = min(count, begin + 8192)
            n = np.arange(begin, stop, dtype=np.float64)
            values = (
                .38 * np.sin(2.0 * np.pi * 311.0 * n / SAMPLE_RATE)
                + .12 * np.sin(2.0 * np.pi * 503.0 * n / SAMPLE_RATE)
            ).astype(np.float32)
            writer.push(FramePacket(SAMPLE_RATE, begin, values))
        return writer.finish().wav_sha256


def _child(source: Path, sha256: str, packet_frames: int,
           timeline: bool) -> dict:
    if type(packet_frames) is not int or not 1 <= packet_frames <= 65536:
        raise ValueError("Invalid child packet size")
    before_rss = _rss_highwater_kib()
    started = perf_counter()
    report = inspect_wav_quality(
        source, packet_frames=packet_frames,
        expected_sha256=sha256, include_timeline=timeline,
    )
    elapsed = perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0 or report.pcm_rails is None:
        raise ValueError("Invalid verified WAV scan result or monotonic timing")
    after_rss = _rss_highwater_kib()
    return {
        "protocol": PROTOCOL,
        "packet_frames": packet_frames,
        "seconds_scanned": report.info.frames / report.info.sample_rate,
        "frames": report.info.frames,
        "source_sha256": report.source_sha256,
        "source_verified": True,
        "peak_amplitude": report.signal_health.peak_amplitude,
        "rms_amplitude": report.signal_health.rms_amplitude,
        "pcm_rail_fraction": report.pcm_rails.rail_fraction,
        "timeline_windows": len(report.timeline.windows) if report.timeline else None,
        "scan_wall_seconds": elapsed,
        "real_time_ratio": (report.info.frames / report.info.sample_rate) / elapsed,
        "rss_before_scan_kib": before_rss,
        "process_peak_rss_kib": after_rss,
        "platform": sys.platform,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "model_weights_loaded": False,
        "perceptual_quality_measured": False,
    }


def _subprocess_scan(
    source: Path, sha256: str, packet_frames: int, timeline: bool,
) -> dict:
    args = [
        sys.executable, "-m", "benchmarks.wav_quality_profile",
        "--child", "--source", str(source),
        "--sha256", sha256, "--child-packet-frames", str(packet_frames),
    ]
    if timeline:
        args.append("--timeline")
    completed = subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=90,
    )
    result = json.loads(completed.stdout)
    if (not isinstance(result, dict)
            or result.get("protocol") != PROTOCOL
            or result.get("source_sha256") != sha256
            or result.get("packet_frames") != packet_frames
            or result.get("source_verified") is not True
            or result.get("model_weights_loaded") is not False):
        raise ValueError("Untrusted or incomplete isolated scan result")
    return result


def profile_quality_scan(
    seconds: int = 121, *, packet_frames: tuple[int, ...] = (1024, 8192, 32768),
    timeline: bool = False,
) -> dict:
    """Compare verified scans on one fixed, ephemeral source WAV."""
    _validate(seconds, packet_frames, timeline)
    with tempfile.TemporaryDirectory(prefix="sonumio-wav-profile-") as scratch:
        source = Path(scratch) / "deterministic-source.wav"
        sha256 = _make_source(source, seconds)
        size = source.stat().st_size
        variants = [
            _subprocess_scan(source, sha256, size_frames, timeline)
            for size_frames in packet_frames
        ]
    reference = variants[0]
    for item in variants:
        if (item["frames"] != seconds * SAMPLE_RATE
                or not math.isclose(
                    item["rms_amplitude"], reference["rms_amplitude"],
                    rel_tol=1e-9, abs_tol=1e-11,
                )
                or item["peak_amplitude"] != reference["peak_amplitude"]
                or item["pcm_rail_fraction"] != reference["pcm_rail_fraction"]
                or (timeline and item["timeline_windows"] != seconds)):
            raise ValueError("Packet-size variants disagree on verified source metrics")
    return {
        "protocol": PROTOCOL,
        "source_frames": seconds * SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "source_bytes": size,
        "source_sha256": sha256,
        "packet_variants": variants,
        "temporary_source_removed": True,
        "published_audio_artifacts": False,
        "model_weights_loaded": False,
        "human_listening_performed": False,
        "limitations": (
            "Peak RSS is whole-process high-water, including imports and "
            "NumPy; unavailable on Windows. Timing excludes process startup "
            "and file creation. No neural generation, cross-host speed ranking, "
            "perceptual quality or guaranteed memory ceiling is inferred."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=121)
    parser.add_argument("--packet-frames", type=int, nargs="+",
                        default=[1024, 8192, 32768])
    parser.add_argument("--timeline", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sha256", help=argparse.SUPPRESS)
    parser.add_argument("--child-packet-frames", type=int, default=8192,
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.child:
            if args.source is None or args.sha256 is None:
                raise ValueError("Child requires an existing source and checksum")
            result = _child(args.source, args.sha256, args.child_packet_frames,
                            args.timeline)
        else:
            result = profile_quality_scan(
                args.seconds, packet_frames=tuple(args.packet_frames),
                timeline=args.timeline,
            )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Sonumio verified WAV resource benchmark: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
