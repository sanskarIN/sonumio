"""Small local-first CLI for generating audio or serving the REST API."""
import argparse
import json
from pathlib import Path
import sys
from . import __version__
from .errors import GenerationError
from .registry import list_models
from .service import generate_audio
from .scenes import SceneLayer, compose_scene
from .audio_core.wave_pipeline import (
    WavProcessingConfig, process_wav_file,
)
from .atomic_cli_wav import publish_generated_wav
from .audio_core.quality import QualityLimits
from .audio_core.stream_quality import inspect_wav_quality
from .audio_core.resample import ResampleSettings
from .audio_core.wave_resample import resample_wav_file
from .audio_core.transform_cli import add_transform_parser, run_transform_command

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sonumio", description="Imagine it. Hear it.")
    parser.add_argument("--version", action="version", version=f"sonumio {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate", help="Generate a WAV file from a supported prompt")
    create.add_argument("prompt")
    create.add_argument("--seconds", type=float, default=5.0)
    create.add_argument("--model", default="procedural-sfx-v0")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--output", type=Path, default=Path("sonumio.wav"))
    create.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing ordinary WAV file")
    create.add_argument("--expected-output-sha256",
                        help="Require the exact SHA256 before creating the output")
    scene = sub.add_parser("scene", help="Compose a timed, multi-layer procedural WAV scene from JSON")
    scene.add_argument("json_file", type=Path)
    scene.add_argument("--output", type=Path, default=Path("scene.wav"))
    scene.add_argument("--overwrite", action="store_true",
                       help="Explicitly replace an existing ordinary WAV file")
    scene.add_argument("--expected-output-sha256",
                       help="Require the exact SHA256 before creating the output")
    post = sub.add_parser(
        "wav-process",
        help="Locally postprocess a strict PCM WAV with two-pass SHA verification",
    )
    post.add_argument("input", type=Path, help="Existing canonical PCM WAV")
    post.add_argument("--output", type=Path, required=True)
    post.add_argument("--gain-db", type=float, default=0.0)
    post.add_argument("--peak-ceiling-dbfs", type=float)
    post.add_argument("--allow-peak-boost", action="store_true")
    post.add_argument("--remove-dc", action="store_true",
                      help="Subtract measured per-channel PCM mean before gain")
    post.add_argument("--trim-silence-dbfs", type=float,
                      help="Trim quiet file edges above this sample threshold "
                           "(-120 through -1 dBFS; original PCM before gain)")
    post.add_argument("--trim-pre-roll-frames", type=int, default=0,
                      help="Keep up to N original frames before the first active frame; "
                           "requires --trim-silence-dbfs")
    post.add_argument("--trim-post-roll-frames", type=int, default=0,
                      help="Keep up to N original frames after the last active frame; "
                           "requires --trim-silence-dbfs")
    post.add_argument("--fade-curve", choices=("linear", "cosine"),
                      default="linear",
                      help="Amplitude shape for both exact-frame edge fades")
    post.add_argument("--fade-in-frames", type=int, default=0,
                      help="Fade from zero over N retained output frames (opt-in)")
    post.add_argument("--fade-out-frames", type=int, default=0,
                      help="Fade to zero over N retained output frames (opt-in)")
    post.add_argument("--clip-policy", choices=("reject", "hard"),
                      default="reject")
    post.add_argument("--packet-frames", type=int, default=8192)
    post.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32))
    post.add_argument("--dither-seed", type=int,
                      help="Opt in to seeded, chunk-invariant streaming TPDF")
    post.add_argument("--expected-sha256",
                      help="Trusted lowercase SHA256 of the full input WAV")
    post.add_argument("--expected-output-sha256",
                      help="Require full output WAV digest before publication")
    post.add_argument("--overwrite", action="store_true")
    post.add_argument("--create-parent", action="store_true")
    convert = sub.add_parser(
        "wav-resample", help="Convert canonical PCM WAV to another sample clock with SHA verification",
    )
    convert.add_argument("input", type=Path)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--target-rate", type=int, required=True)
    convert.add_argument("--lobes", type=int, default=24)
    convert.add_argument("--window", choices=("kaiser", "hann", "blackman"), default="kaiser")
    convert.add_argument("--beta", type=float, default=8.6)
    convert.add_argument("--packet-frames", type=int, default=4096)
    convert.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32))
    convert.add_argument("--dither-seed", type=int)
    convert.add_argument("--clip-policy", choices=("reject", "hard"), default="reject")
    convert.add_argument("--expected-sha256", help="Trusted full input WAV SHA256")
    convert.add_argument("--expected-output-sha256", help="Require exact output digest before publication")
    convert.add_argument("--overwrite", action="store_true")
    convert.add_argument("--create-parent", action="store_true")
    inspect = sub.add_parser(
        "wav-inspect",
        help="Verify a canonical local PCM WAV and inspect signal health with bounded memory",
    )
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--packet-frames", type=int, default=8192)
    inspect.add_argument("--expected-sha256",
                         help="Require an exact lowercase digest of the whole input WAV")
    inspect.add_argument("--timeline", action="store_true",
                         help="Include exact one-second signal diagnostics (max 3600 windows)")
    inspect.add_argument("--continuity", action="store_true",
                         help="Report decoded frame-to-frame sample steps, including packet joins")
    inspect.add_argument("--max-adjacent-step", type=float,
                         help="Optional [0,2] maximum PCM frame step (not proof of clicks)")
    inspect.add_argument("--quiet-intervals", action="store_true",
                         help="Report quiet intervals on the original PCM clock")
    inspect.add_argument("--max-internal-quiet-frames", type=int,
                         help="Optional longest internal quiet interval budget in frames")
    inspect.add_argument("--channel-relationships", action="store_true",
                         help="Observe pairwise centered correlation and equal-mix RMS retention")
    inspect.add_argument("--fail-on-warnings", action="store_true",
                         help="Return 1 after verified JSON if any quality limit fails")
    inspect.add_argument("--max-peak", type=float, default=1.0)
    inspect.add_argument("--max-clipping-fraction", type=float, default=0.0)
    inspect.add_argument("--max-pcm-rail-fraction", type=float,
                         help="Optional exact integer PCM endpoint occupancy budget [0,1]")
    inspect.add_argument("--max-dc-absolute", type=float, default=0.05)
    inspect.add_argument("--max-boundary-jump", type=float, default=0.25)
    inspect.add_argument("--max-channel-imbalance-db", type=float, default=24.0)
    inspect.add_argument("--min-active-fraction", type=float, default=0.001)
    inspect.add_argument("--activity-floor-dbfs", type=float, default=-70.0)
    add_transform_parser(sub)
    sub.add_parser("models", help="Inspect provenance and weights licenses")
    serve = sub.add_parser("serve", help="Start a local REST API")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "wav-transform":
        return run_transform_command(args)
    if args.command == "models":
        for model in list_models():
            print(f"{model['id']} | {model['model_kind']} | {model['license']}")
        return 0
    if args.command == "serve":
        import uvicorn
        uvicorn.run("sonumio.api:app", host="127.0.0.1", port=args.port)
        return 0
    if args.command == "wav-resample":
        try:
            settings = ResampleSettings(
                args.target_rate, lobes=args.lobes, window=args.window, beta=args.beta,
            )
            report = resample_wav_file(
                args.input, args.output, settings=settings, packet_frames=args.packet_frames,
                output_bits=args.output_bits, dither_seed=args.dither_seed,
                clip_policy=args.clip_policy, expected_source_sha256=args.expected_sha256,
                expected_output_sha256=args.expected_output_sha256,
                overwrite=args.overwrite, create_parent=args.create_parent,
            )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            print(f"Sonumio WAV resampling: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report.as_dict(), sort_keys=True, allow_nan=False))
        return 0
    if args.command == "wav-inspect":
        try:
            limits = QualityLimits(
                max_peak=args.max_peak,
                max_clipping_fraction=args.max_clipping_fraction,
                max_dc_absolute=args.max_dc_absolute,
                max_boundary_jump=args.max_boundary_jump,
                max_channel_imbalance_db=args.max_channel_imbalance_db,
                min_active_fraction=args.min_active_fraction,
                activity_floor_dbfs=args.activity_floor_dbfs,
            )
            report = inspect_wav_quality(
                args.input, packet_frames=args.packet_frames,
                expected_sha256=args.expected_sha256, limits=limits,
                include_timeline=args.timeline,
                max_pcm_rail_fraction=args.max_pcm_rail_fraction,
                include_continuity=args.continuity,
                max_adjacent_step=args.max_adjacent_step,
                include_quiet_intervals=args.quiet_intervals,
                max_internal_quiet_frames=args.max_internal_quiet_frames,
                include_channel_relationships=args.channel_relationships,
            )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            print(f"Sonumio WAV inspection: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report.as_dict(), sort_keys=True))
        return 1 if args.fail_on_warnings and not report.inspection_policy_passed else 0
    if args.command == "wav-process":
        try:
            config = WavProcessingConfig(
                gain_db=args.gain_db,
                peak_ceiling_dbfs=args.peak_ceiling_dbfs,
                allow_peak_boost=args.allow_peak_boost,
                remove_dc=args.remove_dc,
                trim_silence_dbfs=args.trim_silence_dbfs,
                trim_pre_roll_frames=args.trim_pre_roll_frames,
                trim_post_roll_frames=args.trim_post_roll_frames,
                fade_curve=args.fade_curve,
                fade_in_frames=args.fade_in_frames,
                fade_out_frames=args.fade_out_frames,
                clip_policy=args.clip_policy,
                packet_frames=args.packet_frames,
                output_bits=args.output_bits,
                dither_seed=args.dither_seed,
            )
            report = process_wav_file(
                args.input, args.output, config=config,
                expected_source_sha256=args.expected_sha256,
                expected_output_sha256=args.expected_output_sha256,
                overwrite=args.overwrite, create_parent=args.create_parent,
            )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            print(f"Sonumio WAV processing: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({
            "source": str(report.source),
            "destination": str(report.destination),
            "frames": report.frames,
            "sample_rate": report.sample_rate,
            "channels": report.channels,
            "source_bits": report.source_bits,
            "output_bits": report.output_bits,
            "source_sha256": report.source_sha256,
            "output_sha256": report.output_sha256,
            "effective_gain_db": report.effective_gain_db,
            "source_peak": report.source_peak,
            "source_rms": report.source_rms,
            "output_peak_before_quantization": report.output_peak_before_quantization,
            "output_rms_before_quantization": report.output_rms_before_quantization,
            "clipped_samples": report.clipped_samples,
            "output_bytes": report.output_bytes,
            "original_model_inference": report.original_model_inference,
            "dither_applied": report.dither_applied,
            "dither_seed": report.dither_seed,
            "dither_protocol": report.dither_protocol,
            "dc_removed": report.dc_removed,
            "removed_dc_offsets": report.removed_dc_offsets,
            "source_frames": report.source_frames,
            "retained_start_frame": report.retained_start_frame,
            "retained_end_frame": report.retained_end_frame,
            "trim_silence_dbfs": report.trim_silence_dbfs,
            "fade_in_frames": report.fade_in_frames,
            "fade_out_frames": report.fade_out_frames,
            "trim_pre_roll_frames": report.trim_pre_roll_frames,
            "trim_post_roll_frames": report.trim_post_roll_frames,
            "fade_curve": report.fade_curve,
        }, sort_keys=True))
        return 0
    try:
        if args.command == "scene":
            payload = json.loads(args.json_file.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("layers"), list):
                raise GenerationError("scene JSON must contain a layers array")
            layers = [SceneLayer(**item) for item in payload["layers"]]
            generated = compose_scene(layers, payload["duration_seconds"])
        else:
            generated = generate_audio(args.prompt, args.seconds, args.model, args.seed)
        published = publish_generated_wav(
            args.output, generated.wav_bytes,
            overwrite=args.overwrite,
            expected_sha256=args.expected_output_sha256,
        )
    except (GenerationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Sonumio: {exc}", file=sys.stderr)
        return 2
    print(
        f"Wrote {published.path} "
        f"({generated.result.duration_seconds:.2f}s, "
        f"{generated.result.sample_rate} Hz, "
        f"SHA256 {published.wav_sha256})"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
