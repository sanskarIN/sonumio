"""Small local-first CLI for generating audio or serving the REST API."""
import argparse
import json
from pathlib import Path
import sys
from .errors import GenerationError
from .registry import list_models
from .service import generate_audio
from .scenes import SceneLayer, compose_scene
from .audio_core.wave_pipeline import (
    WavProcessingConfig, process_wav_file,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sonumio", description="Imagine it. Hear it.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate", help="Generate a WAV file from a supported prompt")
    create.add_argument("prompt")
    create.add_argument("--seconds", type=float, default=5.0)
    create.add_argument("--model", default="procedural-sfx-v0")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--output", type=Path, default=Path("sonumio.wav"))
    scene = sub.add_parser("scene", help="Compose a timed, multi-layer procedural WAV scene from JSON")
    scene.add_argument("json_file", type=Path)
    scene.add_argument("--output", type=Path, default=Path("scene.wav"))
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
    sub.add_parser("models", help="Inspect provenance and weights licenses")
    serve = sub.add_parser("serve", help="Start a local REST API")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "models":
        for model in list_models():
            print(f"{model['id']} | {model['model_kind']} | {model['license']}")
        return 0
    if args.command == "serve":
        import uvicorn
        uvicorn.run("sonumio.api:app", host="127.0.0.1", port=args.port)
        return 0
    if args.command == "wav-process":
        try:
            config = WavProcessingConfig(
                gain_db=args.gain_db,
                peak_ceiling_dbfs=args.peak_ceiling_dbfs,
                allow_peak_boost=args.allow_peak_boost,
                remove_dc=args.remove_dc,
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(generated.wav_bytes)
    except (GenerationError, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Sonumio: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output} ({generated.result.duration_seconds:.2f}s, {generated.result.sample_rate} Hz)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
