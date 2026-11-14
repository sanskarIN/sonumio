"""Small local-first CLI for generating audio or serving the REST API."""
import argparse
from pathlib import Path
import sys
from .errors import GenerationError
from .registry import list_models
from .service import generate_audio

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sonumio", description="Imagine it. Hear it.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("generate", help="Generate a WAV file from a supported prompt")
    create.add_argument("prompt")
    create.add_argument("--seconds", type=float, default=5.0)
    create.add_argument("--model", default="procedural-sfx-v0")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--output", type=Path, default=Path("sonumio.wav"))
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
    try:
        generated = generate_audio(args.prompt, args.seconds, args.model, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(generated.wav_bytes)
    except (GenerationError, OSError) as exc:
        print(f"Sonumio: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output} ({generated.result.duration_seconds:.2f}s, {generated.result.sample_rate} Hz)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
