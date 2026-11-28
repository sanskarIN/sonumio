"""Standalone local-only MusicXML / MXL score conversion CLI.

This entrypoint is intentionally separate from the general workstation CLI:
it can be installed and smoke-tested independently without altering existing
command names or making MusicXML import implicit in MIDI import paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .contracts import MAX_MIDI_BYTES, MidiWorkstationError, read_regular_file
from .musicxml_export import encode_musicxml
from .musicxml_import import decode_musicxml
from .mxl import decode_mxl, encode_mxl
from .publish import PublicationOptions, publish_bytes, publish_score
from .score_io import load_score


def _source(parser):
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-sha256", help="SHA-256 guard for source bytes")


def _output(parser):
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--create-parent", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sonumio-musicxml",
        description="Explicit, bounded MusicXML performance interchange")
    sub = root.add_subparsers(dest="command", required=True)
    for operation in ("to-xml", "to-mxl", "from-xml", "from-mxl",
                      "inspect", "audit", "preview-xml", "preview-mxl"):
        node = sub.add_parser(operation)
        _source(node)
        if operation not in ("inspect", "audit"):
            _output(node)
        if operation.startswith("preview-"):
            node.add_argument("--sample-rate", type=int, default=24000)
            node.add_argument("--channels", type=int, default=2)
            node.add_argument("--waveform", choices=("sine", "saw", "square", "triangle"), default="triangle")
            node.add_argument("--gain", type=float, default=.15)
            node.add_argument("--bits", type=int, default=16)
            node.add_argument("--dither-seed", type=int)
    return root


def _options(args) -> PublicationOptions:
    return PublicationOptions(overwrite=args.overwrite,
        create_parent=args.create_parent,
        expected_sha256=args.expected_output_sha256)


def _run(args) -> dict:
    if args.command.startswith("preview-"):
        from .musicxml_preview import export_musicxml_preview
        from .score_preview import ScorePreviewOptions
        from .wavetable_bridge import WavetableBridgeSettings
        data = read_regular_file(args.source, limit=MAX_MIDI_BYTES,
                                 expected_sha256=args.expected_sha256)
        is_mxl = args.command == "preview-mxl"
        expected_suffix = ".mxl" if is_mxl else ".musicxml"
        if args.source.suffix.lower() != expected_suffix:
            raise MidiWorkstationError("Preview source extension does not match requested format")
        bridge = WavetableBridgeSettings(sample_rate=args.sample_rate,
            channels=args.channels, waveform=args.waveform, master_gain=args.gain)
        preview_options = ScorePreviewOptions(bridge=bridge, output_bits=args.bits,
            dither_seed=args.dither_seed, overwrite=args.overwrite,
            create_parent=args.create_parent,
            expected_output_sha256=args.expected_output_sha256)
        preview, details = export_musicxml_preview(
            data.content, args.output, compressed=is_mxl,
            options=preview_options, source_path=args.source)
        return {"protocol": "sonumio-musicxml-wav-preview-cli-v1",
                "preview": preview.as_dict(), **details}
    if args.command in ("to-xml", "to-mxl"):
        snapshot = load_score(args.source, expected_sha256=args.expected_sha256)
        raw, receipt = encode_musicxml(snapshot.score)
        if args.command == "to-mxl":
            raw, package = encode_mxl(raw)
            suffix = ".mxl"
        else:
            package = None
            suffix = ".musicxml"
        published = publish_bytes(args.output, raw, _options(args),
                                  input_path=args.source, expected_extension=suffix)
        return {"protocol": "sonumio-musicxml-export-cli-v1",
                "source": snapshot.as_dict(),
                "musicxml": receipt.as_dict(),
                "package": package.as_dict() if package else None,
                "publication": published.as_dict()}
    source = read_regular_file(args.source, limit=MAX_MIDI_BYTES,
                               expected_sha256=args.expected_sha256)
    if args.command == "from-mxl" or (args.command in ("inspect", "audit") and args.source.suffix.lower() == ".mxl"):
        xml, package = decode_mxl(source.content)
    else:
        xml = source.content
        package = None
    if args.command == "audit":
        from .musicxml_audit import audit_musicxml
        audit = audit_musicxml(xml)
        return {"protocol": "sonumio-musicxml-audit-cli-v1",
                "source_sha256": source.sha256,
                "audit": audit.as_dict(),
                "package": package.as_dict() if package else None,
                "source_unchanged": True}
    score, receipt = decode_musicxml(xml)
    if args.command == "inspect":
        return {"protocol": "sonumio-musicxml-inspection-cli-v1",
                "source_sha256": source.sha256,
                "conversion": receipt.as_dict(),
                "score": score.description(),
                "package": package.as_dict() if package else None,
                "source_unchanged": True}
    expected_suffix = ".mxl" if args.command == "from-mxl" else ".musicxml"
    if args.source.suffix.lower() != expected_suffix:
        raise MidiWorkstationError("Input extension does not match requested MusicXML command")
    published = publish_score(args.output, score, _options(args), input_path=args.source)
    return {"protocol": "sonumio-musicxml-import-cli-v1",
            "source_sha256": source.sha256,
            "conversion": receipt.as_dict(),
            "score_sha256": score.sha256,
            "publication": published.as_dict()}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = _run(args)
        encoded = json.dumps(result, sort_keys=True, ensure_ascii=True,
                             allow_nan=False, separators=(",", ":"))
    except (MidiWorkstationError, OSError, OverflowError, RecursionError) as exc:
        print(f"Sonumio MusicXML: {exc}", file=sys.stderr)
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
