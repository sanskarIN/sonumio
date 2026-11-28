"""Atomic, receipt-backed legacy arrangement to new score import."""
from __future__ import annotations

from argparse import _SubParsersAction
from pathlib import Path

from .arrangement_import import (
    LegacyInstrumentRoute, LegacyImportPolicy, arrangement_to_score,
)
from .contracts import MidiWorkstationError, MidiFormatError, read_regular_file
from .publish import PublicationOptions, publish_score

PROTOCOL = "sonumio-legacy-arrangement-import-v1"


def _route(text: str) -> LegacyInstrumentRoute:
    """Explicit instrument:track-id:channel:program[:bank-msb[:bank-lsb]]."""
    if type(text) is not str:
        raise MidiWorkstationError("Legacy route must be a string")
    fields = text.split(":")
    if len(fields) not in (4, 5, 6) or not all(fields):
        raise MidiWorkstationError("Expected instrument:track-id:channel:program[:msb[:lsb]]")
    try:
        numbers = [int(value, 10) for value in fields[2:]]
    except ValueError as exc:
        raise MidiWorkstationError("Legacy route channel/patch values must be integers") from exc
    numbers += [0] * (4 - len(numbers))
    return LegacyInstrumentRoute(fields[0], fields[1], *numbers)


def register_arrangement_import(sub: _SubParsersAction) -> None:
    parser = sub.add_parser(
        "arrangement-import", help="Import an existing legacy arrangement into editable score JSON",
    )
    parser.add_argument("source", type=Path, help="Canonical legacy arrangement score JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route", action="append", required=True,
                        help="instrument:track-id:channel:program[:msb[:lsb]]")
    parser.add_argument("--ppqn", type=int, default=480)
    parser.add_argument("--allow-velocity-quantization", action="store_true")
    parser.add_argument("--allow-pan-quantization", action="store_true")
    parser.add_argument("--allow-swing-loss", action="store_true")
    parser.add_argument("--maximum-events", type=int, default=4096)
    parser.add_argument("--expected-sha256", help="Pin canonical legacy source bytes")
    parser.add_argument("--expected-output-sha256", help="Pin new score JSON bytes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--create-parent", action="store_true")


def run_arrangement_import(args) -> dict:
    from sonumio.music_score_io import decode_score, MAX_SCORE_BYTES
    from sonumio.errors import GenerationError

    snapshot = read_regular_file(
        args.source, limit=MAX_SCORE_BYTES, expected_sha256=args.expected_sha256,
    )
    try:
        legacy = decode_score(snapshot.content)
    except GenerationError as exc:
        raise MidiFormatError("Invalid canonical legacy arrangement: " + str(exc)) from exc
    routes = tuple(sorted((_route(x) for x in args.route),key=lambda r:r.instrument))
    policy = LegacyImportPolicy(
        routes=routes, ppqn=args.ppqn,
        allow_velocity_quantization=args.allow_velocity_quantization,
        allow_pan_quantization=args.allow_pan_quantization,
        allow_swing_loss=args.allow_swing_loss,
        maximum_events=args.maximum_events,
    )
    score, report = arrangement_to_score(legacy, policy)
    receipt = publish_score(
        args.output, score,
        PublicationOptions(
            overwrite=args.overwrite, create_parent=args.create_parent,
            expected_sha256=args.expected_output_sha256,
        ),
        input_path=args.source,
    )
    return {
        "protocol": PROTOCOL,
        "legacy_source_sha256": snapshot.sha256,
        "legacy_source_bytes": snapshot.size,
        "conversion": report,
        "publication": receipt.as_dict(),
    }


COMMANDS = {"arrangement-import": run_arrangement_import}
