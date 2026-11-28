"""Safe canonical workstation-score to long-form arrangement export."""
from __future__ import annotations

from argparse import _SubParsersAction
from pathlib import Path

from .arrangement_bridge import (
    ArrangementRoute, ArrangementBridgePolicy, bridge_to_arrangement,
)
from .contracts import MidiFormatError, MidiWorkstationError, read_regular_file
from .publish import PublicationOptions, publish_bytes, _distinct
from .score_io import load_score

PROTOCOL = "sonumio-workstation-arrangement-export-v1"


def _route(text: str) -> ArrangementRoute:
    if type(text) is not str:
        raise MidiWorkstationError("Expected source:instrument[:semitones] mapping")
    fields = text.split(":")
    if len(fields) not in (2, 3) or not fields[0] or not fields[1]:
        raise MidiWorkstationError("Expected source:instrument[:semitones] mapping")
    if len(fields) == 3:
        if not fields[2] or fields[2].strip() != fields[2]:
            raise MidiWorkstationError("Invalid transposition in route")
        try:
            transpose = int(fields[2], 10)
        except ValueError as exc:
            raise MidiWorkstationError("Route transposition must be integer") from exc
    else:
        transpose = 0
    return ArrangementRoute(fields[0], fields[1], transpose)


def register_arrangement_bridge(sub: _SubParsersAction) -> None:
    parser = sub.add_parser(
        "arrangement-export",
        help="Export a strict workstation score as a legacy Sonumio arrangement",
    )
    parser.add_argument("source", type=Path, help="Canonical workstation score JSON")
    parser.add_argument("--template", type=Path, required=True,
                        help="Canonical legacy arrangement score JSON")
    parser.add_argument("--route", action="append", required=True,
                        help="Exact sourceTrackID:target[:transpose] (repeat per track)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", help="Pin workstation source bytes")
    parser.add_argument("--expected-template-sha256", help="Pin template source bytes")
    parser.add_argument("--expected-output-sha256", help="Pin generated arrangement bytes")
    parser.add_argument("--maximum-events", type=int, default=4096)
    parser.add_argument("--include-muted", action="store_true")
    parser.add_argument("--center-pan", action="store_true",
                        help="Ignore MIDI note pan; use 0 on every legacy event")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--create-parent", action="store_true")


def run_arrangement_bridge(args) -> dict:
    from sonumio.music_score_io import decode_score, encode_score, MAX_SCORE_BYTES
    from sonumio.errors import GenerationError

    source = load_score(args.source, expected_sha256=args.expected_sha256)
    # Compare both input filenames against output BEFORE doing publication.
    _distinct(args.source, args.output)
    _distinct(args.template, args.output)
    template_snapshot = read_regular_file(
        args.template, limit=MAX_SCORE_BYTES,
        expected_sha256=args.expected_template_sha256,
    )
    try:
        template = decode_score(template_snapshot.content)
    except GenerationError as exc:
        raise MidiFormatError("Invalid canonical arrangement template: " + str(exc)) from exc
    policy = ArrangementBridgePolicy(
        routes=tuple(sorted((_route(text) for text in args.route),
                            key=lambda route: route.source_track_id)),
        include_muted=args.include_muted,
        preserve_note_pan=not args.center_pan,
        maximum_events=args.maximum_events,
    )
    arrangement, ledger = bridge_to_arrangement(source.score, template, policy)
    generated = encode_score(arrangement)
    # Re-check both source/output relationships immediately before publication.
    _distinct(args.source, args.output)
    _distinct(args.template, args.output)
    receipt = publish_bytes(
        args.output, generated,
        PublicationOptions(
            overwrite=args.overwrite, create_parent=args.create_parent,
            expected_sha256=args.expected_output_sha256,
        ),
        input_path=args.source, expected_extension=".json",
    )
    return {
        "protocol": PROTOCOL,
        "source": source.as_dict(),
        "template_source_sha256": template_snapshot.sha256,
        "template_source_bytes": template_snapshot.size,
        "bridge": ledger,
        "publication": receipt.as_dict(),
        "model_inference": False,
    }


COMMANDS = {"arrangement-export": run_arrangement_bridge}
