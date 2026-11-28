"""Local-only Sonumio symbolic MIDI workstation command-line interface.

All operations are explicit and reproducible: inspect, validate, convert,
quantize, humanize, analyze and publish. No GitHub or network mutation occurs.
The command never modifies an input file, including on failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .analysis import score_inspection
from .contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
    canonical_json,
)
from .performance import PerformanceSettings, performance_report, reconstruct_performance
from .publish import PublicationOptions, publish_midi, publish_score
from .quantize import HumanizeSettings, QuantizeGrid, humanize_score, quantize_score
from .score import Score, from_midi, to_midi
from .score_io import load_score, parse_score_snapshot
from .smf import read_smf, validate_roundtrip
from .tempo import tempo_summary
from .score_commands import add_score_commands, COMMANDS as SCORE_COMMANDS
from .history_cli import register_history_commands, COMMANDS as HISTORY_COMMANDS


def _source_digest(parser):
    parser.add_argument("--expected-sha256", help="Pin input SHA-256")


def _destination(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-output-sha256", help="Pin generated output digest")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing file")
    parser.add_argument("--create-parent", action="store_true", help="Create missing parent directories")


def _publication(args):
    return PublicationOptions(
        overwrite=args.overwrite, create_parent=args.create_parent,
        expected_sha256=args.expected_output_sha256,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m sonumio.midi_workstation",
        description="Safe local symbolic MIDI research and score editing",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("midi-inspect", help="Inspect bounded MIDI with explicit semantic limits")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--track", type=int)
    inspect.add_argument("--notes", action="store_true", help="Include reconstructed note intervals")
    inspect.add_argument("--overlap", choices=("fifo", "lifo", "reject"), default="fifo")
    inspect.add_argument("--close-dangling", action="store_true")
    _source_digest(inspect)

    validate = sub.add_parser("midi-validate", help="Verify binary MIDI semantic roundtrip")
    validate.add_argument("source", type=Path)
    _source_digest(validate)

    to_json = sub.add_parser("midi-import", help="Convert synchronous PPQN MIDI to JSON score")
    to_json.add_argument("source", type=Path)
    to_json.add_argument("--title", default="Imported original MIDI")
    to_json.add_argument("--use-audible-duration", action="store_true")
    to_json.add_argument("--close-dangling", action="store_true")
    to_json.add_argument("--allow-patch-rechannel", action="store_true",
                         help="Opt in to reassigning program changes to free MIDI channels")
    _source_digest(to_json)
    _destination(to_json)

    to_binary = sub.add_parser("midi-export", help="Export strict JSON score as MIDI file")
    to_binary.add_argument("source", type=Path)
    to_binary.add_argument("--only-active", action="store_true")
    _source_digest(to_binary)
    _destination(to_binary)

    score_inspect = sub.add_parser("score-inspect", help="Inspect score metrics and chord hypotheses")
    score_inspect.add_argument("source", type=Path)
    score_inspect.add_argument("--window-ticks", type=int)
    score_inspect.add_argument("--include-muted", action="store_true")
    _source_digest(score_inspect)

    quantize = sub.add_parser("score-quantize", help="Quantize original note events to explicit grid")
    quantize.add_argument("source", type=Path)
    quantize.add_argument("--step-ticks", required=True, type=int)
    quantize.add_argument("--offset-ticks", type=int, default=0)
    quantize.add_argument("--swing-percent", type=int, default=0)
    quantize.add_argument("--strength-percent", type=int, default=100)
    quantize.add_argument("--quantize-releases", action="store_true")
    _source_digest(quantize)
    _destination(quantize)

    humanize = sub.add_parser("score-humanize", help="Add bounded deterministic note expression")
    humanize.add_argument("source", type=Path)
    humanize.add_argument("--timing-ticks", type=int, default=4)
    humanize.add_argument("--velocity-units", type=int, default=8)
    humanize.add_argument("--duration-ticks", type=int, default=3)
    humanize.add_argument("--seed", type=int, default=0)
    humanize.add_argument("--independent-chord-timing", action="store_true")
    _source_digest(humanize)
    _destination(humanize)

    new = sub.add_parser("score-new", help="Generate a minimal, original blank score")
    new.add_argument("--ppqn", type=int, default=480)
    new.add_argument("--end", type=int, default=1920)
    new.add_argument("--title", default="Original composition")
    _destination(new)
    add_score_commands(sub)
    register_history_commands(sub)
    return parser


def _midi_inspect(args) -> dict:
    original = read_smf(args.source, expected_sha256=args.expected_sha256)
    midi = original.midi
    tempo = tempo_summary(midi, track=args.track)
    result = {
        "protocol": "sonumio-midi-workstation-inspection-v1",
        "source_sha256": original.sha256, "source_bytes": original.raw_bytes,
        "format": midi.format, "tracks": len(midi.tracks),
        "events": len(midi.events), "end_tick": midi.end,
        "track_names": [track.name for track in midi.tracks],
        "tempo": tempo,
        "audio_rendered": False,
        "model_inference": False,
    }
    if args.notes:
        policy = PerformanceSettings(
            same_pitch=args.overlap,
            dangling_note_on="close" if args.close_dangling else "reject",
        )
        performance = reconstruct_performance(midi, policy)
        result["performance"] = performance_report(performance)
    return result


def _midi_validate(args) -> dict:
    source = read_smf(args.source, expected_sha256=args.expected_sha256)
    data = source.midi
    # Validate the original decoded structure via a separate semantic encode
    # pass; a canonical file hash may differ due to running-status choices.
    from .smf import encode_smf
    receipt = validate_roundtrip(encode_smf(data))
    receipt["input_sha256"] = source.sha256
    receipt["input_bytes"] = source.raw_bytes
    return receipt


def _midi_import(args) -> dict:
    source = read_smf(args.source, expected_sha256=args.expected_sha256)
    settings = PerformanceSettings(
        dangling_note_on="close" if args.close_dangling else "reject",
    )
    score, ledger = from_midi(source.midi, title=args.title, settings=settings,
                              audible_duration=args.use_audible_duration,
                              allow_patch_changes=args.allow_patch_rechannel)
    publication = publish_score(args.output, score, _publication(args),
                                input_path=args.source)
    return {
        "protocol": "sonumio-workstation-import-receipt-v1",
        "source_sha256": source.sha256,
        "source_bytes": source.raw_bytes,
        "result_score_sha256": score.sha256,
        "publication": publication.as_dict(),
        "conversion": ledger,
    }


def _midi_export(args) -> dict:
    source = load_score(args.source, expected_sha256=args.expected_sha256)
    midi = to_midi(source.score, include_muted=not args.only_active)
    publication = publish_midi(args.output, midi, _publication(args),
                               input_path=args.source)
    return {
        "protocol": "sonumio-workstation-export-receipt-v1",
        "source_sha256": source.source_sha256,
        "canonical_score_sha256": source.canonical_sha256,
        "publication": publication.as_dict(),
        "format": midi.format,
        "track_count": len(midi.tracks),
        "midi_audio_rendered": False,
    }


def _score_inspect(args) -> dict:
    source = load_score(args.source, expected_sha256=args.expected_sha256)
    analysis = score_inspection(
        source.score, window_ticks=args.window_ticks,
        include_muted=args.include_muted,
    )
    return {"source": source.as_dict(), "analysis": analysis}


def _score_quantize(args) -> dict:
    source = load_score(args.source, expected_sha256=args.expected_sha256)
    settings = QuantizeGrid(
        args.step_ticks, offset_ticks=args.offset_ticks,
        swing_percent=args.swing_percent,
        strength_percent=args.strength_percent,
        quantize_releases=args.quantize_releases,
    )
    result, report = quantize_score(source.score, settings)
    published = publish_score(args.output, result, _publication(args),
                              input_path=args.source)
    return {"protocol": "sonumio-workstation-quantize-receipt-v1",
            "source": source.as_dict(), "published": published.as_dict(),
            "quantize": report.as_dict()}


def _score_humanize(args) -> dict:
    source = load_score(args.source, expected_sha256=args.expected_sha256)
    settings = HumanizeSettings(
        timing_ticks=args.timing_ticks, velocity_units=args.velocity_units,
        duration_ticks=args.duration_ticks, seed=args.seed,
        preserve_simultaneous_chords=not args.independent_chord_timing,
    )
    result, report = humanize_score(source.score, settings)
    published = publish_score(args.output, result, _publication(args),
                              input_path=args.source)
    return {"protocol": "sonumio-workstation-humanize-receipt-v1",
            "source": source.as_dict(), "published": published.as_dict(),
            "expression": report}


def _score_new(args) -> dict:
    score = Score.empty(ppqn=args.ppqn, end=args.end, title=args.title)
    published = publish_score(args.output, score, _publication(args))
    return {"protocol": "sonumio-workstation-new-score-v1",
            "published": published.as_dict(),
            "score": score.description()}

_DISPATCH = {
    "midi-inspect": _midi_inspect,
    "midi-validate": _midi_validate,
    "midi-import": _midi_import,
    "midi-export": _midi_export,
    "score-inspect": _score_inspect,
    "score-quantize": _score_quantize,
    "score-humanize": _score_humanize,
    "score-new": _score_new,
    **SCORE_COMMANDS,
    **HISTORY_COMMANDS,
}


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        receipt = _DISPATCH[args.command](args)
        payload = json.dumps(receipt, ensure_ascii=True,
                             sort_keys=True, allow_nan=False)
    except (MidiWorkstationError, OSError, OverflowError, RecursionError) as exc:
        # A failure does not publish an inaccurate success receipt on stdout.
        print(f"Sonumio MIDI workstation: {exc}", file=sys.stderr)
        return 2
    print(payload)
    return 0
