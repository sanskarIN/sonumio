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
from .raw_edit import MidiTransformSettings, transform_midi
from .lint import LintSettings, lint_midi
from .score_preview import ScorePreviewOptions, export_score_preview
from .wavetable_bridge import WavetableBridgeSettings
from .score import Score, from_midi, to_midi
from .score_io import load_score, parse_score_snapshot
from .smf import read_smf, validate_roundtrip
from .tempo import tempo_summary
from .score_commands import add_score_commands, COMMANDS as SCORE_COMMANDS
from .history_cli import register_history_commands, COMMANDS as HISTORY_COMMANDS
from .arrangement_bridge_cli import register_arrangement_bridge, COMMANDS as BRIDGE_COMMANDS
from .arrangement_import_cli import register_arrangement_import, COMMANDS as IMPORT_COMMANDS


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

    midi_lint = sub.add_parser(
        "midi-lint", help="Inspect raw MIDI for ambiguous musical semantics"
    )
    midi_lint.add_argument("source", type=Path)
    midi_lint.add_argument("--max-findings", type=int, default=256)
    midi_lint.add_argument("--no-information", action="store_true")
    midi_lint.add_argument("--fail-on-warnings", action="store_true")
    midi_lint.add_argument("--fail-on-errors", action="store_true")
    _source_digest(midi_lint)

    midi_transform = sub.add_parser(
        "midi-transform",
        help="Non-destructively transpose selected MIDI notes and scale key-on velocity",
    )
    midi_transform.add_argument("source", type=Path)
    midi_transform.add_argument("--semitones", type=int, default=0)
    midi_transform.add_argument("--velocity-numerator", type=int, default=1)
    midi_transform.add_argument("--velocity-denominator", type=int, default=1)
    midi_transform.add_argument("--velocity-offset", type=int, default=0)
    midi_transform.add_argument("--pitch-low", type=int, default=0)
    midi_transform.add_argument("--pitch-high", type=int, default=127)
    midi_transform.add_argument("--track", action="append", type=int)
    midi_transform.add_argument("--channel", action="append", type=int)
    midi_transform.add_argument("--include-percussion", action="store_true")
    midi_transform.add_argument("--transpose-key-signatures", action="store_true")
    midi_transform.add_argument("--allow-new-overlaps", action="store_true",
                                help="Explicitly allow new same-pitch overlapping notes")
    _source_digest(midi_transform)
    _destination(midi_transform)

    preview = sub.add_parser(
        "score-preview-wav",
        help="Render an original symbolic score through Sonumio audio_core wavetable",
    )
    preview.add_argument("source", type=Path)
    preview.add_argument("--sample-rate", type=int, default=24000)
    preview.add_argument("--channels", type=int, default=2)
    preview.add_argument("--waveform", choices=("sine", "saw", "square", "triangle"),
                         default="triangle")
    preview.add_argument("--table-size", type=int, default=2048)
    preview.add_argument("--max-harmonics", type=int, default=64)
    preview.add_argument("--master-gain", type=float, default=.15)
    preview.add_argument("--attack-frames", type=int, default=64)
    preview.add_argument("--decay-frames", type=int, default=256)
    preview.add_argument("--sustain-level", type=float, default=.7)
    preview.add_argument("--release-frames", type=int, default=512)
    preview.add_argument("--include-muted", action="store_true")
    preview.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32), default=16)
    preview.add_argument("--dither-seed", type=int)
    _source_digest(preview)
    _destination(preview)

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
    register_arrangement_bridge(sub)
    register_arrangement_import(sub)
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


def _score_preview_wav(args) -> dict:
    source = load_score(args.source, expected_sha256=args.expected_sha256)
    bridge = WavetableBridgeSettings(
        sample_rate=args.sample_rate,
        channels=args.channels,
        waveform=args.waveform,
        table_size=args.table_size,
        max_harmonics=args.max_harmonics,
        master_gain=args.master_gain,
        attack_frames=args.attack_frames,
        decay_frames=args.decay_frames,
        sustain_level=args.sustain_level,
        release_frames=args.release_frames,
        include_muted=args.include_muted,
    )
    result = export_score_preview(
        source.score, args.output,
        ScorePreviewOptions(
            bridge=bridge, output_bits=args.output_bits,
            dither_seed=args.dither_seed,
            overwrite=args.overwrite, create_parent=args.create_parent,
            expected_output_sha256=args.expected_output_sha256,
        ),
        source_path=args.source,
    )
    return {
        "protocol": "sonumio-workstation-preview-receipt-v1",
        "source": source.as_dict(),
        "preview": result.as_dict(),
    }


def _midi_lint(args) -> dict:
    source = read_smf(args.source, expected_sha256=args.expected_sha256)
    report = lint_midi(source.midi, LintSettings(
        maximum_findings=args.max_findings,
        include_information=not args.no_information,
    ))
    if args.fail_on_warnings and report.status != "clean":
        raise MidiWorkstationError("MIDI lint gate detected warnings or errors")
    if args.fail_on_errors and report.status == "errors":
        raise MidiWorkstationError("MIDI lint gate detected errors")
    return {
        "protocol": "sonumio-workstation-midi-lint-receipt-v1",
        "source_sha256": source.sha256,
        "source_bytes": source.raw_bytes,
        "lint": report.as_dict(),
    }


def _midi_transform(args) -> dict:
    source = read_smf(args.source, expected_sha256=args.expected_sha256)
    settings = MidiTransformSettings(
        semitones=args.semitones,
        velocity_numerator=args.velocity_numerator,
        velocity_denominator=args.velocity_denominator,
        velocity_offset=args.velocity_offset,
        pitch_low=args.pitch_low,
        pitch_high=args.pitch_high,
        tracks=tuple(sorted(set(args.track))) if args.track else None,
        channels=tuple(sorted(set(args.channel))) if args.channel else None,
        include_percussion=args.include_percussion,
        transpose_key_signatures=args.transpose_key_signatures,
        reject_new_collisions=not args.allow_new_overlaps,
    )
    transformed, report = transform_midi(source.midi, settings)
    publication = publish_midi(args.output, transformed, _publication(args),
                               input_path=args.source)
    if publication.output_sha256 != report.result_sha256:
        raise MidiIntegrityError("Published MIDI differs from transformed stream")
    return {
        "protocol": "sonumio-workstation-transform-receipt-v1",
        "source_sha256": source.sha256,
        "source_bytes": source.raw_bytes,
        "canonical_source_sha256": report.source_sha256,
        "transform": report.as_dict(),
        "publication": publication.as_dict(),
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
    "midi-transform": _midi_transform,
    "midi-lint": _midi_lint,
    "score-preview-wav": _score_preview_wav,
    "score-inspect": _score_inspect,
    "score-quantize": _score_quantize,
    "score-humanize": _score_humanize,
    "score-new": _score_new,
    **SCORE_COMMANDS,
    **HISTORY_COMMANDS,
    **BRIDGE_COMMANDS,
    **IMPORT_COMMANDS,
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
