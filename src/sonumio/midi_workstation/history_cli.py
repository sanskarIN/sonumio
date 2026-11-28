"""Local checksum-aware undo/redo journal and musical comparison commands.

This interface is intentionally separate from Sonumio's existing main CLI.
It accepts only bounded ordinary score/journal files and writes only to an
explicit caller-selected destination using atomic, no-clobber publication.
"""
from __future__ import annotations

import json
from pathlib import Path

from .compare import CompareSettings, compare_scores
from .contracts import (
    MidiFormatError, MidiIntegrityError, MidiLimitError, MidiWorkstationError,
    read_regular_file,
)
from .journal import MAX_HISTORY_BYTES, RevisionJournal
from .publish import PublicationOptions, publish_bytes, publish_score, _distinct
from .score_io import load_score


def _pin(parser, name="--expected-sha256"):
    parser.add_argument(name)


def _out(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--create-parent", action="store_true")


def register_history_commands(sub):
    compare = sub.add_parser(
        "score-compare", help="Bounded note matching and tempo-clock diagnostics",
    )
    compare.add_argument("source", type=Path)
    compare.add_argument("target", type=Path)
    compare.add_argument("--onset-tolerance", type=int, default=120)
    compare.add_argument("--release-tolerance", type=int, default=120)
    compare.add_argument("--pitch-mode", choices=("exact", "class"), default="exact")
    compare.add_argument("--route-mode", choices=("track", "channel"), default="track")
    compare.add_argument("--include-muted", action="store_true")
    compare.add_argument("--source-sha256")
    compare.add_argument("--target-sha256")

    init = sub.add_parser("history-init", help="Start a digest-chained local score journal")
    init.add_argument("source", type=Path)
    init.add_argument("--label", default="Initial score")
    _pin(init)
    _out(init)

    commit = sub.add_parser("history-commit", help="Append a pinned score revision to journal")
    commit.add_argument("source", type=Path)
    commit.add_argument("score", type=Path)
    commit.add_argument("--label", required=True)
    commit.add_argument("--score-sha256")
    commit.add_argument("--expected-current-sha256", required=True)
    _pin(commit)
    _out(commit)

    for command in ("history-undo", "history-redo"):
        op = sub.add_parser(command, help=f"Advance local score history cursor: {command}")
        op.add_argument("source", type=Path)
        op.add_argument("--expected-current-sha256")
        _pin(op)
        _out(op)

    inspect = sub.add_parser("history-inspect", help="Read-only journal inspection")
    inspect.add_argument("source", type=Path)
    _pin(inspect)

    checkout = sub.add_parser("history-checkout", help="Export selected journal snapshot")
    checkout.add_argument("source", type=Path)
    checkout.add_argument("--index", type=int)
    _pin(checkout)
    _out(checkout)


def _options(args):
    return PublicationOptions(
        overwrite=args.overwrite, create_parent=args.create_parent,
        expected_sha256=args.expected_output_sha256,
    )


def _journal(path, expected_sha256):
    snapshot = read_regular_file(path, limit=MAX_HISTORY_BYTES,
                                 expected_sha256=expected_sha256)
    return RevisionJournal.from_bytes(snapshot.content), snapshot


def _protect(sources, output):
    for source in sources:
        _distinct(source, output)


def _publish_journal(journal, args, sources):
    _protect(sources, args.output)
    raw = journal.to_bytes()
    published = publish_bytes(args.output, raw, _options(args),
                              input_path=sources[0], expected_extension=".json")
    return {
        "publication": published.as_dict(),
        "journal": journal.summary(),
        "input_files_unchanged": True,
    }


def score_compare_command(args):
    a = load_score(args.source, expected_sha256=args.source_sha256)
    b = load_score(args.target, expected_sha256=args.target_sha256)
    settings = CompareSettings(
        onset_tolerance_ticks=args.onset_tolerance,
        release_tolerance_ticks=args.release_tolerance,
        pitch_mode=args.pitch_mode, route_mode=args.route_mode,
        include_muted=args.include_muted,
    )
    return {
        "protocol": "sonumio-workstation-compare-cli-v1",
        "source_file_sha256": a.source_sha256,
        "target_file_sha256": b.source_sha256,
        "comparison": compare_scores(a.score, b.score, settings).as_dict(),
        "files_written": 0,
    }


def history_init_command(args):
    _protect((args.source,), args.output)
    source = load_score(args.source, expected_sha256=args.expected_sha256)
    journal = RevisionJournal.begin(source.score, label=args.label)
    receipt = _publish_journal(journal, args, (args.source,))
    return {"protocol": "sonumio-history-initialization-v1",
            "source_sha256": source.source_sha256, **receipt}


def history_commit_command(args):
    sources = (args.source, args.score)
    _protect(sources, args.output)
    journal, snapshot = _journal(args.source, args.expected_sha256)
    source_score = load_score(args.score, expected_sha256=args.score_sha256)
    updated = journal.commit(
        source_score.score, args.label,
        expected_current_sha256=args.expected_current_sha256,
    )
    receipt = _publish_journal(updated, args, sources)
    return {"protocol": "sonumio-history-commit-v1",
            "input_journal_sha256": snapshot.sha256,
            "input_score_sha256": source_score.source_sha256, **receipt}


def _history_cursor(args, direction):
    _protect((args.source,), args.output)
    journal, snapshot = _journal(args.source, args.expected_sha256)
    if direction == "undo":
        updated = journal.undo(expected_current_sha256=args.expected_current_sha256)
    else:
        updated = journal.redo(expected_current_sha256=args.expected_current_sha256)
    receipt = _publish_journal(updated, args, (args.source,))
    return {"protocol": f"sonumio-history-{direction}-v1",
            "input_journal_sha256": snapshot.sha256, **receipt}


def history_undo_command(args):
    return _history_cursor(args, "undo")


def history_redo_command(args):
    return _history_cursor(args, "redo")


def history_inspect_command(args):
    journal, snapshot = _journal(args.source, args.expected_sha256)
    return {"protocol": "sonumio-history-inspect-v1",
            "input_sha256": snapshot.sha256,
            "journal": journal.summary(),
            "files_written": 0}


def history_checkout_command(args):
    _protect((args.source,), args.output)
    journal, snapshot = _journal(args.source, args.expected_sha256)
    current = journal if args.index is None else journal.seek(args.index)
    publication = publish_score(args.output, current.current, _options(args),
                                input_path=args.source)
    return {"protocol": "sonumio-history-checkout-v1",
            "input_sha256": snapshot.sha256,
            "selected_cursor": current.cursor,
            "selected_score_sha256": current.current_digest,
            "publication": publication.as_dict()}


COMMANDS = {
    "score-compare": score_compare_command,
    "history-init": history_init_command,
    "history-commit": history_commit_command,
    "history-undo": history_undo_command,
    "history-redo": history_redo_command,
    "history-inspect": history_inspect_command,
    "history-checkout": history_checkout_command,
}
