"""Optional local, digest-pinned revision and clip commands for the workstation.

All commands read ordinary local score files with strict JSON validation.
Only explicitly named output destinations may be modified. Conflicted
merges fail without publishing anything, including success receipts.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import MidiFormatError, MidiLimitError, MidiWorkstationError
from .publish import PublicationOptions, publish_score, _distinct
from .regions import TickRegion, concatenate_scores, crop_score
from .revisions import diff_scores, merge_scores
from .score_io import load_score
from .pattern_io import load_pattern_recipe


def add_score_commands(sub):
    difference = sub.add_parser(
        "score-diff", help="Report ID-aware difference between two local scores",
    )
    difference.add_argument("before", type=Path)
    difference.add_argument("after", type=Path)
    difference.add_argument("--before-sha256")
    difference.add_argument("--after-sha256")

    merge = sub.add_parser(
        "score-merge", help="Three-way merge of independently edited scores",
    )
    merge.add_argument("base", type=Path)
    merge.add_argument("ours", type=Path)
    merge.add_argument("theirs", type=Path)
    merge.add_argument("--base-sha256")
    merge.add_argument("--ours-sha256")
    merge.add_argument("--theirs-sha256")
    merge.add_argument("--strict-revision", action="store_true")
    _output_arguments(merge)

    crop = sub.add_parser(
        "score-crop", help="Extract exact half-open score region with tempo context",
    )
    crop.add_argument("source", type=Path)
    crop.add_argument("--start-tick", required=True, type=int)
    crop.add_argument("--stop-tick", required=True, type=int)
    crop.add_argument("--reject-crossing", action="store_true")
    crop.add_argument("--expected-sha256")
    _output_arguments(crop)

    concatenate = sub.add_parser(
        "score-concat", help="Concatenate up to 16 matched local symbolic clips",
    )
    concatenate.add_argument("sources", nargs="+", type=Path)
    concatenate.add_argument("--title")
    _output_arguments(concatenate)

    pattern = sub.add_parser("score-pattern", help="Render a strict original symbolic pattern recipe")
    pattern.add_argument("source", type=Path)
    pattern.add_argument("--expected-sha256")
    _output_arguments(pattern)


def _output_arguments(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--create-parent", action="store_true")


def _options(args):
    return PublicationOptions(
        overwrite=args.overwrite,
        create_parent=args.create_parent,
        expected_sha256=args.expected_output_sha256,
    )


def score_diff_command(args):
    source = load_score(args.before, expected_sha256=args.before_sha256)
    result = load_score(args.after, expected_sha256=args.after_sha256)
    return {
        "protocol": "sonumio-score-diff-cli-v1",
        "before_file_sha256": source.source_sha256,
        "after_file_sha256": result.source_sha256,
        "difference": diff_scores(source.score, result.score).as_dict(),
        "output_files_created": 0,
    }


def _protect_inputs(paths, output):
    for source in paths:
        _distinct(source, output)


def score_merge_command(args):
    sources = (args.base, args.ours, args.theirs)
    _protect_inputs(sources, args.output)
    a = load_score(args.base, expected_sha256=args.base_sha256)
    b = load_score(args.ours, expected_sha256=args.ours_sha256)
    c = load_score(args.theirs, expected_sha256=args.theirs_sha256)
    merged = merge_scores(a.score, b.score, c.score,
                          strict_revision=args.strict_revision)
    if not merged.succeeded:
        locations = ", ".join(change.path for change in merged.conflicts[:6])
        raise MidiWorkstationError(
            f"Score merge has {len(merged.conflicts)} unresolved conflicts: {locations}"
        )
    published = publish_score(args.output, merged.score, _options(args),
                              input_path=args.base)
    return {
        "protocol": "sonumio-score-merge-cli-v1",
        "merge": merged.as_dict(),
        "publication": published.as_dict(),
        "input_file_sha256": [a.source_sha256, b.source_sha256, c.source_sha256],
    }


def score_crop_command(args):
    _protect_inputs((args.source,), args.output)
    source = load_score(args.source, expected_sha256=args.expected_sha256)
    region = TickRegion(args.start_tick, args.stop_tick)
    extracted, report = crop_score(
        source.score, region,
        crossing="reject" if args.reject_crossing else "trim",
    )
    publication = publish_score(args.output, extracted, _options(args),
                                input_path=args.source)
    return {
        "protocol": "sonumio-score-crop-cli-v1",
        "source_sha256": source.source_sha256,
        "region": report.as_dict(),
        "publication": publication.as_dict(),
    }


def score_concat_command(args):
    if len(args.sources) > 16:
        raise MidiLimitError("Score concatenation supports at most sixteen parts")
    _protect_inputs(args.sources, args.output)
    sources = tuple(load_score(path) for path in args.sources)
    result, report = concatenate_scores(
        tuple(source.score for source in sources), title=args.title,
    )
    publication = publish_score(
        args.output, result, _options(args), input_path=args.sources[0],
    )
    return {
        "protocol": "sonumio-score-concat-cli-v1",
        "input_file_sha256": [source.source_sha256 for source in sources],
        "concatenation": report,
        "publication": publication.as_dict(),
    }


def score_pattern_command(args):
    _protect_inputs((args.source,), args.output)
    recipe, snapshot = load_pattern_recipe(args.source, expected_sha256=args.expected_sha256)
    score, report = recipe.render()
    publication = publish_score(args.output, score, _options(args), input_path=args.source)
    return {
        "protocol": "sonumio-score-pattern-cli-v1",
        "recipe": snapshot,
        "render": report.as_dict(),
        "publication": publication.as_dict(),
    }


COMMANDS = {
    "score-diff": score_diff_command,
    "score-merge": score_merge_command,
    "score-crop": score_crop_command,
    "score-concat": score_concat_command,
    "score-pattern": score_pattern_command,
}
