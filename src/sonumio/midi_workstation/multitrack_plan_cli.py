"""Dry-run resource planner: inspect multi-instrument renders without audio."""
from __future__ import annotations

from argparse import _SubParsersAction
from pathlib import Path

from .multitrack_preflight import build_multitrack_preflight
from .multitrack_preview import MultiTrackPreviewSettings
from .multitrack_preview_cli import _patch
from .multitrack_recipe import load_multitrack_recipe
from .score_io import load_score

PROTOCOL = "sonumio-multitrack-resource-plan-cli-v1"


def register_multitrack_plan(sub: _SubParsersAction) -> None:
    plain = sub.add_parser(
        "multitrack-preview-plan",
        help="Check multitrack patch routing, memory and voice budgets without sound",
    )
    plain.add_argument("source", type=Path)
    plain.add_argument("--patch", action="append", required=True,
                       help="Explicit track-id:waveform:gain")
    plain.add_argument("--sample-rate", type=int, default=24000)
    plain.add_argument("--channels", type=int, default=2)
    plain.add_argument("--allow-skip-percussion", action="store_true")
    plain.add_argument("--include-muted", action="store_true")
    plain.add_argument("--normalization", choices=("reject","peak"), default="reject")
    plain.add_argument("--max-peak", type=float, default=.98)
    plain.add_argument("--maximum-notes", type=int, default=2048)
    plain.add_argument("--maximum-work-samples", type=int, default=14_000_000)
    plain.add_argument("--expected-sha256")

    pinned = sub.add_parser(
        "multitrack-plan-from-recipe",
        help="Preflight checksum-pinned multi-instrument recipe without rendering",
    )
    pinned.add_argument("source", type=Path)
    pinned.add_argument("--recipe", type=Path, required=True)
    pinned.add_argument("--expected-sha256")
    pinned.add_argument("--expected-recipe-sha256")


def run_plan(args) -> dict:
    score=load_score(args.source,expected_sha256=args.expected_sha256)
    settings=MultiTrackPreviewSettings(
        tuple(sorted((_patch(x) for x in args.patch),key=lambda p:p.track_id)),
        sample_rate=args.sample_rate,channels=args.channels,
        include_muted=args.include_muted,
        allow_skip_percussion=args.allow_skip_percussion,
        normalization=args.normalization,maximum_peak=args.max_peak,
        maximum_notes=args.maximum_notes,
        maximum_work_samples=args.maximum_work_samples,
    )
    plan=build_multitrack_preflight(score.score,settings)
    return {"protocol":PROTOCOL,"score":score.as_dict(),"plan":plan.as_dict()}


def run_pinned_plan(args) -> dict:
    score=load_score(args.source,expected_sha256=args.expected_sha256)
    recipe,ledger=load_multitrack_recipe(
        args.recipe,expected_sha256=args.expected_recipe_sha256,
    )
    recipe.check_score(score.score)
    plan=build_multitrack_preflight(score.score,recipe.settings)
    return {"protocol":PROTOCOL,"score":score.as_dict(),
            "recipe":ledger,"plan":plan.as_dict()}


COMMANDS={
    "multitrack-preview-plan":run_plan,
    "multitrack-plan-from-recipe":run_pinned_plan,
}
