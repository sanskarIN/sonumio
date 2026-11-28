"""Explicit manifest-bound multi-instrument preview and recipe inspection."""
from __future__ import annotations

from argparse import _SubParsersAction
from pathlib import Path

from .contracts import MidiWorkstationError
from .multitrack_preview import export_multitrack_preview
from .multitrack_recipe import load_multitrack_recipe
from .publish import _distinct
from .score_io import load_score

PROTOCOL = "sonumio-multitrack-recipe-cli-v1"


def register_multitrack_recipe(sub: _SubParsersAction) -> None:
    inspect = sub.add_parser(
        "multitrack-recipe-inspect",
        help="Validate a bounded score-pinned mathematical instrument recipe",
    )
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--expected-sha256")

    render = sub.add_parser(
        "multitrack-preview-from-recipe",
        help="Render a pinned score through explicitly declared track instruments",
    )
    render.add_argument("source", type=Path, help="Canonical source workstation score")
    render.add_argument("--recipe", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--expected-sha256", help="Expected score file checksum")
    render.add_argument("--expected-recipe-sha256")
    render.add_argument("--expected-output-sha256")
    render.add_argument("--output-bits", type=int, choices=(8, 16, 24, 32), default=16)
    render.add_argument("--dither-seed", type=int)
    render.add_argument("--overwrite", action="store_true")
    render.add_argument("--create-parent", action="store_true")


def inspect_recipe(args) -> dict:
    recipe, ledger = load_multitrack_recipe(
        args.source, expected_sha256=args.expected_sha256,
    )
    return {
        "protocol": PROTOCOL,
        "source": ledger,
        "patches": len(recipe.settings.patches),
        "sample_rate": recipe.settings.sample_rate,
        "channels": recipe.settings.channels,
        "normalization": recipe.settings.normalization,
        "score_sha256": recipe.score_sha256,
        "audio_rendered": False,
    }


def render_recipe(args) -> dict:
    if args.source == args.recipe:
        raise MidiWorkstationError("Recipe and score must be distinct source files")
    _distinct(args.source, args.output)
    _distinct(args.recipe, args.output)
    score = load_score(args.source, expected_sha256=args.expected_sha256)
    recipe, recipe_ledger = load_multitrack_recipe(
        args.recipe, expected_sha256=args.expected_recipe_sha256,
    )
    recipe.check_score(score.score)
    result = export_multitrack_preview(
        score.score, args.output, recipe.settings,
        source_path=args.source,
        protected_paths=(args.recipe,),
        output_bits=args.output_bits,
        dither_seed=args.dither_seed,
        overwrite=args.overwrite,
        create_parent=args.create_parent,
        expected_output_sha256=args.expected_output_sha256,
    )
    return {
        "protocol": PROTOCOL,
        "score": score.as_dict(),
        "recipe": recipe_ledger,
        "preview": result,
    }


COMMANDS = {
    "multitrack-recipe-inspect": inspect_recipe,
    "multitrack-preview-from-recipe": render_recipe,
}
