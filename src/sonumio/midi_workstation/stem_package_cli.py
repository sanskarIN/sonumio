"""Standalone local stem-package planning, rendering and verification.

The CLI is separate from the existing workstation entrypoint so this can be
reviewed as a narrow additive change. It never publishes remotely or runs
untrusted custom scripts: recipe files are exact-schema JSON only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256 as _sha256
import json
from pathlib import Path
import sys

from .contracts import (MidiFormatError, MidiIntegrityError, MidiLimitError,
                        MidiWorkstationError, assert_keys, canonical_json,
                        parse_json, read_regular_file, sha256)
from .publish import PublicationOptions
from .score_io import load_score
from .stem_package import (
    MAX_STEM_PACKAGE_BYTES, MAX_STEMS, StemPackagePlan, StemPatch,
    inspect_stem_pack, preflight_stems, publish_stem_pack,
)

RECIPE_PROTOCOL = "sonumio-original-stem-package-recipe-v1"
MAX_STEM_RECIPE_BYTES = 32768


def read_recipe(data: bytes) -> tuple[StemPackagePlan, str]:
    """Parse only a canonical, pinned, resource-bounded declarative recipe."""
    raw = parse_json(data, limit=MAX_STEM_RECIPE_BYTES)
    root = assert_keys(raw, {
        "protocol", "source_score_sha256", "patches", "sample_rate",
        "channels", "bits", "dither_seed", "include_muted",
        "skip_percussion", "maximum_samples",
    }, context="stem package recipe")
    if root["protocol"] != RECIPE_PROTOCOL:
        raise MidiFormatError("Unknown stem recipe protocol")
    source_digest = sha256(root["source_score_sha256"], "expected score digest")
    patches = root["patches"]
    if type(patches) is not list or not 1 <= len(patches) <= MAX_STEMS:
        raise MidiFormatError("Stem recipe requires 1..8 patch objects")
    parsed = []
    for obj in patches:
        fields = assert_keys(obj, {
            "track_id","waveform","gain","table_size","max_harmonics",
            "attack_frames","decay_frames","sustain_level","release_frames",
        }, context="stem patch object")
        parsed.append(StemPatch(**fields))
    settings=StemPackagePlan(
        tuple(parsed), sample_rate=root["sample_rate"],
        channels=root["channels"],bits=root["bits"],
        dither_seed=root["dither_seed"],include_muted=root["include_muted"],
        skip_percussion=root["skip_percussion"],
        maximum_samples=root["maximum_samples"],
    )
    # Canonical source is required so the same SHA pins the same validated
    # settings and Unicode escape/spaces do not change output semantics.
    if encode_recipe(settings, source_digest) != data:
        raise MidiFormatError("Stem recipe must use canonical JSON encoding")
    return settings, source_digest


def encode_recipe(settings: StemPackagePlan, source_score_sha256: str) -> bytes:
    if not isinstance(settings, StemPackagePlan):
        raise MidiWorkstationError("Expected bounded StemPackagePlan")
    source_digest = sha256(source_score_sha256, "source score digest")
    return canonical_json({
        "protocol":RECIPE_PROTOCOL,
        "source_score_sha256":source_digest,
        "patches":[asdict(p) for p in settings.patches],
        "sample_rate":settings.sample_rate,
        "channels":settings.channels,
        "bits":settings.bits,
        "dither_seed":settings.dither_seed,
        "include_muted":settings.include_muted,
        "skip_percussion":settings.skip_percussion,
        "maximum_samples":settings.maximum_samples,
    }, limit=MAX_STEM_RECIPE_BYTES)


def _read_plan(path: Path, expected_digest: str | None):
    snapshot=read_regular_file(path,limit=MAX_STEM_RECIPE_BYTES,
                               expected_sha256=expected_digest)
    settings,score_digest=read_recipe(snapshot.content)
    return settings,score_digest,snapshot.sha256


def make_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(
        prog="sonumio-stem-package",
        description="Source-only, bounded original wavetable audio stem exports",
    )
    sub=parser.add_subparsers(dest="action",required=True)
    for name in ("plan","render"):
        cmd=sub.add_parser(name,help="Inspect required resources" if name=="plan" else "Publish ZIP of aligned WAV stems")
        cmd.add_argument("source",type=Path,help="Canonical workstation JSON score")
        cmd.add_argument("--recipe",type=Path,required=True,help="Canonical SHA-pinned stem recipe")
        cmd.add_argument("--expected-sha256",help="Expected score file SHA-256")
        cmd.add_argument("--expected-recipe-sha256",help="Expected recipe file SHA-256")
        if name=="render":
            cmd.add_argument("--output",required=True,type=Path)
            cmd.add_argument("--overwrite",action="store_true")
            cmd.add_argument("--create-parent",action="store_true")
            cmd.add_argument("--expected-output-sha256")
            cmd.add_argument("--expected-previous-sha256")
    inspect=sub.add_parser("inspect",help="Verify all stem ZIP members and hashes")
    inspect.add_argument("source",type=Path)
    inspect.add_argument("--expected-sha256",help="Pin entire archive SHA-256")
    return parser


def _plan(args):
    snapshot=load_score(args.source,expected_sha256=args.expected_sha256)
    plan,required_score_digest,recipe_digest=_read_plan(
        args.recipe,args.expected_recipe_sha256)
    if snapshot.canonical_sha256!=required_score_digest:
        raise MidiIntegrityError("Stem recipe points to a different canonical score")
    result=preflight_stems(snapshot.score,plan)
    return {"protocol":"sonumio-stem-package-cli-preflight-v1",
            "preflight":result.as_dict(),
            "input_score_sha256":snapshot.source_sha256,
            "recipe_sha256":recipe_digest,
            "rendered":False,"published":False}


def _render(args):
    snapshot=load_score(args.source,expected_sha256=args.expected_sha256)
    plan,required_score_digest,recipe_digest=_read_plan(
        args.recipe,args.expected_recipe_sha256)
    if snapshot.canonical_sha256!=required_score_digest:
        raise MidiIntegrityError("Stem recipe points to a different canonical score")
    # Both source artifacts must survive independent output publication.
    publication=PublicationOptions(
        overwrite=args.overwrite,create_parent=args.create_parent,
        expected_sha256=args.expected_output_sha256,
        expected_previous_sha256=args.expected_previous_sha256,
    )
    report=publish_stem_pack(snapshot.score,plan,args.output,publication,
                             source_path=args.source,protected_paths=(args.recipe,))
    return {"protocol":"sonumio-stem-package-cli-render-v1",
            "render":report,"input_score_sha256":snapshot.source_sha256,
            "recipe_sha256":recipe_digest,"source_unchanged":True,
            "remote_publication":False,"model_inference":False}


def _inspect(args):
    result=read_regular_file(args.source,limit=MAX_STEM_PACKAGE_BYTES,
                             expected_sha256=args.expected_sha256)
    inspected=inspect_stem_pack(result.content,expected_sha256=result.sha256)
    if inspected["package_sha256"]!=result.sha256:
        raise MidiIntegrityError("Stem package changed while being inspected")
    return inspected


def main(argv: list[str] | None = None) -> int:
    args=make_parser().parse_args(argv)
    try:
        if args.action=="inspect":
            receipt=_inspect(args)
        elif args.action=="plan":
            receipt=_plan(args)
        else:
            receipt=_render(args)
        report=json.dumps(receipt,sort_keys=True,ensure_ascii=True,allow_nan=False)
    except (MidiWorkstationError,OSError,OverflowError,RecursionError) as exc:
        print(f"Sonumio stem package: {exc}",file=sys.stderr)
        return 2
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
