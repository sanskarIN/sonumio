"""Portable, SHA-pinned ZIP-stems to PCM WAV mixdown tool.

The input recipe is canonical JSON with a complete set of explicit track
gains and is bound to one immutable stem archive. Users can generate a
unity-gain review recipe, edit it intentionally, and apply it with an
independent digest guard. No network or DAW plugin is involved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .contracts import (MidiFormatError,MidiIntegrityError,MidiWorkstationError,
                        assert_keys,canonical_json,parse_json,read_regular_file,sha256)
from .stem_package import MAX_STEM_PACKAGE_BYTES, inspect_stem_pack
from .stem_mixdown import (
    StemMixGain,StemMixSettings,mix_stem_package,publish_stem_mixdown,
)
from .publish import PublicationOptions,publish_bytes,_distinct

RECIPE_PROTOCOL="sonumio-stem-package-mix-recipe-v1"
MAX_MIX_RECIPE_BYTES=16384


def encode_mix_recipe(settings: StemMixSettings,package_digest: str) -> bytes:
    if not isinstance(settings,StemMixSettings):
        raise MidiWorkstationError("Expected StemMixSettings")
    sha256(package_digest,"stem package digest")
    return canonical_json({
        "protocol":RECIPE_PROTOCOL,
        "stem_package_sha256":package_digest,
        "gains":[asdict(gain) for gain in settings.gains],
        "master_milli":settings.master_milli,
        "peak_policy":settings.peak_policy,
        "target_peak_milli":settings.target_peak_milli,
        "output_bits":settings.output_bits,
        "dither_seed":settings.dither_seed,
        "maximum_samples":settings.maximum_samples,
    },limit=MAX_MIX_RECIPE_BYTES)


def decode_mix_recipe(raw: bytes) -> tuple[StemMixSettings,str]:
    doc=parse_json(raw,limit=MAX_MIX_RECIPE_BYTES)
    attrs=assert_keys(doc,{
        "protocol","stem_package_sha256","gains","master_milli",
        "peak_policy","target_peak_milli","output_bits","dither_seed",
        "maximum_samples",
    },context="stem mix recipe")
    if attrs["protocol"]!=RECIPE_PROTOCOL:
        raise MidiFormatError("Unknown stem mix recipe version")
    package_digest=sha256(attrs["stem_package_sha256"],"stem package digest")
    row=attrs["gains"]
    if type(row) is not list or not 1<=len(row)<=8:
        raise MidiFormatError("Stem mix recipe requires 1..8 explicit gains")
    gains=[]
    for entry in row:
        fields=assert_keys(entry,{"track_id","gain_milli"},
                           context="stem mix gain")
        gains.append(StemMixGain(**fields))
    settings=StemMixSettings(tuple(gains),master_milli=attrs["master_milli"],
           peak_policy=attrs["peak_policy"],target_peak_milli=attrs["target_peak_milli"],
           output_bits=attrs["output_bits"],dither_seed=attrs["dither_seed"],
           maximum_samples=attrs["maximum_samples"])
    if encode_mix_recipe(settings,package_digest)!=raw:
        raise MidiFormatError("Stem mix recipe must be canonical JSON")
    return settings,package_digest


def make_parser():
    parser=argparse.ArgumentParser(prog="sonumio-stem-mixdown",
             description="Original-source, verified PCM stem mixing")
    sub=parser.add_subparsers(dest="action",required=True)
    defaults=sub.add_parser("unity-recipe",help="Create a pinned all-unity gain recipe")
    defaults.add_argument("source",type=Path)
    defaults.add_argument("--output",required=True,type=Path)
    defaults.add_argument("--expected-sha256")
    defaults.add_argument("--overwrite",action="store_true")
    defaults.add_argument("--create-parent",action="store_true")
    defaults.add_argument("--expected-output-sha256")
    for action in ("plan","render"):
        command=sub.add_parser(action,help="Preview deterministic mix" if action=="plan" else "Publish PCM WAV mix")
        command.add_argument("source",type=Path,help="Digest-verified stem ZIP")
        command.add_argument("--recipe",type=Path,required=True)
        command.add_argument("--expected-recipe-sha256")
        if action=="render":
            command.add_argument("--output",required=True,type=Path)
            command.add_argument("--overwrite",action="store_true")
            command.add_argument("--create-parent",action="store_true")
            command.add_argument("--expected-output-sha256")
    return parser


def _read_archive(path:Path, expected_sha256=None):
    return read_regular_file(path,limit=MAX_STEM_PACKAGE_BYTES,
                             expected_sha256=expected_sha256)


def _unity(args):
    source=_read_archive(args.source,expected_sha256=args.expected_sha256)
    metadata=inspect_stem_pack(source.content,expected_sha256=source.sha256)
    ids=[path.split("/")[-1].removesuffix(".wav") for path in metadata["members"]]
    settings=StemMixSettings(tuple(StemMixGain(track) for track in ids))
    raw=encode_mix_recipe(settings,source.sha256)
    published=publish_bytes(args.output,raw,PublicationOptions(
        overwrite=args.overwrite,create_parent=args.create_parent,
        expected_sha256=args.expected_output_sha256),
        input_path=args.source,expected_extension=".json")
    return {"protocol":"sonumio-stem-mix-unity-recipe-v1",
            "stem_package_sha256":source.sha256,
            "recipe_sha256":published.output_sha256,
            "tracks":ids,"publication":published.as_dict(),
            "remote_publication":False,"audio_rendered":False}


def _read_plan(args):
    source=_read_archive(args.source)
    recipe=read_regular_file(args.recipe,limit=MAX_MIX_RECIPE_BYTES,
                             expected_sha256=args.expected_recipe_sha256)
    settings,pin=decode_mix_recipe(recipe.content)
    if pin!=source.sha256:
        raise MidiIntegrityError("Mix recipe does not pin the current stem package")
    _distinct(args.recipe,args.source)
    return source,recipe,settings


def _plan(args):
    source,recipe,settings=_read_plan(args)
    mix=mix_stem_package(source.content,settings)
    return {"protocol":"sonumio-stem-mix-preflight-v1",
            "source_sha256":source.sha256,"recipe_sha256":recipe.sha256,
            "mix":mix.report.as_dict(),"wav_published":False,
            "trained_model_inference":False}


def _render(args):
    source,recipe,settings=_read_plan(args)
    _distinct(args.recipe,args.output)
    result=publish_stem_mixdown(source.content,settings,args.output,
          source_path=args.source,overwrite=args.overwrite,
          create_parent=args.create_parent,
          expected_output_sha256=args.expected_output_sha256)
    return {"protocol":"sonumio-stem-mix-render-cli-v1",
            "source_sha256":source.sha256,"recipe_sha256":recipe.sha256,
            "render":result,"source_unchanged":True,
            "remote_publication":False}


def main(argv:list[str]|None=None) -> int:
    args=make_parser().parse_args(argv)
    try:
        if args.action=="unity-recipe":receipt=_unity(args)
        elif args.action=="plan":receipt=_plan(args)
        else:receipt=_render(args)
        serialized=json.dumps(receipt,ensure_ascii=True,sort_keys=True,allow_nan=False)
    except (MidiWorkstationError,OSError,RecursionError,OverflowError) as exc:
        print(f"Sonumio stem mixdown: {exc}",file=sys.stderr)
        return 2
    print(serialized)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
