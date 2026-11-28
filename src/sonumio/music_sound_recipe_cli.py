"""Create, render and replay complete original sound-design recipes locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .atomic_cli_wav import publish_generated_wav
from .errors import GenerationError
from .music_drum_bank import read_drum_bank
from .music_patch_bank import read_patch_bank
from .music_project_cli import _distinct,_output
from .music_publish import publish_music_artifact
from .music_render_recipe import read_recipe,encode_recipe
from .music_sound_recipe import SoundRecipe,encode_sound_recipe,read_sound_recipe,describe_sound,render_sound
from .music_sound_bundle import create_sound_bundle,verify_sound_bundle,read_sound_bundle


def _publish(args,data,kind):
    receipt=publish_music_artifact(args.output,data,kind=kind,overwrite=args.overwrite,
                                   expected_sha256=args.expected_output_sha256)
    return {"output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written}


def _publish_wav(args,data):
    receipt=publish_generated_wav(args.output,data,overwrite=args.overwrite,
                                  expected_sha256=args.expected_output_sha256)
    return {"output_sha256":receipt.wav_sha256,"bytes_written":receipt.bytes_written}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    create=sub.add_parser("create")
    create.add_argument("recipe",type=Path)
    create.add_argument("--expected-recipe-sha256")
    create.add_argument("--pitched",type=Path)
    create.add_argument("--expected-pitched-sha256")
    create.add_argument("--drums",type=Path)
    create.add_argument("--expected-drums-sha256")
    _output(create)
    for name in ("inspect","render","bundle"):
        p=sub.add_parser(name)
        p.add_argument("sound",type=Path)
        p.add_argument("--expected-sound-sha256")
        if name!="inspect":
            _output(p)
    for name in ("verify","extract-wav","extract-sound","extract-base-recipe"):
        p=sub.add_parser(name)
        p.add_argument("bundle",type=Path)
        p.add_argument("--expected-sound-sha256")
        p.add_argument("--replay",action="store_true")
        if name!="verify":
            _output(p)
    args=parser.parse_args(argv)
    try:
        if args.action=="create":
            _distinct((args.recipe,args.pitched,args.drums),args.output)
            base=read_recipe(args.recipe,expected_sha256=args.expected_recipe_sha256)
            if args.pitched is None and args.expected_pitched_sha256 is not None:
                raise GenerationError("Pitched pin requires a pitched bank")
            if args.drums is None and args.expected_drums_sha256 is not None:
                raise GenerationError("Drum pin requires a drum bank")
            pitched=read_patch_bank(args.pitched,expected_sha256=args.expected_pitched_sha256) if args.pitched is not None else None
            drums=read_drum_bank(args.drums,expected_sha256=args.expected_drums_sha256) if args.drums is not None else None
            sound=SoundRecipe(base,pitched,drums)
            report=describe_sound(sound)
            report.update(_publish(args,encode_sound_recipe(sound),"sound_recipe"))
        elif args.action in ("inspect","render","bundle"):
            sound=read_sound_recipe(args.sound,expected_sha256=args.expected_sound_sha256)
            report=describe_sound(sound)
            if args.action!="inspect":
                _distinct((args.sound,),args.output)
                if args.action=="render":
                    wav,report=render_sound(sound)
                    report={"render_receipt":report,**_publish_wav(args,wav)}
                else:
                    data=create_sound_bundle(sound)
                    _,_,_,report=verify_sound_bundle(data)
                    report.update(_publish(args,data,"render_bundle"))
        else:
            sound,wav,_,report=verify_sound_bundle(read_sound_bundle(args.bundle),
                expected_sha256=args.expected_sound_sha256,replay=args.replay)
            if args.action!="verify":
                _distinct((args.bundle,),args.output)
                if args.action=="extract-wav":
                    report.update(_publish_wav(args,wav))
                elif args.action=="extract-sound":
                    report.update(_publish(args,encode_sound_recipe(sound),"sound_recipe"))
                else:
                    report.update(_publish(args,encode_recipe(sound.recipe),"render_recipe"))
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio complete sound recipe: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
