"""Generate and verify deterministic original procedural sound libraries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .asset_archive import load_pack_plan,export_sound_pack,verify_pack_file
from .asset_pack import plan_jobs
from .asset_selection import list_pack_file,export_pack_asset


def register_asset_pack_commands(sub):
    create=sub.add_parser("sound-pack",help="Generate a verified original procedural WAV ZIP")
    create.add_argument("--plan",required=True,type=Path)
    create.add_argument("--output",required=True,type=Path)
    create.add_argument("--expected-sha256")
    create.add_argument("--overwrite",action="store_true")
    create.add_argument("--create-parent",action="store_true")
    inspect=sub.add_parser("sound-pack-verify",help="Verify canonical archive and integer PCM receipts")
    inspect.add_argument("input",type=Path)
    inspect.add_argument("--expected-sha256")
    inspect.add_argument("--replay",action="store_true",
                         help="Regenerate each original sound and require exact WAV bytes")
    listing=sub.add_parser("sound-pack-list",help="List verified assets by tag and source")
    listing.add_argument("input",type=Path)
    listing.add_argument("--tag",action="append",default=[])
    listing.add_argument("--source",choices=("nature","procedural"))
    listing.add_argument("--expected-sha256")
    select=sub.add_parser("sound-pack-select",help="Publish one verified WAV without archive extraction")
    select.add_argument("input",type=Path)
    select.add_argument("--asset",required=True)
    select.add_argument("--variant",type=int,default=1)
    select.add_argument("--output",type=Path,required=True)
    select.add_argument("--expected-sha256")
    select.add_argument("--expected-wav-sha256")
    select.add_argument("--overwrite",action="store_true")
    select.add_argument("--create-parent",action="store_true")
    preview=sub.add_parser("sound-pack-plan",help="Validate and display jobs without synthesis")
    preview.add_argument("--plan",required=True,type=Path)


def run_asset_pack_command(args):
    try:
        if args.command=="sound-pack-plan":
            plan=load_pack_plan(args.plan)
            report={
                "plan_sha256":plan.sha256,
                "asset_count":plan.asset_count,
                "total_frames":plan.total_frames,
                "payload_bytes":plan.payload_bytes,
                "jobs":[{
                    "member":job.member,"seed":job.seed,"dither_seed":job.dither_seed,
                    "frames":job.recipe.frames,"source":job.recipe.source,
                } for job in plan_jobs(plan)],
            }
        elif args.command=="sound-pack-list":
            report=list_pack_file(args.input,tags=tuple(args.tag),source=args.source,
                                  expected_archive_sha256=args.expected_sha256)
        elif args.command=="sound-pack-select":
            report=export_pack_asset(args.input,args.output,args.asset,args.variant,
                expected_archive_sha256=args.expected_sha256,expected_wav_sha256=args.expected_wav_sha256,
                overwrite=args.overwrite,create_parent=args.create_parent)
        elif args.command=="sound-pack-verify":
            report=verify_pack_file(args.input,expected_sha256=args.expected_sha256,
                                    replay=args.replay).as_dict()
        else:
            if args.plan.resolve()==args.output.resolve():
                raise ValueError("Sound pack plan and output must differ")
            if args.output.exists() and os.path.samefile(args.plan,args.output):
                raise ValueError("Sound pack plan and output share a file identity")
            plan=load_pack_plan(args.plan)
            report=export_sound_pack(plan,args.output,expected_sha256=args.expected_sha256,
                overwrite=args.overwrite,create_parent=args.create_parent).as_dict()
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio sound pack: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    register_asset_pack_commands(sub)
    return run_asset_pack_command(parser.parse_args(argv))


if __name__=="__main__":
    raise SystemExit(main())
