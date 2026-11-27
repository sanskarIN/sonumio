"""CLI for verified named-source moving stereo scenes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

from .spatial_io import SourceBinding,source_id,load_spatial_plan,export_spatial_plan


def register_spatial_command(sub):
    command=sub.add_parser("spatial-scene",help="Render geometric moving-source stereo audio")
    command.add_argument("--plan",type=Path,required=True)
    command.add_argument("--source",action="append",required=True,metavar="ID=PATH",
                         help="Explicit local mono WAV binding; repeat for each source")
    command.add_argument("--source-sha256",action="append",default=[],metavar="ID=SHA256")
    command.add_argument("--output",type=Path,required=True)
    command.add_argument("--output-bits",type=int,choices=(8,16,24,32),default=16)
    command.add_argument("--dither-seed",type=int)
    command.add_argument("--packet-frames",type=int,default=8192)
    command.add_argument("--expected-output-sha256")
    command.add_argument("--overwrite",action="store_true")
    command.add_argument("--create-parent",action="store_true")
    return command


def _pairs(items):
    result={}
    if len(items)>32:
        raise ValueError("Too many spatial source bindings")
    for item in items:
        name,separator,value=item.partition("=")
        source_id(name)
        if not separator or not value or name in result:
            raise ValueError("Expected unique ID=VALUE spatial binding")
        result[name]=value
    return result


def run_spatial_command(args):
    try:
        paths=_pairs(args.source)
        pins=_pairs(args.source_sha256)
        if not set(pins)<=set(paths):
            raise ValueError("Source digest names must refer to a bound source")
        if args.plan.resolve()==args.output.resolve():
            raise ValueError("Spatial plan and output must differ")
        if args.output.exists() and os.path.samefile(args.plan,args.output):
            raise ValueError("Spatial plan and output share file identity")
        plan=load_spatial_plan(args.plan)
        bindings={name:SourceBinding(Path(path),pins.get(name)) for name,path in paths.items()}
        result=export_spatial_plan(plan,bindings,args.output,
            output_bits=args.output_bits,dither_seed=args.dither_seed,
            packet_frames=args.packet_frames,expected_output_sha256=args.expected_output_sha256,
            overwrite=args.overwrite,create_parent=args.create_parent)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio spatial scene: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(),sort_keys=True,allow_nan=False))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    register_spatial_command(sub)
    return run_spatial_command(parser.parse_args(argv))


if __name__=="__main__":
    raise SystemExit(main())
