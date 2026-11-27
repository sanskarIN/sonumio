"""Local procedural material sound generation command."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .modal_synthesis import MATERIALS,EXCITATIONS
from .contact_patterns import ContactPattern,pattern_scene
from .contact_io import load_contact_scene,export_contact_scene


def register_contact_command(sub):
    command=sub.add_parser("contact-sound",help="Create original material contact sound effects")
    source=command.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene",type=Path,help="Strict local JSON scene or pattern")
    source.add_argument("--material",choices=MATERIALS)
    command.add_argument("--excitation",choices=EXCITATIONS,default="step")
    command.add_argument("--count",type=int,default=8)
    command.add_argument("--interval-ms",type=int,default=500)
    command.add_argument("--event-ms",type=int,default=250)
    command.add_argument("--timing-jitter-ms",type=int,default=0)
    command.add_argument("--strength",type=float,default=.5)
    command.add_argument("--hardness",type=float,default=.5)
    command.add_argument("--fundamental-hz",type=float,default=240)
    command.add_argument("--size",type=float,default=1)
    command.add_argument("--damping",type=float,default=1)
    command.add_argument("--seed",type=int,default=0)
    command.add_argument("--sample-rate",type=int,default=24000)
    command.add_argument("--output",type=Path,required=True)
    command.add_argument("--output-bits",type=int,choices=(8,16,24,32),default=16)
    command.add_argument("--dither-seed",type=int)
    command.add_argument("--expected-output-sha256")
    command.add_argument("--overwrite",action="store_true")
    command.add_argument("--create-parent",action="store_true")
    return command


def run_contact_command(args):
    try:
        if args.scene is not None:
            if (args.excitation,args.count,args.interval_ms,args.event_ms,args.timing_jitter_ms,
                args.strength,args.hardness,args.fundamental_hz,args.size,args.damping,args.seed,
                args.sample_rate)!=("step",8,500,250,0,.5,.5,240,1,1,0,24000):
                raise ValueError("Pattern flags cannot override a JSON scene")
            if args.scene.resolve()==args.output.resolve():
                raise ValueError("Contact plan and output must differ")
            scene=load_contact_scene(args.scene)
        else:
            scene=pattern_scene(ContactPattern(material=args.material,excitation=args.excitation,
                count=args.count,interval_ms=args.interval_ms,event_ms=args.event_ms,
                timing_jitter_ms=args.timing_jitter_ms,strength=args.strength,
                hardness=args.hardness,fundamental_hz=args.fundamental_hz,size=args.size,
                damping=args.damping,seed=args.seed,sample_rate=args.sample_rate))
        report=export_contact_scene(scene,args.output,output_bits=args.output_bits,
            dither_seed=args.dither_seed,expected_output_sha256=args.expected_output_sha256,
            overwrite=args.overwrite,create_parent=args.create_parent)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio contact sound: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(),sort_keys=True,allow_nan=False))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    register_contact_command(sub)
    return run_contact_command(parser.parse_args(argv))


if __name__=="__main__":
    raise SystemExit(main())
