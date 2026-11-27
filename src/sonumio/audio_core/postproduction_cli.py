"""Strict JSON-driven local postproduction command."""
from pathlib import Path
import json
import sys

from .postproduction_io import load_postproduction_plan, export_postproduction


def register_postproduction_command(sub):
    command=sub.add_parser("wav-postproduce",help="Apply a verified JSON duration, loop and mastering plan")
    command.add_argument("input",type=Path)
    command.add_argument("--plan",type=Path,required=True)
    command.add_argument("--output",type=Path,required=True)
    command.add_argument("--output-bits",type=int,choices=(8,16,24,32))
    command.add_argument("--dither-seed",type=int)
    command.add_argument("--encoded-peak-policy",choices=("reject","attenuate"),default="attenuate")
    command.add_argument("--packet-frames",type=int,default=8192)
    command.add_argument("--expected-sha256")
    command.add_argument("--expected-source-pcm-sha256")
    command.add_argument("--expected-output-pcm-sha256")
    command.add_argument("--expected-output-sha256")
    command.add_argument("--overwrite",action="store_true")
    command.add_argument("--create-parent",action="store_true")


def run_postproduction_command(args):
    try:
        plan=load_postproduction_plan(args.plan)
        report=export_postproduction(args.input,args.output,plan=plan,
            output_bits=args.output_bits,dither_seed=args.dither_seed,
            encoded_peak_policy=args.encoded_peak_policy,packet_frames=args.packet_frames,
            expected_source_sha256=args.expected_sha256,
            expected_source_pcm_sha256=args.expected_source_pcm_sha256,
            expected_output_pcm_sha256=args.expected_output_pcm_sha256,
            expected_output_sha256=args.expected_output_sha256,
            overwrite=args.overwrite,create_parent=args.create_parent)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio postproduction: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(),sort_keys=True,allow_nan=False))
    return 0
