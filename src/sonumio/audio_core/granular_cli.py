"""Local granular texture commands; original nature synthesis or verified PCM."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .granular import GranularSettings, BOUNDARIES, WINDOWS, NORMALIZATION
from .granular_wav import (
    GranularExportOptions, ProceduralGrainSource,
    export_granular_wav, export_procedural_grains,
)


def register_granular_command(sub):
    parser=sub.add_parser("granular",help="Generate original granular textures from local PCM or procedural nature")
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input",type=Path)
    source.add_argument("--nature",choices=("rain","wind","ocean","thunder","birds","stream","fire"))
    parser.add_argument("--source-frames",type=int,default=12000)
    parser.add_argument("--source-seed",type=int,default=0)
    parser.add_argument("--source-intensity",type=float,default=.5)
    parser.add_argument("--source-texture",type=float,default=.5)
    parser.add_argument("--source-motion",type=float,default=.5)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frames",type=int,required=True)
    parser.add_argument("--grain-frames",type=int,default=1024)
    parser.add_argument("--hop-frames",type=int,default=256)
    parser.add_argument("--source-start",type=float,default=0)
    parser.add_argument("--source-end",type=float,default=1)
    parser.add_argument("--position-jitter",type=float,default=.05)
    parser.add_argument("--time-jitter-frames",type=int,default=0)
    parser.add_argument("--pitch",type=float,default=0)
    parser.add_argument("--pitch-jitter",type=float,default=0)
    parser.add_argument("--reverse-probability",type=float,default=0)
    parser.add_argument("--amplitude",type=float,default=.5)
    parser.add_argument("--amplitude-jitter",type=float,default=0)
    parser.add_argument("--stereo-spread",type=float,default=0)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--boundary",choices=BOUNDARIES,default="wrap")
    parser.add_argument("--window",choices=WINDOWS,default="hann")
    parser.add_argument("--normalization",choices=NORMALIZATION,default="overlap")
    parser.add_argument("--peak-ceiling",type=float,default=.95)
    parser.add_argument("--output-bits",type=int,choices=(8,16,24,32))
    parser.add_argument("--dither-seed",type=int)
    parser.add_argument("--packet-frames",type=int,default=8192)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-source-pcm-sha256")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--create-parent",action="store_true")
    return parser


def run_granular_command(args):
    try:
        settings=GranularSettings(
            output_frames=args.frames,grain_frames=args.grain_frames,
            hop_frames=args.hop_frames,source_start=args.source_start,
            source_end=args.source_end,position_jitter=args.position_jitter,
            time_jitter_frames=args.time_jitter_frames,pitch_semitones=args.pitch,
            pitch_jitter_semitones=args.pitch_jitter,
            reverse_probability=args.reverse_probability,amplitude=args.amplitude,
            amplitude_jitter=args.amplitude_jitter,stereo_spread=args.stereo_spread,
            seed=args.seed,boundary=args.boundary,window=args.window,
            normalization=args.normalization,peak_ceiling=args.peak_ceiling,
        )
        options=GranularExportOptions(settings,args.output_bits,args.dither_seed,args.packet_frames)
        common=dict(options=options,overwrite=args.overwrite,create_parent=args.create_parent,
                    expected_output_sha256=args.expected_output_sha256)
        if args.input is not None:
            if args.expected_source_pcm_sha256 is not None:
                raise ValueError("The local WAV source uses --expected-sha256")
            if (args.source_frames,args.source_seed,args.source_intensity,args.source_texture,args.source_motion)!=(12000,0,.5,.5,.5):
                raise ValueError("Procedural source options require --nature")
            report=export_granular_wav(args.input,args.output,
                expected_sha256=args.expected_sha256,**common)
        else:
            if args.expected_sha256 is not None:
                raise ValueError("Procedural source uses --expected-source-pcm-sha256")
            source=ProceduralGrainSource(args.nature,args.source_frames,
                args.source_intensity,args.source_texture,args.source_motion,args.source_seed)
            report=export_procedural_grains(source,args.output,
                expected_source_pcm_sha256=args.expected_source_pcm_sha256,**common)
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio granular: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(),sort_keys=True,allow_nan=False))
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    register_granular_command(sub)
    return run_granular_command(parser.parse_args(argv))


if __name__=="__main__":
    raise SystemExit(main())
