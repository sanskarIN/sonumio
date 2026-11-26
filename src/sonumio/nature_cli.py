"""Local controllable nature-sound generation with verified WAV export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nature_scene import preset,generate_scene_audio
from .nature_scene_io import encode_scene,read_scene
from .music_publish import publish_music_artifact
from .atomic_cli_wav import publish_generated_wav

PRESETS=("forest","storm","coast","creek","campfire")


def _preset(parser):
    parser.add_argument("preset",choices=PRESETS)
    parser.add_argument("--seconds",type=float,default=8.)
    parser.add_argument("--seed",type=int,default=0)


def _output(parser):
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    create=sub.add_parser("generate",help="Generate a stereo nature preset")
    _preset(create)
    _output(create)
    plan=sub.add_parser("plan",help="Write editable preset scene JSON")
    _preset(plan)
    _output(plan)
    render=sub.add_parser("render",help="Render a verified custom nature scene")
    render.add_argument("scene",type=Path)
    render.add_argument("--expected-scene-sha256")
    _output(render)
    args=parser.parse_args(argv)
    try:
        if args.command=="render":
            scene=read_scene(args.scene,expected_sha256=args.expected_scene_sha256)
        else:
            scene=preset(args.preset,args.seconds,args.seed)
        if args.command=="plan":
            receipt=publish_music_artifact(args.output,encode_scene(scene),kind="score",
                                           overwrite=args.overwrite,
                                           expected_sha256=args.expected_output_sha256)
            from .nature_scene import scene_digest
            report={"scene_sha256":scene_digest(scene),"output_sha256":receipt.sha256,
                    "bytes_written":receipt.bytes_written}
        else:
            data,report=generate_scene_audio(scene)
            receipt=publish_generated_wav(args.output,data,overwrite=args.overwrite,
                                          expected_sha256=args.expected_output_sha256)
            report.update({"output_sha256":receipt.wav_sha256,"bytes_written":receipt.bytes_written})
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f"Nature scene failed: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
