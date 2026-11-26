"""Local controlled-prompt generation with explicit semantic and WAV receipts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio_prompt import parse_prompt,canonical_prompt,build_source
from .music_mix import generate_arrangement_audio
from .nature_scene import generate_scene_audio
from .atomic_cli_wav import publish_generated_wav


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--default-seed",type=int,default=0)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")
    args=parser.parse_args(argv)
    try:
        plan=parse_prompt(args.prompt,default_seed=args.default_seed)
        canonical=canonical_prompt(plan)
        source=build_source(plan)
        if plan.kind=="music":
            data,report=generate_arrangement_audio(source)
        else:
            data,report=generate_scene_audio(source)
        receipt=publish_generated_wav(args.output,data,overwrite=args.overwrite,
                                      expected_sha256=args.expected_output_sha256)
        report.update({"canonical_prompt":canonical,"prompt_kind":plan.kind,
                       "controlled_language":True,"language_model_used":False,
                       "bytes_written":receipt.bytes_written,"output_sha256":receipt.wav_sha256})
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Controlled audio generation failed: {exc}\n")
    print(json.dumps(report,sort_keys=True,ensure_ascii=False,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
