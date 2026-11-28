"""Local creation, verification and selection of original music stem packages."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .arrangement import TRACKS
from .atomic_cli_wav import publish_generated_wav
from .errors import GenerationError
from .music_score_io import read_score
from .music_stem_pack import create_stem_pack, publish_stem_pack, read_stem_pack, select_stem_file


def _output(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256")


def register_stem_commands(sub):
    create = sub.add_parser("music-stems", help="Export eight dry music tracks in a verified ZIP")
    create.add_argument("score", type=Path)
    create.add_argument("--expected-score-sha256")
    _output(create)
    verify = sub.add_parser("music-stems-verify", help="Verify all score and PCM stem bytes")
    verify.add_argument("archive", type=Path)
    verify.add_argument("--expected-pack-sha256")
    verify.add_argument("--replay", action="store_true",
                        help="Repeat the original synthesis and compare exact bytes")
    select = sub.add_parser("music-stem-select", help="Publish one verified stereo dry track")
    select.add_argument("archive", type=Path)
    select.add_argument("--track", choices=TRACKS, required=True)
    select.add_argument("--expected-pack-sha256")
    _output(select)


def _separate(source, output):
    if source.resolve() == output.resolve():
        raise GenerationError("Music stem input and output must differ")
    if output.exists() and os.path.samefile(source, output):
        raise GenerationError("Music stem input and output share file identity")


def run_stem_command(args):
    try:
        if args.command == "music-stems":
            _separate(args.score, args.output)
            score = read_score(args.score, expected_sha256=args.expected_score_sha256)
            raw, _ = create_stem_pack(score)
            report = publish_stem_pack(args.output, raw, overwrite=args.overwrite,
                                       expected_sha256=args.expected_output_sha256)
        elif args.command == "music-stems-verify":
            report = read_stem_pack(args.archive, expected_sha256=args.expected_pack_sha256,
                                    replay=args.replay).as_dict()
        elif args.command == "music-stem-select":
            _separate(args.archive, args.output)
            wav, pack = select_stem_file(args.archive, args.track,
                                         expected_sha256=args.expected_pack_sha256)
            receipt = publish_generated_wav(args.output, wav, overwrite=args.overwrite,
                                            expected_sha256=args.expected_output_sha256)
            report = {"archive_sha256": pack.archive_sha256, "track": args.track,
                      "score_sha256": pack.manifest["score_sha256"],
                      "output_sha256": receipt.wav_sha256,
                      "bytes_written": receipt.bytes_written}
        else:
            raise GenerationError("Unknown music stem command")
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        print(f"Sonumio music stems: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    register_stem_commands(parser.add_subparsers(dest="command", required=True))
    return run_stem_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
