"""Apply a verified harmony plan without changing the source score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_project_cli import _distinct, _output
from .music_publish import publish_music_artifact
from .music_reharmonize import read_harmony_plan, reharmonize
from .music_score_io import read_score, encode_score


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("inspect", "apply"):
        command = sub.add_parser(action)
        command.add_argument("score", type=Path)
        command.add_argument("plan", type=Path)
        command.add_argument("--expected-score-sha256")
        command.add_argument("--expected-plan-sha256")
        if action == "apply":
            _output(command)
    args = parser.parse_args(argv)
    try:
        score = read_score(args.score, expected_sha256=args.expected_score_sha256)
        plan = read_harmony_plan(args.plan)
        if args.expected_plan_sha256 is not None and args.expected_plan_sha256 != plan.sha256:
            raise GenerationError("Harmony plan SHA256 mismatch")
        result, report = reharmonize(score, plan)
        if args.action == "apply":
            _distinct((args.score, args.plan), args.output)
            publication = publish_music_artifact(args.output, encode_score(result), kind="score",
                overwrite=args.overwrite, expected_sha256=args.expected_output_sha256)
            report.update(output_sha256=publication.sha256, bytes_written=publication.bytes_written)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio reharmonization: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
