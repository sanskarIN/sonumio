"""Inspect, reconcile and replay direct sibling music-project revisions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_project import read_project, encode_project
from .music_project_cli import _distinct, _project, _output
from .music_project_merge import describe_project_changes, merge_projects, verify_merge_receipt
from .music_publish import publish_music_artifact
from .music_score_io import _unique, _nonfinite

MAX_RECEIPT_BYTES = 128 * 1024


def _inputs(parser):
    for name in ("base", "left", "right"):
        _project(parser, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    diff = sub.add_parser("diff", help="Inspect semantic project changes")
    _project(diff, "source")
    _project(diff, "target")
    preview = sub.add_parser("preview", help="Inspect a sibling merge without writing files")
    _inputs(preview)
    merge = sub.add_parser("merge", help="Publish a reconciled project if conflict-free")
    _inputs(merge)
    _output(merge)
    verify = sub.add_parser("verify", help="Replay a JSON receipt against project files")
    _inputs(verify)
    _project(verify, "result")
    verify.add_argument("receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "diff":
            source = read_project(args.source, expected_sha256=args.expected_source_sha256)
            target = read_project(args.target, expected_sha256=args.expected_target_sha256)
            report = describe_project_changes(source, target)
        else:
            base, left, right = (
                read_project(getattr(args, name), expected_sha256=getattr(args, "expected_" + name + "_sha256"))
                for name in ("base", "left", "right")
            )
            if args.command == "verify":
                result = read_project(args.result, expected_sha256=args.expected_result_sha256)
                with args.receipt.open("rb") as stream:
                    data = stream.read(MAX_RECEIPT_BYTES + 1)
                if not 1 <= len(data) <= MAX_RECEIPT_BYTES:
                    raise GenerationError("Invalid merge receipt byte count")
                receipt = json.loads(data.decode("utf-8"), object_pairs_hook=_unique,
                                     parse_constant=_nonfinite)
                # Publication metadata is outside the canonical replay receipt.
                if type(receipt) is dict and "publication" in receipt:
                    receipt = {k: v for k, v in receipt.items() if k != "publication"}
                report = verify_merge_receipt(base, left, right, result, receipt)
            else:
                result, report = merge_projects(base, left, right)
                if result is None:
                    print(json.dumps(report, sort_keys=True, allow_nan=False))
                    return 3
                if args.command == "merge":
                    _distinct((args.base, args.left, args.right), args.output)
                    publication = publish_music_artifact(args.output, encode_project(result), kind="project",
                                                         overwrite=args.overwrite,
                                                         expected_sha256=args.expected_output_sha256)
                    report["publication"] = {"sha256": publication.sha256,
                                             "bytes_written": publication.bytes_written}
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, RecursionError) as exc:
        print(f"Sonumio project merge: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
