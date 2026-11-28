"""Inspect or compare canonical score-clock stereo WAV files locally."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from .errors import GenerationError
from .music_score_io import read_score
from .music_render_analysis import AnalysisSettings, read_wav, analyze_render
from .music_render_compare import compare_renders


def _read(path,expected):
    data=read_wav(path)
    if expected is not None and hashlib.sha256(data).hexdigest()!=expected:
        raise GenerationError("Analysis WAV SHA256 mismatch")
    return data


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    for action in ("inspect","compare"):
        command=sub.add_parser(action)
        command.add_argument("score",type=Path)
        command.add_argument("source",type=Path)
        command.add_argument("--expected-score-sha256")
        command.add_argument("--expected-source-sha256")
        if action=="compare":
            command.add_argument("candidate",type=Path)
            command.add_argument("--expected-candidate-sha256")
        command.add_argument("--quiet-dbfs",type=float,default=-70.)
        command.add_argument("--minimum-active-fraction",type=float,default=.001)
        command.add_argument("--maximum-dc",type=float,default=.02)
        command.add_argument("--maximum-boundary-step",type=float,default=.25)
        command.add_argument("--minimum-mono-retention",type=float,default=.2)
    args=parser.parse_args(argv)
    try:
        score=read_score(args.score,expected_sha256=args.expected_score_sha256)
        source=_read(args.source,args.expected_source_sha256)
        settings=AnalysisSettings(args.quiet_dbfs,args.minimum_active_fraction,args.maximum_dc,
                                  args.maximum_boundary_step,args.minimum_mono_retention)
        if args.action=="inspect":
            report=analyze_render(score,source,settings)
        else:
            candidate=_read(args.candidate,args.expected_candidate_sha256)
            report=compare_renders(score,source,candidate,settings)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        return 0
    except (ValueError,OSError,TypeError,RuntimeError) as exc:
        print(f"Sonumio render analysis: {exc}",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
