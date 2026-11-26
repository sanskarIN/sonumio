"""Apply a bounded, explicitly ordered JSON edit transaction to a score."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .music_edit import (
    Retime,Transpose,TrackArticulation,RemoveTrack,LeadBar,QuantizeVelocity,
    edit,MAX_EDITS,
)
from .music_score_io import read_score,encode_score
from .music_publish import publish_music_artifact
from .errors import GenerationError

FORMAT="sonumio-score-edits-v1"
MAX_EDIT_BYTES=64*1024
CLASSES={"retime":Retime,"transpose":Transpose,"articulation":TrackArticulation,
         "remove_track":RemoveTrack,"lead_bar":LeadBar,"quantize_velocity":QuantizeVelocity}


def _unique(pairs):
    obj={}
    for key,value in pairs:
        if key in obj:
            raise GenerationError("Duplicate edit document key")
        obj[key]=value
    return obj


def _constant(value):
    raise GenerationError("Nonfinite edit document number")


def decode_edits(data):
    if type(data) is not bytes or not 1<=len(data)<=MAX_EDIT_BYTES:
        raise GenerationError("Invalid edit document byte count")
    try:
        obj=json.loads(data.decode("utf-8"),object_pairs_hook=_unique,parse_constant=_constant)
        if type(obj) is not dict or set(obj)!={"format","operations"} or obj["format"]!=FORMAT:
            raise GenerationError("Unknown edit document format")
        rows=obj["operations"]
        if type(rows) is not list or not 1<=len(rows)<=MAX_EDITS:
            raise GenerationError("Invalid edit operation count")
        result=[]
        for row in rows:
            if type(row) is not dict or set(row)!={"type","parameters"}:
                raise GenerationError("Malformed edit operation")
            kind=row["type"]
            if type(kind) is not str or kind not in CLASSES:
                raise GenerationError("Unknown edit operation")
            cls=CLASSES[kind]
            params=row["parameters"]
            if type(params) is not dict or not set(params)<={f.name for f in fields(cls)}:
                raise GenerationError("Unexpected edit parameter")
            params=dict(params)
            if cls is LeadBar and "notes" in params:
                notes=params["notes"]
                if type(notes) is not list or any(type(n) is not list for n in notes):
                    raise GenerationError("Expected lead note arrays")
                params["notes"]=tuple(tuple(n) for n in notes)
            result.append(cls(**params))
        return tuple(result)
    except (UnicodeError,RecursionError,TypeError,OverflowError,ValueError) as exc:
        if isinstance(exc,GenerationError):
            raise
        raise GenerationError("Malformed score edit document") from exc


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score",type=Path)
    parser.add_argument("edits",type=Path)
    parser.add_argument("--expected-score-sha256")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--expected-output-sha256")
    args=parser.parse_args(argv)
    try:
        score=read_score(args.score,expected_sha256=args.expected_score_sha256)
        with args.edits.open("rb") as stream:
            operations=decode_edits(stream.read(MAX_EDIT_BYTES+1))
        result,report=edit(score,operations)
        receipt=publish_music_artifact(args.output,encode_score(result),kind="score",
                                       overwrite=args.overwrite,
                                       expected_sha256=args.expected_output_sha256)
        report.update({"output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written})
    except (ValueError,RuntimeError,OSError) as exc:
        parser.exit(2,f"Score editing failed: {exc}\n")
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
