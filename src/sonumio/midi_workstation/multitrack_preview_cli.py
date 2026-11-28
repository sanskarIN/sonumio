"""Local multi-instrument score-preview WAV command registration."""
from __future__ import annotations

from argparse import _SubParsersAction
from pathlib import Path

from .contracts import MidiWorkstationError
from .score_io import load_score
from .multitrack_preview import (
    TrackPatch,MultiTrackPreviewSettings,export_multitrack_preview,
)

PROTOCOL="sonumio-workstation-multitrack-cli-v1"


def _patch(text: str) -> TrackPatch:
    if type(text) is not str:
        raise MidiWorkstationError("Instrument patch must be text")
    fields=text.split(":")
    if len(fields)!=3 or not all(fields):
        raise MidiWorkstationError("Expected track-id:waveform:gain instrument patch")
    try:
        gain=float(fields[2])
    except ValueError as exc:
        raise MidiWorkstationError("Instrument gain must be finite numeric text") from exc
    return TrackPatch(fields[0],fields[1],gain)


def register_multitrack_preview(sub: _SubParsersAction) -> None:
    parser=sub.add_parser(
        "multitrack-preview-wav",
        help="Render explicit per-track mathematical instrument patches to verified WAV",
    )
    parser.add_argument("source",type=Path)
    parser.add_argument("--patch",action="append",required=True,
                        help="Explicit track-id:waveform:gain; repeat for pitched tracks")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--sample-rate",type=int,default=24000)
    parser.add_argument("--channels",type=int,default=2)
    parser.add_argument("--allow-skip-percussion",action="store_true")
    parser.add_argument("--include-muted",action="store_true")
    parser.add_argument("--normalization",choices=("reject","peak"),default="reject")
    parser.add_argument("--max-peak",type=float,default=.98)
    parser.add_argument("--maximum-notes",type=int,default=2048)
    parser.add_argument("--maximum-work-samples",type=int,default=14_000_000)
    parser.add_argument("--output-bits",type=int,choices=(8,16,24,32),default=16)
    parser.add_argument("--dither-seed",type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-output-sha256")
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--create-parent",action="store_true")


def run_multitrack_preview(args) -> dict:
    source=load_score(args.source,expected_sha256=args.expected_sha256)
    settings=MultiTrackPreviewSettings(
        patches=tuple(sorted((_patch(entry) for entry in args.patch),
                             key=lambda item:item.track_id)),
        sample_rate=args.sample_rate,channels=args.channels,
        include_muted=args.include_muted,
        allow_skip_percussion=args.allow_skip_percussion,
        normalization=args.normalization,maximum_peak=args.max_peak,
        maximum_notes=args.maximum_notes,
        maximum_work_samples=args.maximum_work_samples,
    )
    receipt=export_multitrack_preview(
        source.score,args.output,settings,source_path=args.source,
        output_bits=args.output_bits,dither_seed=args.dither_seed,
        overwrite=args.overwrite,create_parent=args.create_parent,
        expected_output_sha256=args.expected_output_sha256,
    )
    return {"protocol":PROTOCOL,"source":source.as_dict(),"preview":receipt}


COMMANDS={"multitrack-preview-wav":run_multitrack_preview}
