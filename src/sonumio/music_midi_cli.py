"""Local MIDI inspection and constrained arrangement import commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from .errors import GenerationError
from .music_midi_read import read_midi,inspect_midi
from .music_midi_import import decode_mapping,import_arrangement,MAX_MAPPING_BYTES
from .music_score_io import read_score,encode_score
from .music_publish import publish_music_artifact


def register_midi_commands(sub):
    inspect=sub.add_parser("music-midi-inspect",help="Inspect a bounded MIDI file and its tempo map")
    inspect.add_argument("midi",type=Path)
    inspect.add_argument("--expected-midi-sha256")
    importer=sub.add_parser("music-midi-import",help="Import mapped MIDI notes using an existing score clock")
    importer.add_argument("midi",type=Path)
    importer.add_argument("--expected-midi-sha256")
    importer.add_argument("--template",type=Path,required=True)
    importer.add_argument("--expected-template-sha256")
    importer.add_argument("--mapping",type=Path)
    importer.add_argument("--output",type=Path,required=True)
    importer.add_argument("--overwrite",action="store_true")
    importer.add_argument("--expected-output-sha256")


def _distinct(inputs,target):
    for source in inputs:
        if source.resolve()==target.resolve() or (target.exists() and os.path.samefile(source,target)):
            raise GenerationError("MIDI import cannot overwrite an input file")


def run_midi_command(args):
    try:
        document=read_midi(args.midi,expected_sha256=args.expected_midi_sha256)
        if args.command=="music-midi-inspect":
            report=inspect_midi(document)
        else:
            sources=(args.midi,args.template)+( (args.mapping,) if args.mapping else ())
            _distinct(sources,args.output)
            template=read_score(args.template,expected_sha256=args.expected_template_sha256)
            mapping=None
            if args.mapping is not None:
                with args.mapping.open("rb") as stream:
                    mapping=decode_mapping(stream.read(MAX_MAPPING_BYTES+1))
            score,report=import_arrangement(document,template,mapping)
            receipt=publish_music_artifact(args.output,encode_score(score),kind="score",
                overwrite=args.overwrite,expected_sha256=args.expected_output_sha256)
            report.update({"output_sha256":receipt.sha256,"bytes_written":receipt.bytes_written})
    except (OSError,ValueError,TypeError,RuntimeError) as exc:
        print(f"Sonumio MIDI interchange: {exc}",file=sys.stderr)
        return 2
    print(json.dumps(report,sort_keys=True,allow_nan=False))
    return 0
