"""CLI commands for verified mastering measurement and offline mastering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .interpolated_peak import PeakEstimateSettings
from .gated_energy import GatedEnergySettings
from .mastering import MasteringSettings, MODES
from .mastering_wav import (
    MasteringWavOptions, ENCODED_PEAK_POLICIES,
    inspect_mastering_wav, master_wav_file,
)


def _measurement_options(command):
    command.add_argument("input", type=Path)
    command.add_argument("--expected-sha256")
    command.add_argument("--packet-frames", type=int, default=8192)
    command.add_argument("--peak-factor", type=int, choices=(2,4,8), default=4)
    command.add_argument("--peak-radius", type=int, default=16)
    command.add_argument("--kaiser-beta", type=float, default=8.6)
    command.add_argument("--energy-window-ms", type=int, default=400)
    command.add_argument("--energy-hop-ms", type=int, default=100)
    command.add_argument("--absolute-gate-dbfs", type=float, default=-70)
    command.add_argument("--relative-gate-db", type=float, default=-10)


def register_mastering_commands(sub):
    inspect = sub.add_parser("master-inspect", help="Measure unweighted gated RMS and estimated intersample peak")
    _measurement_options(inspect)
    master = sub.add_parser("wav-master", help="Apply offline gated RMS gain and estimated peak guard")
    _measurement_options(master)
    master.add_argument("--output", type=Path, required=True)
    target = master.add_mutually_exclusive_group()
    target.add_argument("--target-rms-dbfs", type=float, default=-20)
    target.add_argument("--peak-only", action="store_true")
    master.add_argument("--max-boost-db", type=float, default=0)
    master.add_argument("--peak-ceiling-dbfs", type=float, default=-1)
    master.add_argument("--mode", choices=MODES, default="static")
    master.add_argument("--lookahead-ms", type=int, default=5)
    master.add_argument("--release-ms", type=int, default=80)
    master.add_argument("--output-bits", type=int, choices=(8,16,24,32))
    master.add_argument("--dither-seed", type=int)
    master.add_argument("--encoded-peak-policy", choices=ENCODED_PEAK_POLICIES,
                        default="attenuate")
    master.add_argument("--expected-output-sha256")
    master.add_argument("--overwrite", action="store_true")
    master.add_argument("--create-parent", action="store_true")


def _settings(args):
    return (
        PeakEstimateSettings(args.peak_factor,args.peak_radius,args.kaiser_beta),
        GatedEnergySettings(args.energy_window_ms,args.energy_hop_ms,
                            args.absolute_gate_dbfs,args.relative_gate_db),
    )


def run_mastering_command(args):
    try:
        peak, energy = _settings(args)
        if args.command == "master-inspect":
            report = inspect_mastering_wav(
                args.input, peak=peak, energy=energy,
                packet_frames=args.packet_frames,
                expected_sha256=args.expected_sha256,
            )
        else:
            settings = MasteringSettings(
                target_rms_dbfs=None if args.peak_only else args.target_rms_dbfs,
                max_boost_db=args.max_boost_db,
                estimated_peak_ceiling_dbfs=args.peak_ceiling_dbfs,
                mode=args.mode, peak=peak, energy=energy,
                lookahead_ms=args.lookahead_ms, release_ms=args.release_ms,
                packet_frames=args.packet_frames,
            )
            options = MasteringWavOptions(
                mastering=settings, output_bits=args.output_bits,
                dither_seed=args.dither_seed, encoded_peak_policy=args.encoded_peak_policy,
            )
            report = master_wav_file(
                args.input, args.output, options=options,
                expected_sha256=args.expected_sha256,
                expected_output_sha256=args.expected_output_sha256,
                overwrite=args.overwrite, create_parent=args.create_parent,
            )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Sonumio offline mastering: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(), sort_keys=True, allow_nan=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register_mastering_commands(sub)
    return run_mastering_command(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
