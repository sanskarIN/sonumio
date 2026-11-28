"""Score-clock automation over bounded stem blocks and verified replay."""
from __future__ import annotations

import numpy as np

from .arrangement import TRACKS, arrangement_digest
from .errors import GenerationError
from .music_automation import AutomationPlan, compile_lanes, PROTOCOL as AUTOMATION_PROTOCOL
from .music_blockwise import (
    DEFAULT_BLOCK, _validate, _Room, _Edges, _Stats, _encode_replayed_blocks,
    iter_stem_blocks,
)

PROTOCOL="sonumio-blockwise-automation-v1"


def _automate(stem,start,level,gain,balance):
    block=stem.astype(np.float64)
    block*=level
    count=block.shape[1]
    if gain is not None:
        block*=gain.render(start,count)[None,:]
    if balance is not None:
        curve=balance.render(start,count)
        block[0]*=1.-np.maximum(curve,0.)
        block[1]*=1.+np.minimum(curve,0.)
    return block


def iter_automated_wet_blocks(score,plan,block_frames=DEFAULT_BLOCK,*,track_stats=None):
    total=_validate(score,block_frames)
    if not isinstance(plan,AutomationPlan):
        raise GenerationError("Expected AutomationPlan for blockwise rendering")
    lanes=compile_lanes(score,plan)
    room=_Room(plan.mix.room,total)
    edges=_Edges(total)
    for start,stems in iter_stem_blocks(score,block_frames):
        count=stems[TRACKS[0]].shape[1]
        dry=np.zeros((2,count),dtype=np.float64)
        for track,level in zip(TRACKS,plan.mix.gains,strict=True):
            if not plan.mix.active(track):
                continue
            block=_automate(stems[track],start,level,lanes.get((track,"gain")),
                            lanes.get((track,"balance")))
            if track_stats is not None:
                track_stats[track].add(block)
            dry+=block
        wet=room.apply(start,dry)
        edges.apply(start,wet)
        if not np.isfinite(wet).all():
            raise GenerationError("Nonfinite blockwise automated mix")
        yield start,wet
    if room.cursor!=total:
        raise GenerationError("Incomplete automated room traversal")


def generate_blockwise_automated_music(score,plan,*,block_frames=DEFAULT_BLOCK):
    """Render automation with no full-song float arrays.

    Automation lanes are compiled against the exact arrangement clock.
    The encoder verifies replay of every unscaled float64 stereo block
    and applies a single global attenuation before PCM16 encoding.
    """
    total=_validate(score,block_frames)
    if not isinstance(plan,AutomationPlan):
        raise GenerationError("Expected AutomationPlan")
    lanes=compile_lanes(score,plan)
    tracks={name:_Stats() for name in TRACKS}

    def blocks(first):
        return iter_automated_wet_blocks(score,plan,block_frames,
                                        track_stats=tracks if first else None)

    wav,report=_encode_replayed_blocks(total,plan.mix.peak_ceiling,blocks)
    report.update({"protocol":AUTOMATION_PROTOCOL,"renderer_protocol":PROTOCOL,
                   "score_sha256":arrangement_digest(score),
                   "automation_sha256":plan.sha256,"automation_lanes":len(lanes),
                   "room_amount":float(plan.mix.room),"block_frames":block_frames,
                   "tracks":{track:{
                       "automated_peak":tracks[track].peak,
                       "automated_rms":tracks[track].rms if tracks[track].samples else 0.,
                       "active":plan.mix.active(track),"static_gain":float(level),
                       "gain_automated":(track,"gain") in lanes,
                       "balance_automated":(track,"balance") in lanes,
                   } for track,level in zip(TRACKS,plan.mix.gains,strict=True)}})
    return wav,report
