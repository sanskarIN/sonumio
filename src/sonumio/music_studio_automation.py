"""Strict transport for score-timed automation in the local studio.

No path, executable code, external source, or model selection is accepted.
The same validated automation core is used by the offline CLI and studio.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from .arrangement import Arrangement, TRACKS, arrangement_digest
from .errors import GenerationError
from .music_automation import AutomationPlan, MixKey, MixLane, PARAMETERS, MAX_KEYS
from .music_mix import MixPlan

Track = Literal["keys", "pad", "bass", "lead", "pluck", "kick", "snare", "hat"]
Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")]
MAX_STUDIO_TICKS = 512
MAX_LANES = len(TRACKS) * len(PARAMETERS)


class AutomationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AutomationKeyRequest(AutomationModel):
    tick: int = Field(ge=0, le=MAX_STUDIO_TICKS)
    value: float = Field(ge=-1, le=2, allow_inf_nan=False)
    law: Literal["linear", "smoothstep", "step"] = "linear"

    def to_key(self):
        return MixKey(self.tick, self.value, self.law)


class AutomationLaneRequest(AutomationModel):
    track: Track
    parameter: Literal["gain", "balance"]
    keys: list[AutomationKeyRequest] = Field(min_length=1, max_length=MAX_KEYS)

    def to_lane(self):
        return MixLane(self.track, self.parameter, tuple(key.to_key() for key in self.keys))


class StudioAutomationRequest(AutomationModel):
    lanes: list[AutomationLaneRequest] = Field(default_factory=list, max_length=MAX_LANES)
    expected_automation_sha256: Digest | None = None

    def to_plan(self, score, mix):
        if not isinstance(score, Arrangement) or not isinstance(mix, MixPlan):
            raise GenerationError("Expected validated score and mix for automation")
        # The existing core enforces canonical lane order, key order,
        # independent parameter bounds and the exact score clock.
        plan = AutomationPlan(tuple(lane.to_lane() for lane in self.lanes), mix)
        plan.validate_score(score)
        if self.expected_automation_sha256 is not None and plan.sha256 != self.expected_automation_sha256:
            raise GenerationError("Automation digest mismatch; validate the mix again")
        return plan


def describe_automation(score, plan):
    if not isinstance(plan, AutomationPlan):
        raise GenerationError("Expected AutomationPlan")
    plan.validate_score(score)
    return {
        "score_sha256": arrangement_digest(score),
        "automation_sha256": plan.sha256,
        "lanes": [
            {"track": lane.track, "parameter": lane.parameter,
             "keys": [{"tick": key.tick, "value": float(key.value), "law": key.law,
                       "frame": score.plan.frame(key.tick)} for key in lane.keys]}
            for lane in plan.lanes
        ],
        "frames": score.plan.frame(score.plan.ticks),
        "sample_rate": 24000,
        "mix": {
            "gains": list(plan.mix.gains), "mute": [t for t in TRACKS if t in plan.mix.mute],
            "solo": [t for t in TRACKS if t in plan.mix.solo],
            "room": plan.mix.room, "peak_ceiling": plan.mix.peak_ceiling,
        },
        "keys": sum(len(lane.keys) for lane in plan.lanes),
        "engine": "sonumio-mix-automation-v1",
        "audio_rendered": False,
    }
