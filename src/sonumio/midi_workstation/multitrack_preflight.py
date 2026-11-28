"""No-audio preflight for deterministic multi-instrument workstation previews.

Preflight returns an inspectable schedule and resource ledger BEFORE importing
Sonumio's audio engine. The same validated plan drives actual synthesis, so
CLI estimates cannot silently diverge from the production rendering path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .contracts import MidiLimitError, MidiWorkstationError
from .score import Score
from .transport import Schedule, ScheduledNote, TransportSettings, build_schedule

if TYPE_CHECKING:
    from .multitrack_preview import MultiTrackPreviewSettings, TrackPatch

PROTOCOL = "sonumio-multitrack-preflight-v1"
MAX_ACTIVE_VOICE_FRAMES = 40_000_000
MAX_POLYPHONY = 64


@dataclass(frozen=True, slots=True)
class InstrumentPlan:
    track_id: str
    waveform: str
    notes: int
    gate_frames: int
    release_voice_frames: int
    maximum_polyphony: int


@dataclass(frozen=True, slots=True)
class PreviewPreflight:
    source_score_sha256: str
    schedule: Schedule
    selected_notes: tuple[ScheduledNote, ...]
    skipped_percussion_notes: int
    maximum_polyphony: int
    active_voice_frames: int
    float_workspace_bytes: int
    work_samples: int
    stems: tuple[InstrumentPlan, ...]

    @property
    def output_frames(self) -> int:
        return self.schedule.output_frames

    @property
    def rendered_notes(self) -> int:
        return len(self.selected_notes)

    def as_dict(self) -> dict:
        return {
            "protocol": PROTOCOL,
            "source_score_sha256": self.source_score_sha256,
            "sample_rate": self.schedule.settings.sample_rate,
            "output_frames": self.output_frames,
            "timeline_end_frame": self.schedule.timeline_end_frame,
            "collision_adjustments": self.schedule.collision_adjustments,
            "rendered_notes": self.rendered_notes,
            "skipped_percussion_notes": self.skipped_percussion_notes,
            "maximum_polyphony": self.maximum_polyphony,
            "active_voice_frames": self.active_voice_frames,
            "float_workspace_bytes": self.float_workspace_bytes,
            "work_samples": self.work_samples,
            "stems": [asdict(s) for s in self.stems],
            "audio_rendered": False,
            "network_used": False,
            "trained_model_inference": False,
            "source_score_modified": False,
        }


def _peak_polyphony(schedule: Schedule, patch_map: dict) -> int:
    """Stable half-open voice count including each patch's release tail."""
    markers = []
    for note in schedule.notes:
        patch = patch_map.get(note.track_id)
        if patch is None:
            continue
        markers.append((note.start_frame, 1))
        markers.append((note.stop_frame + patch.release_frames, -1))
    count = maximum = 0
    for _, delta in sorted(markers):
        count += delta
        if count < 0:
            raise MidiWorkstationError("Invalid synthesized note interval order")
        maximum = max(count, maximum)
        if maximum > MAX_POLYPHONY:
            raise MidiLimitError("Multitrack mix exceeds 64 simultaneous voices")
    if count:
        raise MidiWorkstationError("Unbalanced multitrack voice intervals")
    return maximum


def _instrument_polyphony(notes: tuple[ScheduledNote,...], release_frames: int) -> int:
    markers=[]
    for note in notes:
        markers.append((note.start_frame,1))
        markers.append((note.stop_frame+release_frames,-1))
    active=maximum=0
    for _,delta in sorted(markers):
        active += delta
        maximum=max(maximum, active)
    return maximum


def build_multitrack_preflight(
    score: Score, settings: "MultiTrackPreviewSettings",
) -> PreviewPreflight:
    """Preflight identical note selection and work budgets used by the renderer."""
    # Import the settings type on demand to avoid a dependency cycle: render
    # and CLI can both import this module without importing the audio engine.
    from .multitrack_preview import (
        MultiTrackPreviewSettings, MAX_MIX_FLOAT_BYTES,
    )
    if not isinstance(score, Score) or not isinstance(settings, MultiTrackPreviewSettings):
        raise MidiWorkstationError("Expected workstation Score and MultiTrackPreviewSettings")
    patches = {patch.track_id: patch for patch in settings.patches}
    selected_ids = (
        {track.id for track in score.tracks} if settings.include_muted
        else set(score.active_track_ids)
    )
    selected_pitched = {
        track.id for track in score.tracks
        if track.id in selected_ids and track.pitched and track.notes
    }
    if set(patches) != selected_pitched:
        raise MidiWorkstationError(
            "Instrument patches must cover exactly audible pitched tracks"
        )
    tail = max(patch.release_frames for patch in settings.patches)
    schedule = build_schedule(score, TransportSettings(
        sample_rate=settings.sample_rate,
        tail_frames=tail,
        include_muted=settings.include_muted,
    ))
    if schedule.output_frames > 120 * settings.sample_rate:
        raise MidiLimitError("Multitrack preview exceeds synthesizer 120-second budget")
    percussion = tuple(note for note in schedule.notes
                       if note.track_id not in selected_pitched)
    if percussion and not settings.allow_skip_percussion:
        raise MidiWorkstationError(
            "Percussion requires explicit skip approval or another renderer"
        )
    selected = tuple(note for note in schedule.notes
                     if note.track_id in selected_pitched)
    if len(selected) > settings.maximum_notes:
        raise MidiLimitError("Multitrack preview exceeds note count budget")
    if settings.channels == 1 and any(note.pan != 64 for note in selected):
        raise MidiWorkstationError(
            "Mono multi-instrument output requires centered note pan"
        )
    float_workspace_bytes = schedule.output_frames * settings.channels * (8+4+4)
    if float_workspace_bytes > MAX_MIX_FLOAT_BYTES:
        raise MidiLimitError("Multitrack mix exceeds float PCM memory budget")
    work_samples = schedule.output_frames * settings.channels * len(patches)
    if work_samples > settings.maximum_work_samples:
        raise MidiLimitError("Multitrack mix exceeds total track work budget")
    active_voice_frames = sum(
        note.frames + patches[note.track_id].release_frames
        for note in selected
    )
    if active_voice_frames > MAX_ACTIVE_VOICE_FRAMES:
        raise MidiLimitError("Multitrack mix exceeds total voice processing budget")
    polyphony = _peak_polyphony(schedule, patches)
    stems=[]
    for patch in settings.patches:
        notes = tuple(n for n in selected if n.track_id == patch.track_id)
        stems.append(InstrumentPlan(
            patch.track_id, patch.waveform,
            len(notes),
            sum(n.frames for n in notes),
            sum(n.frames + patch.release_frames for n in notes),
            _instrument_polyphony(notes,patch.release_frames),
        ))
    return PreviewPreflight(
        score.sha256, schedule, selected, len(percussion), polyphony,
        active_voice_frames, float_workspace_bytes, work_samples, tuple(stems),
    )
