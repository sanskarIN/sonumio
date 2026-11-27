"""Bounded, local-first signal processing primitives.

The numerical core is independent of neural model training. Explicit
procedural generation helpers reuse Sonumio's original local sources. Audio buffers are channel-first;
no model weights, network calls, user audio uploads or implicit file writes.
The existing API continues to use its historical renderer.
"""

from .errors import AudioContractError, AudioFormatError, AudioLimitError
from .contracts import (
    AudioBuffer, AudioLayout, AudioSpec, as_channel_first, check_audio,
    check_rate, finite_float, make_buffer, validate_allocation,
)
from .timebase import (
    FrameInterval, FramePosition, TimeGrid, frames_from_seconds,
    seconds_from_frames, split_intervals,
)
from .numerics import (
    Decibel, db_to_gain, gain_to_db, dbfs, linear_ramp, equal_power_ramps,
    stable_rms, stable_peak, finite_clamp, saturate, mix_accumulate,
)
from .hashing import (
    digest_audio, digest_file, canonical_json_bytes, checksum_record,
    verify_record, strict_sha256,
)
from .windows import (
    hann, hamming, blackman, tukey, sqrt_hann, envelope, window_bank,
    overlap_envelope, validate_cola, weighted_overlap_add,
)


from .wave_stream import WavPacketSpec, WavPacketResult, WavPacketWriter
from .stream_resample import StreamingSincResampler
from .wave_resample import WavResampleResult, resample_wav_file
from .stream_dither import StreamDitherQuantizer, STREAM_TPDF_VERSION
from .wave_reader import WavPacketReader, WavPacketInfo
from .wave_pipeline import WavProcessingConfig, WavProcessingResult, process_wav_file
from .stream_quality import StreamingQualityAccumulator, VerifiedWavQuality, inspect_wav_quality, PcmRailReport
from .quality_timeline import QualityWindow, QualityTimeline, StreamingQualityTimeline
from .frame_transitions import FrameTransitionReport, StreamingFrameTransitionAccumulator
from .quiet_intervals import QuietIntervalReport, StreamingQuietIntervalAccumulator
from .channel_relationships import (
    ChannelPairReport, ChannelRelationshipReport, StreamingChannelRelationshipAccumulator,
)

from .streaming import (
    FramePacket, FramerConfig, PacketCollector, StreamingFramer,
    WindowPacket, split_packets, offline_window_reference,
)
from .quality import (
    QualityFinding, QualityLimits, QualityReport, measure_quality,
    waveform_integrity_report, compare_quality_reports,
)
from .spectral_effects import (
    SpectralGateOptions, SpectralBand, spectral_gate,
    spectral_eq, spectral_tilt, heuristic_harmonic_percussive,
)

from .time_scale import TimeScaleSettings, TimeScaleResult, TimeScaleReport, stretch_audio, target_frames_for_ratio
from .pitch_shift import PitchShiftSettings, PitchShiftResult, PitchShiftReport, pitch_shift
from .wav_transform import WavTransformConfig, WavTransformReport, transform_wav_file

from .loop_analysis import LoopSearchPlan, LoopSearchResult, LoopCandidate, LoopScoreWeights, search_loop, measure_loop_seam
from .loop_render import LoopRenderPlan, LoopRenderResult, LoopRenderReport, render_loop, circular_diagnostics
from .wave_snapshot import VerifiedPcmSnapshot, read_verified_pcm
from .interpolated_peak import PeakEstimateSettings, InterpolatedPeakAccumulator, InterpolatedPeakReport, estimate_interpolated_peak
from .gated_energy import GatedEnergySettings, GatedEnergyAccumulator, GatedEnergyReport, measure_gated_energy
from .lookahead_limiter import LookaheadSettings, LimiterResult, LimiterReport, limit_audio
from .mastering import MasteringSettings, MasteringResult, MasteringReport, SignalMeasurement, MasteringMeasurement, measure_mastering, master_audio
from .mastering_wav import MasteringWavOptions, VerifiedMasteringInspection, VerifiedMasteringExport, inspect_mastering_wav, master_wav_file

from .postproduction import PostProductionPlan, PostProductionResult, run_postproduction, verify_postproduction_manifest
from .postproduction_io import PostProductionExport, parse_postproduction_plan, export_postproduction

from .granular import GranularSettings, GrainEvent, GranularResult, GranularReport, grain_schedule, synthesize_grains
from .granular_wav import GranularExportOptions, ProceduralGrainSource, GranularExport, export_granular_wav, export_procedural_grains

from .modal_synthesis import ResonanceMode, Resonator, ContactEvent, ContactScene, ContactSceneResult, ContactSceneReport, material_resonator, render_contact_scene
from .contact_patterns import ContactPattern, pattern_scene
from .contact_io import ContactExport, parse_contact_scene, export_contact_scene
from .spatial_motion import Position, MotionKey, Trajectory, Room, Receiver, MovingSource, SpatialScene, SpatialResult, SpatialReport, render_spatial_scene
from .spatial_io import SpatialPlacement, SpatialPlan, SourceBinding, SpatialExport, parse_spatial_plan, export_spatial_plan
from .asset_pack import AssetRecipe, PackQuality, SoundPackPlan, AssetRecord, plan_jobs, parse_pack_plan
from .asset_archive import VerifiedPack, PublishedPack, create_pack_bytes, verify_pack_bytes, verify_pack_file, export_sound_pack
from .asset_selection import SelectedAsset, select_asset, list_pack_assets, export_pack_asset

__all__ = [
    "ResonanceMode", "Resonator", "ContactEvent", "ContactScene", "ContactSceneResult", "ContactSceneReport",
    "material_resonator", "render_contact_scene", "ContactPattern", "pattern_scene",
    "ContactExport", "parse_contact_scene", "export_contact_scene",
    "Position", "MotionKey", "Trajectory", "Room", "Receiver", "MovingSource", "SpatialScene", "SpatialResult", "SpatialReport", "render_spatial_scene",
    "SpatialPlacement", "SpatialPlan", "SourceBinding", "SpatialExport", "parse_spatial_plan", "export_spatial_plan",
    "AssetRecipe", "PackQuality", "SoundPackPlan", "AssetRecord", "plan_jobs", "parse_pack_plan",
    "VerifiedPack", "PublishedPack", "create_pack_bytes", "verify_pack_bytes", "verify_pack_file", "export_sound_pack",
    "SelectedAsset", "select_asset", "list_pack_assets", "export_pack_asset",
    "GranularSettings", "GrainEvent", "GranularResult", "GranularReport", "grain_schedule", "synthesize_grains",
    "GranularExportOptions", "ProceduralGrainSource", "GranularExport", "export_granular_wav", "export_procedural_grains",
    "PostProductionPlan", "PostProductionResult", "run_postproduction", "verify_postproduction_manifest",
    "PostProductionExport", "parse_postproduction_plan", "export_postproduction",
    "LoopSearchPlan", "LoopSearchResult", "LoopCandidate", "LoopScoreWeights", "search_loop", "measure_loop_seam",
    "LoopRenderPlan", "LoopRenderResult", "LoopRenderReport", "render_loop", "circular_diagnostics",
    "VerifiedPcmSnapshot", "read_verified_pcm",
    "PeakEstimateSettings", "InterpolatedPeakAccumulator", "InterpolatedPeakReport", "estimate_interpolated_peak",
    "GatedEnergySettings", "GatedEnergyAccumulator", "GatedEnergyReport", "measure_gated_energy",
    "LookaheadSettings", "LimiterResult", "LimiterReport", "limit_audio",
    "MasteringSettings", "MasteringResult", "MasteringReport", "SignalMeasurement", "MasteringMeasurement", "measure_mastering", "master_audio",
    "MasteringWavOptions", "VerifiedMasteringInspection", "VerifiedMasteringExport", "inspect_mastering_wav", "master_wav_file",
    "TimeScaleSettings", "TimeScaleResult", "TimeScaleReport", "stretch_audio", "target_frames_for_ratio",
    "PitchShiftSettings", "PitchShiftResult", "PitchShiftReport", "pitch_shift",
    "WavTransformConfig", "WavTransformReport", "transform_wav_file",
    "WavPacketSpec", "WavPacketResult", "WavPacketWriter",
    "StreamingSincResampler", "WavResampleResult", "resample_wav_file",
    "StreamDitherQuantizer", "STREAM_TPDF_VERSION",
    "WavPacketReader", "WavPacketInfo",
    "WavProcessingConfig", "WavProcessingResult", "process_wav_file",
    "StreamingQualityAccumulator", "VerifiedWavQuality", "inspect_wav_quality",
    "PcmRailReport",
    "QualityWindow", "QualityTimeline", "StreamingQualityTimeline",
    "FrameTransitionReport", "StreamingFrameTransitionAccumulator",
    "QuietIntervalReport", "StreamingQuietIntervalAccumulator",
    "ChannelPairReport", "ChannelRelationshipReport", "StreamingChannelRelationshipAccumulator",
    "FramePacket",
    "FramerConfig",
    "PacketCollector",
    "StreamingFramer",
    "WindowPacket",
    "split_packets",
    "offline_window_reference",
    "QualityFinding",
    "QualityLimits",
    "QualityReport",
    "measure_quality",
    "waveform_integrity_report",
    "compare_quality_reports",
    "SpectralGateOptions",
    "SpectralBand",
    "spectral_gate",
    "spectral_eq",
    "spectral_tilt",
    "heuristic_harmonic_percussive",
    "AudioContractError", "AudioFormatError", "AudioLimitError",
    "AudioBuffer", "AudioLayout", "AudioSpec", "as_channel_first",
    "check_audio", "check_rate", "finite_float", "make_buffer",
    "validate_allocation", "FrameInterval", "FramePosition", "TimeGrid",
    "frames_from_seconds", "seconds_from_frames", "split_intervals",
    "Decibel", "db_to_gain", "gain_to_db", "dbfs", "linear_ramp",
    "equal_power_ramps", "stable_rms", "stable_peak", "finite_clamp",
    "saturate", "mix_accumulate", "digest_audio", "digest_file",
    "canonical_json_bytes", "checksum_record", "verify_record",
    "strict_sha256", "hann", "hamming", "blackman", "tukey",
    "sqrt_hann", "envelope", "window_bank", "overlap_envelope",
    "validate_cola", "weighted_overlap_add",
]
