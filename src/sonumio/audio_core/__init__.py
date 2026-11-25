"""Bounded, local-first signal processing primitives.

This package is deliberately independent of the original procedural
generator and all neural model training. Audio buffers are channel-first;
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

__all__ = [
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
