"""Frame-exact time calculations for clips, transitions and alignments.

A source clock is integer Hz. Frame positions should remain integers through
rendering; float seconds are accepted only at the boundary and rounded using
a documented round-half-to-even policy (Python's round behavior).
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
from typing import Iterator

from .contracts import check_rate, MAX_SECONDS, finite_float
from .errors import AudioContractError, AudioLimitError

# Materialized partition tuples are bounded independently of total PCM time.
# split_intervals() remains lazy for callers that stream many intervals.
MAX_PARTITION_INTERVALS = 50_000


def _decimal_seconds(seconds: int | float | Decimal) -> Decimal:
    if type(seconds) is bool:
        raise AudioContractError("Time cannot be boolean")
    if isinstance(seconds, Decimal):
        result = seconds
    elif type(seconds) in (float, int):
        if not math.isfinite(seconds):
            raise AudioContractError("Time must be finite")
        result = Decimal(str(seconds))
    else:
        raise AudioContractError("Time must be finite real seconds")
    if not result.is_finite():
        raise AudioContractError("Time must be finite")
    return result


def frames_from_seconds(seconds: int | float | Decimal | Fraction, rate: int,
                        *, allow_zero: bool = True,
                        max_seconds: float = MAX_SECONDS) -> int:
    check_rate(rate)
    finite_float(max_seconds, "max_seconds", minimum=0.01, maximum=86400)
    if isinstance(seconds, Fraction):
        ratio = seconds
    else:
        decimal = _decimal_seconds(seconds)
        if decimal < 0 or decimal > Decimal(str(max_seconds)):
            raise AudioLimitError("Requested time exceeds bounded frame clock")
        # Decimal multiplication also obeys the ambient 28-digit context:
        # values extremely close to half frames can be rounded *before*
        # integer quantization. Convert the exact finite decimal to a ratio.
        parts = decimal.as_tuple()
        if len(parts.digits) > 1200 or abs(parts.exponent) > 1024:
            raise AudioLimitError("Timestamp precision exceeds budget")
        ratio = Fraction(decimal)
    if (ratio.numerator.bit_length() > 4096 or
            ratio.denominator.bit_length() > 4096):
        raise AudioLimitError("Fractional timestamp precision exceeds budget")
    if ratio < 0 or ratio > Fraction(Decimal(str(max_seconds))):
        raise AudioLimitError("Requested time exceeds bounded frame clock")
    quotient, remainder = divmod(ratio.numerator * rate, ratio.denominator)
    doubled = remainder * 2
    count = quotient + int(doubled > ratio.denominator or
                           (doubled == ratio.denominator and quotient % 2 == 1))
    if count == 0 and not allow_zero:
        raise AudioContractError("Time rounded to zero frames")
    return count


def seconds_from_frames(frames: int, rate: int) -> Fraction:
    check_rate(rate)
    if type(frames) is not int or frames < 0:
        raise AudioContractError("frames must be a nonnegative integer")
    return Fraction(frames, rate)


@dataclass(frozen=True, order=True, slots=True)
class FramePosition:
    """A nonnegative frame location on exactly one fixed source clock."""

    value: int
    sample_rate: int

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if type(self.value) is not int or self.value < 0:
            raise AudioContractError("Frame position cannot be negative")

    @property
    def seconds(self) -> Fraction:
        return seconds_from_frames(self.value, self.sample_rate)

    def advance(self, frames: int) -> "FramePosition":
        if type(frames) is not int:
            raise AudioContractError("Frame delta must be integer")
        return FramePosition(self.value + frames, self.sample_rate)

    def distance(self, other: "FramePosition") -> int:
        if not isinstance(other, FramePosition) or self.sample_rate != other.sample_rate:
            raise AudioContractError("Cannot mix different source clocks")
        return self.value - other.value


@dataclass(frozen=True, order=True, slots=True)
class FrameInterval:
    """Half-open frame interval [start,end) with explicit sample rate."""

    start: int
    stop: int
    sample_rate: int

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if (type(self.start) is not int or type(self.stop) is not int
                or self.start < 0 or self.stop <= self.start):
            raise AudioContractError("Expected nonempty half-open frame interval")

    @property
    def frames(self) -> int:
        return self.stop - self.start

    @property
    def start_seconds(self) -> Fraction:
        return seconds_from_frames(self.start, self.sample_rate)

    @property
    def stop_seconds(self) -> Fraction:
        return seconds_from_frames(self.stop, self.sample_rate)

    def overlaps(self, other: "FrameInterval") -> bool:
        self._same_clock(other)
        return max(self.start, other.start) < min(self.stop, other.stop)

    def touches(self, other: "FrameInterval") -> bool:
        self._same_clock(other)
        return self.stop == other.start or other.stop == self.start

    def intersection(self, other: "FrameInterval") -> "FrameInterval | None":
        self._same_clock(other)
        left, right = max(self.start, other.start), min(self.stop, other.stop)
        if left >= right:
            return None
        return FrameInterval(left, right, self.sample_rate)

    def union(self, other: "FrameInterval") -> "FrameInterval":
        self._same_clock(other)
        if not self.overlaps(other) and not self.touches(other):
            raise AudioContractError("Cannot union separated frame intervals")
        return FrameInterval(min(self.start, other.start),
                             max(self.stop, other.stop), self.sample_rate)

    def shifted(self, delta: int) -> "FrameInterval":
        if type(delta) is not int or self.start + delta < 0:
            raise AudioContractError("Invalid frame offset")
        return FrameInterval(self.start + delta, self.stop + delta, self.sample_rate)

    def _same_clock(self, other: "FrameInterval") -> None:
        if not isinstance(other, FrameInterval) or other.sample_rate != self.sample_rate:
            raise AudioContractError("Source clocks differ")


@dataclass(frozen=True, slots=True)
class TimeGrid:
    """One media clock with explicitly bounded, sample-aligned timeline."""

    sample_rate: int
    frames: int

    def __post_init__(self) -> None:
        check_rate(self.sample_rate)
        if type(self.frames) is not int or self.frames <= 0:
            raise AudioContractError("TimeGrid requires positive frame count")
        if self.frames > self.sample_rate * MAX_SECONDS:
            raise AudioLimitError("TimeGrid exceeds local audio duration limit")

    @property
    def duration(self) -> Fraction:
        return seconds_from_frames(self.frames, self.sample_rate)

    def at(self, seconds: int | float | Decimal | Fraction) -> FramePosition:
        frames = frames_from_seconds(seconds, self.sample_rate)
        if frames > self.frames:
            raise AudioContractError("Position outside TimeGrid")
        return FramePosition(frames, self.sample_rate)

    def interval(self, begin: int | float | Decimal | Fraction,
                 end: int | float | Decimal | Fraction) -> FrameInterval:
        a = self.at(begin).value
        b = self.at(end).value
        if a >= b:
            raise AudioContractError("Time interval is empty after rounding")
        return FrameInterval(a, b, self.sample_rate)

    def validate_interval(self, interval: FrameInterval) -> None:
        if (not isinstance(interval, FrameInterval)
                or interval.sample_rate != self.sample_rate
                or interval.stop > self.frames):
            raise AudioContractError("Frame interval outside TimeGrid")

    def partition(self, chunk_frames: int) -> tuple[FrameInterval, ...]:
        """Materialize only a bounded number of non-overlapping intervals.

        For very fine partitions of long audio, consume the lazy
        split_intervals() iterator instead of allocating millions of objects.
        """
        if type(chunk_frames) is not int or chunk_frames <= 0:
            raise AudioContractError("Partition chunk size must be positive integer")
        count = (self.frames + chunk_frames - 1) // chunk_frames
        if count > MAX_PARTITION_INTERVALS:
            raise AudioLimitError("Materialized timeline partition exceeds interval cap")
        return tuple(split_intervals(self.frames, self.sample_rate, chunk_frames))


def split_intervals(frames: int, sample_rate: int,
                    chunk_frames: int, *, overlap: int = 0,
                    include_last: bool = True) -> Iterator[FrameInterval]:
    """Yield bounded overlapping/adjacent chunks without trailing empty spans."""
    check_rate(sample_rate)
    if (type(frames) is not int or frames <= 0 or frames > MAX_SECONDS * sample_rate
            or type(chunk_frames) is not int or chunk_frames <= 0
            or type(overlap) is not int or not 0 <= overlap < chunk_frames
            or type(include_last) is not bool):
        raise AudioContractError("Invalid frame chunk geometry")
    step = chunk_frames - overlap
    start = 0
    while start < frames:
        end = min(frames, start + chunk_frames)
        if end - start < chunk_frames and not include_last:
            break
        yield FrameInterval(start, end, sample_rate)
        if end == frames:
            break
        start += step
