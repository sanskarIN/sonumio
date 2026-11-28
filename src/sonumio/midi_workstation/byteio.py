"""Bounded Standard MIDI File binary primitives and integer codecs."""
from __future__ import annotations

from dataclasses import dataclass
import struct
from .contracts import (
    MAX_MIDI_BYTES, MAX_TICK, MidiFormatError, MidiLimitError,
    MidiWorkstationError, integer, raw_bytes,
)


@dataclass(slots=True)
class Cursor:
    data: memoryview
    offset: int = 0

    def __post_init__(self):
        if type(self.data) is not memoryview:
            raise MidiWorkstationError("Cursor requires a memoryview")
        if len(self.data) > MAX_MIDI_BYTES:
            raise MidiLimitError("Cursor source exceeds MIDI budget")
        integer(self.offset, "cursor position", 0, len(self.data))

    @classmethod
    def from_bytes(cls, data: bytes) -> "Cursor":
        raw_bytes(data, "MIDI source")
        return cls(memoryview(data))

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def take(self, size: int) -> bytes:
        integer(size, "read size", 0, MAX_MIDI_BYTES)
        if size > self.remaining:
            raise MidiFormatError("Truncated MIDI chunk")
        start = self.offset
        self.offset += size
        return self.data[start:self.offset].tobytes()

    def skip(self, size: int) -> None:
        self.take(size)

    def byte(self) -> int:
        if self.offset >= len(self.data):
            raise MidiFormatError("Unexpected end of MIDI input")
        number = self.data[self.offset]
        self.offset += 1
        return number

    def uint16(self) -> int:
        return int.from_bytes(self.take(2), "big")

    def uint32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def subcursor(self, size: int) -> "Cursor":
        return Cursor.from_bytes(self.take(size))

    def expect(self, pattern: bytes) -> None:
        raw_bytes(pattern, "expected bytes")
        if self.take(len(pattern)) != pattern:
            raise MidiFormatError(f"Expected MIDI signature {pattern!r}")

    def vlq(self) -> int:
        value = 0
        for index in range(4):
            number = self.byte()
            value = (value << 7) | (number & 0x7F)
            if not number & 0x80:
                return value
        raise MidiFormatError("MIDI variable-length integer exceeds four bytes")

    def exact_end(self) -> None:
        if self.remaining:
            raise MidiFormatError("Unexpected trailing bytes")


def encode_vlq(value: int) -> bytes:
    integer(value, "MIDI variable-length integer", 0, MAX_TICK)
    output = [value & 0x7F]
    value >>= 7
    while value:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(output))


def decode_vlq(data: bytes) -> int:
    stream = Cursor.from_bytes(data)
    number = stream.vlq()
    stream.exact_end()
    return number


def be16(value: int) -> bytes:
    return integer(value, "uint16", 0, 65535).to_bytes(2, "big")


def be32(value: int) -> bytes:
    return integer(value, "uint32", 0, 0xFFFFFFFF).to_bytes(4, "big")


def chunk(signature: bytes, payload: bytes) -> bytes:
    if signature not in (b"MThd", b"MTrk"):
        raise MidiWorkstationError("Unrecognized Standard MIDI File chunk")
    raw_bytes(payload, "MIDI chunk payload")
    if len(payload) + 8 > MAX_MIDI_BYTES:
        raise MidiLimitError("MIDI chunk exceeds file budget")
    return signature + be32(len(payload)) + payload


def signed_8(value: int) -> int:
    integer(value, "signed MIDI byte", 0, 255)
    return value if value < 128 else value - 256


def pack_signed_8(value: int) -> bytes:
    integer(value, "signed MIDI byte", -128, 127)
    return bytes((value & 255,))


def nibble(value: int) -> int:
    return integer(value, "MIDI channel", 0, 15)


def data_byte(value: int) -> int:
    return integer(value, "MIDI data byte", 0, 127)


def delta_sort_key(tick: int, status: int, data: bytes) -> tuple[int, int, bytes]:
    """Stable semantic ordering: note release before same-tick attack."""
    integer(tick, "event tick", 0, MAX_TICK)
    integer(status, "MIDI status", 0, 255)
    if 0x80 <= status <= 0x8F or (0x90 <= status <= 0x9F and len(data) == 2
                                  and data[1] == 0):
        priority = 0
    elif status == 0xFF:
        priority = 2
    elif 0x90 <= status <= 0x9F:
        priority = 3
    else:
        priority = 1
    return tick, priority, data
