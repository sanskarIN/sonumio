"""Canonical fixed-width latent quantization and bounded in-memory packets.

A packet contains numeric codes only, not model parameters or executable
objects. It is a research format, not a production codec, encryption, or
entropy coder. The decoder requires the corresponding privately trained model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct

import numpy as np

FRAMES=(2048,8192)
LATENT_CHANNELS=16
TEMPORAL_STRIDE=64
BITS=(2,4,6,8)
MAGIC=b"SLCROP01"
# magic, waveform frames, channels, bit width, code count
HEADER=struct.Struct("<8sHBBI")
DIGEST_BYTES=32
MAX_CODES=max(FRAMES)//TEMPORAL_STRIDE*LATENT_CHANNELS
MAX_PACKET_BYTES=HEADER.size+MAX_CODES+DIGEST_BYTES


class CropCodecError(ValueError):
    """Invalid bounded crop codec input or packet."""


def check_frames(frames):
    if type(frames) is not int or frames not in FRAMES:
        raise CropCodecError("Unsupported crop waveform length")
    return frames


def check_bits(bits):
    if type(bits) is not int or bits not in BITS:
        raise CropCodecError("Unsupported fixed-width latent quantization")
    return bits


def signed_limit(bits):
    return (1<<(check_bits(bits)-1))-1


def code_shape(frames):
    return LATENT_CHANNELS,check_frames(frames)//TEMPORAL_STRIDE


def _checked_codes(codes,frames,bits):
    limit=signed_limit(bits)
    if (not isinstance(codes,np.ndarray) or codes.dtype!=np.int16
            or codes.shape!=code_shape(frames)
            or np.any(codes<-limit) or np.any(codes>limit)):
        raise CropCodecError("Invalid signed latent code geometry or range")
    return codes


def packet_bytes(frames,bits):
    channels,n=code_shape(frames)
    return HEADER.size+(channels*n*check_bits(bits)+7)//8+DIGEST_BYTES


def pack_codes(codes,*,frames,bits):
    """Pack signed symmetric levels using little-bit-order fixed-width words.

    The all-ones unsigned word is reserved; no ambiguous negative endpoint.
    The checksum covers the complete header and payload, not just data bits.
    """
    codes=_checked_codes(codes,frames,bits)
    words=(codes.astype(np.int32).ravel(order="C")+signed_limit(bits)).astype(np.uint16)
    bit_matrix=((words[:,None]>>np.arange(bits,dtype=np.uint16))&1).astype(np.uint8)
    payload=np.packbits(bit_matrix.ravel(),bitorder="little").tobytes()
    header=HEADER.pack(MAGIC,frames,LATENT_CHANNELS,bits,words.size)
    raw=header+payload
    result=raw+hashlib.sha256(raw).digest()
    if len(result)!=packet_bytes(frames,bits):
        raise CropCodecError("Unexpected latent packet byte count")
    return result


@dataclass(frozen=True,slots=True,eq=False)
class LatentPacket:
    frames:int
    bits:int
    codes:np.ndarray
    sha256:str
    bytes_written:int

    @property
    def nominal_payload_bits(self):
        return self.codes.size*self.bits


def unpack_codes(packet):
    """Check every header field and digest before decoding bounded bits."""
    if type(packet) is not bytes or not HEADER.size+DIGEST_BYTES<len(packet)<=MAX_PACKET_BYTES:
        raise CropCodecError("Invalid bounded latent packet")
    magic,frames,channels,bits,count=HEADER.unpack_from(packet)
    if magic!=MAGIC or channels!=LATENT_CHANNELS:
        raise CropCodecError("Unsupported latent packet header")
    shape=code_shape(frames)
    limit=signed_limit(bits)
    if count!=shape[0]*shape[1] or len(packet)!=packet_bytes(frames,bits):
        raise CropCodecError("Latent packet count or byte length mismatch")
    raw=packet[:-DIGEST_BYTES]
    if hashlib.sha256(raw).digest()!=packet[-DIGEST_BYTES:]:
        raise CropCodecError("Latent packet SHA256 mismatch")
    payload=np.frombuffer(raw[HEADER.size:],dtype=np.uint8)
    unpacked=np.unpackbits(payload,bitorder="little")
    used=count*bits
    if np.any(unpacked[used:]):
        raise CropCodecError("Noncanonical latent packet padding")
    words=unpacked[:used].reshape(count,bits).astype(np.uint16)@(1<<np.arange(bits,dtype=np.uint16))
    if np.any(words>2*limit):
        raise CropCodecError("Reserved latent quantization word")
    codes=(words.astype(np.int32)-limit).astype(np.int16).reshape(shape)
    codes=np.frombuffer(codes.tobytes(),dtype=np.int16).reshape(shape)
    return LatentPacket(frames,bits,codes,hashlib.sha256(packet).hexdigest(),len(packet))


def payload_rate_report(frames,bits):
    channels,n=code_shape(frames)
    count=channels*n
    check_bits(bits)
    size=packet_bytes(frames,bits)
    return {
        "waveform_frames":frames,"latent_scalars":count,"scalar_compression_ratio":frames/count,
        "quantization_bits_per_scalar":bits,"representable_levels":2*signed_limit(bits)+1,
        "nominal_payload_bits":count*bits,"fixed_width_packet_bytes":size,
        "header_and_digest_bytes":HEADER.size+DIGEST_BYTES,
        "packet_bits_per_input_sample":size*8/frames,
        "ratio_to_float32_pcm_bytes":frames*4/size,
        "entropy_coding_used":False,"production_bitstream_claimed":False,
    }
