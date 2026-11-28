"""Safe, deterministic compressed MusicXML (.mxl) package interchange.

Compressed notation archives contain exactly the canonical MusicXML payload,
MIME marker and standard OCF container manifest. No media, links or arbitrary
zip members are supported. Input archive member metadata is validated *before*
anything is decompressed, then CRC, decoded bytes and XML shape are checked.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED, BadZipFile
import stat
import struct
import zlib
import xml.etree.ElementTree as ET

from .contracts import MidiFormatError, MidiLimitError, MidiWorkstationError
from .musicxml_common import MAX_MUSICXML_BYTES, parse_musicxml

MXL_MIME = b"application/vnd.recordare.musicxml"
MXL_PATH = "score.musicxml"
MXL_MANIFEST = "META-INF/container.xml"
MXL_MEMBERS = ("mimetype", MXL_MANIFEST, MXL_PATH)
MAX_MXL_BYTES = MAX_MUSICXML_BYTES
MANIFEST = (
    b'<?xml version="1.0" encoding="utf-8"?>'
    b'<container version="1.0" '
    b'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    b'<rootfiles><rootfile full-path="score.musicxml" '
    b'media-type="application/vnd.recordare.musicxml+xml"/>'
    b'</rootfiles></container>'
)


def _zip_info(name: str, *, compression: int = ZIP_DEFLATED) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


@dataclass(frozen=True, slots=True)
class MxlReceipt:
    archive_sha256: str
    archive_bytes: int
    xml_sha256: str
    xml_bytes: int
    compressed: bool
    protocol: str = "sonumio-canonical-mxl-v1"

    def as_dict(self) -> dict:
        return {
            "protocol": self.protocol,
            "archive_sha256": self.archive_sha256,
            "archive_bytes": self.archive_bytes,
            "xml_sha256": self.xml_sha256,
            "xml_bytes": self.xml_bytes,
            "compressed": self.compressed,
            "external_resources": False,
            "model_inference": False,
        }


def encode_mxl(xml: bytes) -> tuple[bytes, MxlReceipt]:
    if type(xml) is not bytes:
        raise MidiFormatError("Expected MusicXML bytes to archive")
    parse_musicxml(xml)
    stream = BytesIO()
    with ZipFile(stream, "w", allowZip64=False, strict_timestamps=True) as archive:
        archive.writestr(_zip_info("mimetype", compression=ZIP_STORED), MXL_MIME)
        archive.writestr(_zip_info(MXL_MANIFEST), MANIFEST)
        archive.writestr(_zip_info(MXL_PATH), xml)
    encoded = stream.getvalue()
    if len(encoded) > MAX_MXL_BYTES:
        raise MidiLimitError("Compressed MusicXML output exceeds package budget")
    return encoded, MxlReceipt(sha256(encoded).hexdigest(), len(encoded),
                               sha256(xml).hexdigest(), len(xml), True)


def decode_mxl(data: bytes) -> tuple[bytes, MxlReceipt]:
    if type(data) is not bytes or not 200 <= len(data) <= MAX_MXL_BYTES:
        raise MidiLimitError("Compressed MusicXML package outside size budget")
    try:
        with ZipFile(BytesIO(data), "r", allowZip64=False) as archive:
            members = archive.infolist()
            if tuple(entry.filename for entry in members) != MXL_MEMBERS:
                raise MidiFormatError("MXL must contain exactly the canonical three members")
            if members[0].compress_type != ZIP_STORED:
                raise MidiFormatError("MXL mimetype must be first and uncompressed")
            for info in members:
                if (info.flag_bits & 1 or info.is_dir() or info.file_size < 1
                        or info.file_size > MAX_MUSICXML_BYTES
                        or info.compress_size > MAX_MXL_BYTES
                        or info.compress_type not in (ZIP_STORED, ZIP_DEFLATED)):
                    raise MidiFormatError("Unsafe MXL archive member metadata")
                kind = (info.external_attr >> 16) & 0o170000
                if kind not in (0, stat.S_IFREG):
                    raise MidiFormatError("MXL archive contains a non-regular member")
            if archive.read("mimetype") != MXL_MIME:
                raise MidiFormatError("Invalid compressed MusicXML mimetype")
            manifest = archive.read(MXL_MANIFEST)
            if len(manifest) > 4096 or b"<!" in manifest.upper():
                raise MidiFormatError("Unsafe MXL container manifest")
            root = ET.fromstring(manifest)
            if root.tag.rsplit("}", 1)[-1] != "container":
                raise MidiFormatError("MXL manifest root must be container")
            rootfiles = [el for el in root.iter() if el.tag.rsplit("}", 1)[-1] == "rootfile"]
            if len(rootfiles) != 1 or rootfiles[0].get("full-path") != MXL_PATH:
                raise MidiFormatError("MXL manifest must reference only score.musicxml")
            xml = archive.read(MXL_PATH)
            if len(xml) != members[2].file_size:
                raise MidiFormatError("MXL member size changed during decompression")
            parse_musicxml(xml)
    except (BadZipFile, EOFError, OSError, RuntimeError, ET.ParseError,
            ValueError, struct.error, zlib.error, IndexError, KeyError) as exc:
        if isinstance(exc, MidiWorkstationError):
            raise
        raise MidiFormatError("Malformed or damaged compressed MusicXML package") from exc
    return xml, MxlReceipt(sha256(data).hexdigest(), len(data),
                           sha256(xml).hexdigest(), len(xml), True)
