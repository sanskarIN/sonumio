"""Compressed MusicXML archive security, reproducibility and corruption cases."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED, ZIP_STORED
import os
import random
import stat

import pytest

from sonumio.midi_workstation.contracts import MidiFormatError, MidiLimitError
from sonumio.midi_workstation.musicxml_export import encode_musicxml
from sonumio.midi_workstation.musicxml_import import decode_musicxml
from sonumio.midi_workstation.mxl import (
    MAX_MXL_BYTES, MANIFEST, MXL_MIME, MXL_PATH, MXL_MANIFEST,
    MXL_MEMBERS, decode_mxl, encode_mxl,
)
from sonumio.midi_workstation.score import Score


def fixture():
    return encode_musicxml(Score.empty())[0]


def repack(changes=None, *, order=None, compression=ZIP_DEFLATED):
    data = {"mimetype": MXL_MIME, MXL_MANIFEST: MANIFEST,
            MXL_PATH: fixture()}
    if changes:
        data.update(changes)
    output = BytesIO()
    with ZipFile(output, "w") as zf:
        for name in (order or MXL_MEMBERS):
            info = ZipInfo(name, date_time=(1980,1,1,0,0,0))
            info.compress_type = ZIP_STORED if name == "mimetype" else compression
            zf.writestr(info, data[name])
    return output.getvalue()


def test_canonical_archive_roundtrip():
    raw = fixture()
    archive, details = encode_mxl(raw)
    decoded, receipt = decode_mxl(archive)
    assert decoded == raw
    assert receipt.as_dict() == details.as_dict()
    assert decode_musicxml(decoded)[0].note_count == 0


def test_zip_has_stable_entry_order_and_mimetype_is_stored():
    archive, _ = encode_mxl(fixture())
    with ZipFile(BytesIO(archive)) as zf:
        assert tuple(zf.namelist()) == MXL_MEMBERS
        assert zf.infolist()[0].compress_type == ZIP_STORED
        assert zf.read("mimetype") == MXL_MIME
        assert zf.read(MXL_PATH) == fixture()


def test_repeated_archive_encoding_is_bit_identical():
    raw = fixture()
    first, _ = encode_mxl(raw)
    second, _ = encode_mxl(raw)
    assert first == second


@pytest.mark.parametrize("bad", [b"", b"a"*250, b"PK\x03\x04"+b"a"*245,
                                     bytes(range(256)), b"0"*(MAX_MXL_BYTES+1)])
def test_invalid_container_header_or_size_rejected(bad):
    with pytest.raises((MidiFormatError, MidiLimitError)):
        decode_mxl(bad)


def test_bad_mimetype_rejected():
    with pytest.raises(MidiFormatError, match="mimetype"):
        decode_mxl(repack({"mimetype": b"text/xml"}))


def test_reordered_members_rejected():
    data = repack(order=(MXL_PATH, MXL_MANIFEST, "mimetype"))
    with pytest.raises(MidiFormatError, match="exactly"):
        decode_mxl(data)


def test_missing_container_member_rejected():
    data = repack(order=("mimetype", MXL_PATH))
    with pytest.raises(MidiFormatError, match="exactly"):
        decode_mxl(data)


def test_added_unlisted_member_rejected_even_when_content_valid():
    data = repack(changes={"docs/readme.txt":b"not executed"},
                  order=(*MXL_MEMBERS,"docs/readme.txt"))
    with pytest.raises(MidiFormatError, match="exactly"):
        decode_mxl(data)


def test_zip_slip_path_rejected():
    data = repack(changes={"../score.musicxml":fixture()},
                  order=("mimetype", MXL_MANIFEST, "../score.musicxml"))
    with pytest.raises(MidiFormatError, match="exactly"):
        decode_mxl(data)


def test_absolute_zip_member_rejected():
    data = repack(changes={"/tmp/score.musicxml":fixture()},
                  order=("mimetype", MXL_MANIFEST, "/tmp/score.musicxml"))
    with pytest.raises(MidiFormatError, match="exactly"):
        decode_mxl(data)


def test_wrong_rootfile_in_manifest_rejected():
    changed = MANIFEST.replace(b'score.musicxml',b'../score.musicxml')
    with pytest.raises(MidiFormatError, match="manifest"):
        decode_mxl(repack({MXL_MANIFEST:changed}))


def test_two_rootfiles_not_supported():
    changed = MANIFEST.replace(b"</rootfiles>",
        b'<rootfile full-path="score.musicxml"/></rootfiles>')
    with pytest.raises(MidiFormatError, match="manifest"):
        decode_mxl(repack({MXL_MANIFEST:changed}))


def test_container_doctype_rejected():
    changed = MANIFEST.replace(b"<container ",
        b"<!DOCTYPE container><container ")
    with pytest.raises(MidiFormatError, match="Unsafe"):
        decode_mxl(repack({MXL_MANIFEST:changed}))


def test_trailing_archive_bytes_does_not_break_consistent_zip_package():
    original, _ = encode_mxl(fixture())
    assert decode_mxl(original + b"test")[0] == fixture()


def test_corrupt_member_detected_by_crc_or_xml_parser():
    original, _ = encode_mxl(fixture())
    index = original.find(b"score-partwise")
    if index > 0:
        modified = original[:index] + b"INVALID" + original[index+7:]
    else:
        modified = original[:40] + bytes([original[40]^255]) + original[41:]
    with pytest.raises((MidiFormatError, MidiLimitError)):
        decode_mxl(modified)


def test_symlink_entry_mode_disallowed():
    stream = BytesIO()
    with ZipFile(stream, "w") as zf:
        for name, body in (("mimetype",MXL_MIME), (MXL_MANIFEST,MANIFEST), (MXL_PATH,fixture())):
            info = ZipInfo(name, (1980,1,1,0,0,0))
            info.create_system = 3
            info.external_attr = ((stat.S_IFLNK if name==MXL_PATH else stat.S_IFREG)|0o777)<<16
            info.compress_type = ZIP_STORED
            zf.writestr(info, body)
    with pytest.raises(MidiFormatError, match="non-regular"):
        decode_mxl(stream.getvalue())


def test_duplicate_zip_names_rejected():
    stream = BytesIO()
    with ZipFile(stream,"w") as zf:
        zf.writestr("mimetype", MXL_MIME)
        zf.writestr(MXL_MANIFEST, MANIFEST)
        zf.writestr(MXL_PATH, fixture())
        zf.writestr(MXL_PATH, fixture())
    with pytest.raises(MidiFormatError, match="exactly"):
        decode_mxl(stream.getvalue())


def test_xml_embedded_in_zip_still_follows_xml_security_guards():
    payload = b"<!DOCTYPE score-partwise>" + fixture()
    with pytest.raises(MidiFormatError):
        decode_mxl(repack({MXL_PATH:payload}))


def test_mxl_does_not_create_files_while_decoding(tmp_path):
    before = set(tmp_path.iterdir())
    archive, _ = encode_mxl(fixture())
    assert decode_mxl(archive)[0] == fixture()
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("seed", range(35))
def test_random_single_byte_archive_corruption_does_not_execute_or_raise_other(seed):
    original, _ = encode_mxl(fixture())
    r = random.Random(seed)
    index = r.randrange(len(original))
    broken = bytearray(original)
    broken[index] ^= r.randrange(1, 256)
    try:
        result, receipt = decode_mxl(bytes(broken))
    except (MidiFormatError, MidiLimitError):
        return
    assert result == fixture()
