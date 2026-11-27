"""Deterministic stored ZIP sound packs and bounded integrity verification.

Verification never extracts files. The archive format has one canonical
manifest and one canonical WAV per planned job. Reconstructing the stored
ZIP representation rejects ambiguous local headers, extra entries, trailing
data and alternate member metadata. Integrity checks are not a signature or
proof that somebody else's claimed source recipe was actually used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import os
import stat
import tempfile
import zipfile

from .asset_pack import (
    SoundPackPlan, MAX_ASSETS, MAX_PAYLOAD_BYTES, MAX_MEMBER_BYTES,
    plan_jobs, parse_pack_plan, generate_pack_members, inspect_asset,
    build_manifest, render_asset,
)
from .errors import AudioContractError, AudioFormatError, AudioIntegrityError, AudioLimitError
from .hashing import canonical_json_bytes, parse_canonical_json, strict_sha256

MAX_MANIFEST_BYTES=512*1024
MAX_ARCHIVE_BYTES=MAX_PAYLOAD_BYTES+MAX_MANIFEST_BYTES+128*1024
MANIFEST_NAME="manifest.json"
FIXED_DATE=(1980,1,1,0,0,0)


def _archive_digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _member_info(name):
    info=zipfile.ZipInfo(filename=name,date_time=FIXED_DATE)
    info.create_system=3
    info.external_attr=0o100644<<16
    info.compress_type=zipfile.ZIP_STORED
    info.extra=b""
    info.comment=b""
    return info


def _canonical_archive(manifest_raw,members):
    if not isinstance(manifest_raw,bytes) or not 0<len(manifest_raw)<=MAX_MANIFEST_BYTES:
        raise AudioLimitError("Sound pack manifest exceeds byte limit")
    if type(members) is not dict or not 1<=len(members)<=MAX_ASSETS:
        raise AudioLimitError("Sound pack member count exceeds limit")
    if any(not isinstance(x,bytes) or not 44<len(x)<=MAX_MEMBER_BYTES for x in members.values()):
        raise AudioLimitError("Sound pack WAV member exceeds byte limit")
    if sum(map(len,members.values()))>MAX_PAYLOAD_BYTES:
        raise AudioLimitError("Sound pack payload exceeds byte limit")
    out=io.BytesIO()
    with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_STORED,allowZip64=False) as archive:
        archive.writestr(_member_info(MANIFEST_NAME),manifest_raw)
        for name in sorted(members):
            archive.writestr(_member_info(name),members[name])
    raw=out.getvalue()
    if len(raw)>MAX_ARCHIVE_BYTES:
        raise AudioLimitError("Sound pack archive exceeds byte limit")
    return raw


def _validate_raw(raw,expected_sha256):
    if not isinstance(raw,bytes) or not raw:
        raise AudioFormatError("Expected immutable sound pack ZIP bytes")
    if len(raw)>MAX_ARCHIVE_BYTES:
        raise AudioLimitError("Sound pack archive exceeds byte limit")
    digest=_archive_digest(raw)
    if expected_sha256 is not None:
        strict_sha256(expected_sha256)
        if digest!=expected_sha256:
            raise AudioIntegrityError("Sound pack ZIP SHA256 mismatch")
    return digest


def _verify_entry(info,limit):
    if (info.compress_type!=zipfile.ZIP_STORED or info.flag_bits!=0
            or info.extra or info.comment or info.is_dir()):
        raise AudioFormatError("Sound pack requires ordinary stored ZIP members")
    if info.file_size!=info.compress_size or not 0<info.file_size<=limit:
        raise AudioLimitError("Sound pack member exceeds declared byte limit")


def _read_member(archive,info,limit):
    _verify_entry(info,limit)
    with archive.open(info,"r") as stream:
        value=stream.read(limit+1)
        if len(value)!=info.file_size or stream.read(1):
            raise AudioIntegrityError("Sound pack member length mismatch")
    return value


def _read_archive(raw):
    """Validate names and geometry before loading the audio members."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw),"r") as archive:
            if archive.comment:
                raise AudioFormatError("Sound pack ZIP comments are not permitted")
            infos=archive.infolist()
            if not 2<=len(infos)<=MAX_ASSETS+1:
                raise AudioLimitError("Sound pack ZIP member count exceeds bound")
            if infos[0].filename!=MANIFEST_NAME:
                raise AudioFormatError("Sound pack manifest must be first")
            names=[info.filename for info in infos]
            if len(names)!=len(set(names)):
                raise AudioFormatError("Duplicate sound pack ZIP member")
            manifest_raw=_read_member(archive,infos[0],MAX_MANIFEST_BYTES)
            obj=parse_canonical_json(manifest_raw,max_bytes=MAX_MANIFEST_BYTES)
            if type(obj) is not dict or type(obj.get("plan")) is not dict:
                raise AudioFormatError("Sound pack manifest is missing its plan")
            plan=parse_pack_plan(canonical_json_bytes(obj["plan"]))
            jobs=plan_jobs(plan)
            expected=[MANIFEST_NAME,*[job.member for job in jobs]]
            # Canonical ZIP order is lexicographic by complete filename.
            if names!=[MANIFEST_NAME,*sorted(expected[1:])]:
                raise AudioIntegrityError("Archive members differ from declared plan")
            if sum(info.file_size for info in infos[1:])!=plan.payload_bytes:
                raise AudioIntegrityError("Archive PCM byte budget differs from plan")
            members={info.filename:_read_member(archive,info,MAX_MEMBER_BYTES)
                     for info in infos[1:]}
    except (zipfile.BadZipFile,zipfile.LargeZipFile,EOFError,NotImplementedError) as exc:
        raise AudioFormatError("Invalid sound pack ZIP structure") from exc
    return manifest_raw,plan,members


@dataclass(frozen=True,slots=True)
class VerifiedPack:
    archive_sha256: str
    archive_bytes: int
    manifest_sha256: str
    manifest: dict
    replayed_assets: int=0
    protocol: str="sonumio-verified-sound-pack-v1"

    def as_dict(self):
        return {
            "protocol":self.protocol,
            "archive_sha256":self.archive_sha256,
            "archive_bytes":self.archive_bytes,
            "manifest_sha256":self.manifest_sha256,
            "manifest":self.manifest,
            "replayed_assets":self.replayed_assets,
            "recipe_replay_verified":self.replayed_assets==self.manifest["asset_count"],
            "signature_verified":False,
        }


def verify_pack_bytes(raw,*,expected_sha256=None,replay=False):
    """Verify canonical container, exact planned geometry and PCM receipts.

    A replay additionally runs the original procedural renderer for every
    asset and requires exact WAV bytes. Replay requires a compatible local
    numerical runtime; a cross-runtime floating-point change can fail it.
    """
    if type(replay) is not bool:
        raise AudioContractError("Replay flag must be boolean")
    digest=_validate_raw(raw,expected_sha256)
    manifest_raw,plan,members=_read_archive(raw)
    jobs=plan_jobs(plan)
    records=tuple(inspect_asset(job,members[job.member],plan.output_bits) for job in jobs)
    expected=build_manifest(plan,records)
    canonical=canonical_json_bytes(expected,max_bytes=MAX_MANIFEST_BYTES)
    if manifest_raw!=canonical:
        raise AudioIntegrityError("Sound pack manifest does not match exact WAV receipts")
    if _canonical_archive(manifest_raw,members)!=raw:
        raise AudioIntegrityError("Sound pack ZIP differs from canonical representation")
    if replay:
        for job in jobs:
            if render_asset(job,plan)!=members[job.member]:
                raise AudioIntegrityError("Procedural recipe replay mismatch: "+job.member)
    return VerifiedPack(digest,len(raw),_archive_digest(manifest_raw),expected,
                        len(jobs) if replay else 0)


def create_pack_bytes(plan):
    if not isinstance(plan,SoundPackPlan):
        raise AudioContractError("Expected SoundPackPlan")
    members,manifest=generate_pack_members(plan)
    encoded=canonical_json_bytes(manifest,max_bytes=MAX_MANIFEST_BYTES)
    raw=_canonical_archive(encoded,members)
    return raw,verify_pack_bytes(raw)


def _snapshot(path,limit):
    """Bounded descriptor snapshot with best-effort local change detection."""
    from .._file_identity import same_path_descriptor,same_path_identity,same_snapshot
    path=Path(path)
    try:
        before=path.lstat()
    except OSError as exc:
        raise AudioFormatError("Expected ordinary local pack input") from exc
    if not stat.S_ISREG(before.st_mode):
        raise AudioFormatError("Expected ordinary local pack input")
    flags=(os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_NONBLOCK",0)
           |getattr(os,"O_BINARY",0))
    fd=os.open(path,flags)
    with os.fdopen(fd,"rb") as stream:
        first=os.fstat(stream.fileno())
        if not stat.S_ISREG(first.st_mode) or not same_path_descriptor(before,first):
            raise AudioIntegrityError("Pack input changed during descriptor open")
        if first.st_size>limit:
            raise AudioLimitError("Pack input exceeds byte limit")
        value=stream.read(limit+1)
        after=os.fstat(stream.fileno())
        try:
            at_path=path.lstat()
        except OSError as exc:
            raise AudioIntegrityError("Pack input path disappeared") from exc
        if (not same_snapshot(first,after) or not same_path_identity(before,at_path)
                or not same_snapshot(before,at_path) or not same_path_descriptor(at_path,after)
                or len(value)!=first.st_size):
            raise AudioIntegrityError("Pack input changed during bounded read")
    if len(value)>limit:
        raise AudioLimitError("Pack input exceeds byte limit")
    return value


def load_pack_plan(path):
    from .asset_pack import MAX_PLAN_BYTES
    return parse_pack_plan(_snapshot(path,MAX_PLAN_BYTES))


def verify_pack_file(path,*,expected_sha256=None,replay=False):
    if expected_sha256 is not None:
        strict_sha256(expected_sha256)
    return verify_pack_bytes(_snapshot(path,MAX_ARCHIVE_BYTES),
                             expected_sha256=expected_sha256,replay=replay)


@dataclass(frozen=True,slots=True)
class PublishedPack:
    path: Path
    verified: VerifiedPack

    def as_dict(self):
        return {**self.verified.as_dict(),"destination":str(self.path)}


def _parent(target,create_parent):
    parent=Path(os.path.abspath(target.parent))
    if any(p.is_symlink() for p in (parent,*parent.parents)):
        raise AudioFormatError("Pack output parent must not contain symlinks")
    if create_parent:
        parent.mkdir(parents=True,exist_ok=True)
    if not parent.is_dir() or any(p.is_symlink() for p in (parent,*parent.parents)):
        raise AudioFormatError("Expected ordinary pack output parent directory")


def _target(target,overwrite):
    try:
        metadata=target.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(metadata.st_mode):
        raise AudioFormatError("Pack output must be an ordinary file")
    if not overwrite:
        raise AudioFormatError("Existing pack requires explicit overwrite")


def publish_pack_bytes(raw,destination,*,expected_sha256=None,overwrite=False,
                       create_parent=False):
    """Validate and publish one bounded archive without partial output files."""
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Pack publication flags must be boolean")
    target=Path(destination)
    if target.suffix.lower()!=".zip":
        raise AudioFormatError("Sound pack destination must end in .zip")
    receipt=verify_pack_bytes(raw,expected_sha256=expected_sha256)
    _parent(target,create_parent)
    _target(target,overwrite)
    scratch=None
    try:
        # Write to the exclusively created descriptor, never reopen an
        # attacker-selectable temporary filename.
        with tempfile.NamedTemporaryFile(mode="w+b",dir=target.parent,
                prefix=".sonumio-pack-",suffix=".partial",delete=False) as stream:
            scratch=Path(stream.name)
            if stream.write(raw)!=len(raw):
                raise OSError("Short staged sound pack write")
            stream.flush()
            os.fsync(stream.fileno())
            stream.seek(0)
            h=hashlib.sha256()
            length=0
            while True:
                data=stream.read(1024*1024)
                if not data:
                    break
                length+=len(data)
                h.update(data)
            if length!=len(raw) or h.hexdigest()!=receipt.archive_sha256:
                raise AudioIntegrityError("Staged sound pack checksum mismatch")
        _parent(target,False)
        _target(target,overwrite)
        if overwrite:
            os.replace(scratch,target)
            scratch=None
        else:
            try:
                os.link(scratch,target)
            except FileExistsError as exc:
                raise AudioFormatError("Existing pack requires explicit overwrite") from exc
            except OSError as exc:
                raise AudioFormatError("Atomic no-clobber pack publication unsupported") from exc
        return PublishedPack(target,receipt)
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)


def export_sound_pack(plan,destination,*,expected_sha256=None,overwrite=False,
                      create_parent=False):
    if expected_sha256 is not None:
        strict_sha256(expected_sha256)
    if type(overwrite) is not bool or type(create_parent) is not bool:
        raise AudioContractError("Pack publication flags must be boolean")
    if Path(destination).suffix.lower()!=".zip":
        raise AudioFormatError("Sound pack destination must end in .zip")
    raw,_=create_pack_bytes(plan)
    return publish_pack_bytes(raw,destination,expected_sha256=expected_sha256,
                              overwrite=overwrite,create_parent=create_parent)
