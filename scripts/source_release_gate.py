"""Audit and smoke-test a pinned Sonumio source candidate without publishing it.

Run: python scripts/source_release_gate.py --smoke
Uses tracked Git source only, then a disposable virtual environment.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unicodedata
import venv
import wave
import zipfile

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = frozenset({
    "pyproject.toml", "README.md", "LICENSE",
    "src/sonumio/__init__.py", "src/sonumio/cli.py",
    "src/sonumio/api.py", "src/sonumio/engines/procedural.py",
    "web/package.json", "scripts/source_release_gate.py",
    "src/sonumio/demo.py", "src/sonumio/audio_core/stream_spectrum.py",
    "src/sonumio/audio_core/spectrum_cli.py",
    "src/sonumio/audio_core/granular.py", "src/sonumio/audio_core/modal_synthesis.py",
    "src/sonumio/audio_core/spatial_motion.py", "src/sonumio/audio_core/mastering.py",
    "src/sonumio/audio_core/asset_pack.py", "src/sonumio/audio_core/asset_archive.py",
    "src/sonumio/audio_core/asset_selection.py",
})
BANNED_PARTS = frozenset({
    ".git", "node_modules", "dist", "build", "outputs", "checkpoints",
    "recordings", "private-data", ".venv", "__pycache__", ".pytest_cache",
})
BANNED_SUFFIXES = frozenset({
    ".wav", ".flac", ".mp3", ".ogg", ".mp4", ".npz", ".npy", ".pt",
    ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle",
    ".h5", ".hdf5", ".exe", ".dll", ".dylib", ".so",
    ".pem", ".key", ".p12", ".pfx", ".sqlite", ".db",
    ".scorpus", ".stcheckpoint", ".partial",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar",
    ".aiff", ".aif", ".m4a", ".aac", ".webm", ".mkv", ".opus",
    ".whl", ".msi", ".dmg", ".iso", ".obj", ".o", ".lib", ".a",
})
# Check known binary signatures independently of extensions. These are
# narrow format guards, not general secret scanning or authenticity checks.
PRIVATE_ARTIFACT_MAGIC = (
    b"SCORPUS1", b"STCP0001", b"SLHAND01", b"SLCROP01",
    b"\x93NUMPY", b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08",
    b"\x1f\x8b\x08", b"BZh", b"\xfd7zXZ\x00", b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07", b"\x7fELF", b"\x89HDF\r\n\x1a\n",
    b"SQLite format 3\x00", b"\x1a\x45\xdf\xa3",
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
)

def reject_artifact_contents(path: str, data: bytes) -> None:
    if data.startswith(PRIVATE_ARTIFACT_MAGIC):
        raise RuntimeError(f"Binary artifact signature in source entry: {path}")
    if (len(data) >= 12 and data[:4] in (b"RIFF", b"RF64", b"RIFX")
            and data[8:12] == b"WAVE"):
        raise RuntimeError(f"Binary artifact signature in source entry: {path}")
    if data.startswith((b"fLaC", b"OggS")):
        raise RuntimeError(f"Binary artifact signature in source entry: {path}")
    # Container formats with identifiers at fixed nonzero offsets.
    if ((len(data) >= 12 and data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"))
            or (len(data) >= 12 and data[4:8] == b"ftyp")
            or (len(data) >= 263 and data[257:262] == b"ustar")):
        raise RuntimeError(f"Binary artifact signature in source entry: {path}")
    # A DOS header alone can also occur in legitimate text. Require the
    # bounded PE pointer and its signature before classifying it as PE.
    if len(data) >= 64 and data[:2] == b"MZ":
        offset = int.from_bytes(data[60:64], "little")
        if 64 <= offset <= len(data) - 4 and data[offset:offset + 4] == b"PE\x00\x00":
            raise RuntimeError(f"Binary artifact signature in source entry: {path}")


WINDOWS_DEVICES = frozenset({
    "con", "prn", "aux", "nul", "conin$", "conout$",
    *(f"{kind}{digit}" for kind in ("com", "lpt") for digit in "123456789¹²³"),
})
WINDOWS_FORBIDDEN = frozenset('<>:"|?*\\')


def _portable_path_part(part: str) -> bool:
    """Check one archive component before any extraction on any platform."""
    if (not part or part.endswith((" ", "."))
            or any(ch in WINDOWS_FORBIDDEN or unicodedata.category(ch) == "Cc" for ch in part)
            or unicodedata.normalize("NFC", part) != part):
        return False
    # Windows treats devices as reserved even with a suffix; spaces before
    # the first dot do not make a reserved device name safe.
    stem = part.split(".", 1)[0].rstrip(" ").casefold()
    return stem not in WINDOWS_DEVICES

MAX_FILES = 2000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


def _run(args: list[str], *, cwd: Path,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, env=env, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _validate_version(version: str) -> str:
    if (type(version) is not str
            or not re.fullmatch(r"(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})\.(?:0|[1-9][0-9]{0,5})", version)):
        raise RuntimeError("Expected canonical numeric source version (major.minor.patch)")
    return version


@dataclass(frozen=True)
class SourceCandidate:
    commit: str
    version: str

    def __post_init__(self):
        if type(self.commit) is not str or not re.fullmatch(r"(?:[a-f0-9]{40}|[a-f0-9]{64})", self.commit):
            raise RuntimeError("Expected full pinned Git commit hash")
        _validate_version(self.version)

    @property
    def tag(self) -> str:
        return f"v{self.version}"

    @property
    def prefix(self) -> str:
        return f"sonumio-{self.tag}/"


def resolve_candidate(expected_version: str | None = None) -> SourceCandidate:
    if expected_version is not None:
        _validate_version(expected_version)
    commit = _run(["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=ROOT).stdout.strip()
    source = _run(["git", "show", f"{commit}:pyproject.toml"], cwd=ROOT).stdout
    project = tomllib.loads(source)["project"]
    version = _validate_version(project.get("version"))
    if expected_version is not None and version != expected_version:
        raise RuntimeError(f"Expected source version {expected_version}, found {version}")
    return SourceCandidate(commit, version)


def _make_archive(archive: Path, candidate: SourceCandidate, *, format: str = 'tar.gz') -> None:
    if format not in ('tar.gz', 'zip'):
        raise ValueError('Unsupported source archive format')
    # On Windows, core.autocrlf=true can change bytes INSIDE git archive.
    # GitHub source snapshots must preserve committed blobs exactly.
    _run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
          "archive", f"--format={format}", f"--prefix={candidate.prefix}",
          f"--output={archive}", candidate.commit], cwd=ROOT)
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError("Git source archive was not created")


MAX_ARCHIVE_BYTES = 40 * 1024 * 1024
MAX_ENTRIES = 4000


class SourceEntries:
    """Shared bounded archive path and content accumulator (never extracts)."""

    def __init__(self, candidate):
        self.prefix = candidate.prefix
        self.seen_paths = set()
        self.files = {}
        self.directories = set()
        self.total_size = 0
        self.entries = 0
        self.root_seen = False

    def accept(self, name, *, directory, regular, size, read):
        self.entries += 1
        if self.entries > MAX_ENTRIES:
            raise RuntimeError("Source archive exceeds entry review limits")
        if name in (self.prefix, self.prefix.rstrip("/")):
            if not directory or self.root_seen or size != 0:
                raise RuntimeError("Invalid or duplicate archive root")
            self.root_seen = True
            return
        if not name.startswith(self.prefix):
            raise RuntimeError("Unexpected source archive prefix")
        relative = name[len(self.prefix):]
        path = PurePosixPath(relative)
        if (path.is_absolute() or ".." in path.parts or "\\" in relative
                or not path.parts or any(ord(ch) < 32 or ord(ch) == 127 for ch in relative)
                or any(not _portable_path_part(part) for part in path.parts)):
            raise RuntimeError(f"Unsafe archive path: {relative}")
        canonical = path.as_posix()
        if relative != canonical and not (directory and relative == canonical + "/"):
            raise RuntimeError(f"Noncanonical source path: {relative}")
        folded = canonical.casefold()
        if folded in self.seen_paths:
            raise RuntimeError(f"Duplicate or case-colliding source path: {relative}")
        self.seen_paths.add(folded)
        if any(part.casefold() in BANNED_PARTS for part in path.parts):
            raise RuntimeError(f"Private or generated directory: {relative}")
        if directory:
            if size != 0:
                raise RuntimeError(f"Unexpected directory payload: {relative}")
            self.directories.add(canonical)
            return
        if not regular:
            raise RuntimeError(f"Non-regular archive entry: {relative}")
        if (path.suffix.lower() in BANNED_SUFFIXES
                or path.name.lower() in {".env", ".npmrc", "id_rsa", "id_ed25519"}):
            raise RuntimeError(f"Binary or sensitive artifact: {relative}")
        if not 0 <= size <= MAX_FILE_BYTES:
            raise RuntimeError(f"Unexpectedly large or invalid source entry: {relative}")
        self.total_size += size
        if self.total_size > MAX_TOTAL_BYTES or len(self.files) >= MAX_FILES:
            raise RuntimeError("Source archive exceeds review limits")
        data = read()
        if len(data) != size:
            raise RuntimeError(f"Source archive entry changed: {relative}")
        reject_artifact_contents(relative, data)
        self.files[canonical] = data


def _read_archive(archive: Path, candidate: SourceCandidate, format: str) -> SourceEntries:
    if not 0 < archive.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise RuntimeError("Compressed source archive exceeds review limits")
    entries = SourceEntries(candidate)
    if format == "tar.gz":
        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                def read():
                    stream = tar.extractfile(member)
                    if stream is None:
                        raise RuntimeError(f"Unreadable source entry: {member.name}")
                    with stream:
                        return stream.read(MAX_FILE_BYTES + 1)
                entries.accept(member.name, directory=member.isdir(), regular=member.isfile(),
                               size=member.size, read=read)
    elif format == "zip":
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                mode = stat.S_IFMT(member.external_attr >> 16)
                directory = member.is_dir()
                regular = mode in (0, stat.S_IFREG)
                if (member.flag_bits & 1 or member.compress_type not in
                        (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                        or (directory and mode not in (0, stat.S_IFDIR))
                        or (not directory and not regular)):
                    raise RuntimeError(f"Unsupported or non-regular ZIP entry: {member.filename}")
                # zipfile truncates names at NUL; compare with original filename.
                if member.orig_filename != member.filename:
                    raise RuntimeError("Noncanonical ZIP entry name")
                def read():
                    with zipped.open(member) as stream:
                        return stream.read(MAX_FILE_BYTES + 1)
                entries.accept(member.filename, directory=directory, regular=regular,
                               size=member.file_size, read=read)
    else:
        raise ValueError("Unsupported source archive format")
    return entries


def _audit_archive(archive: Path, candidate: SourceCandidate, *, format: str = "tar.gz") -> dict[str, object]:
    entries = _read_archive(archive, candidate, format)
    files, total_size = entries.files, entries.total_size
    # Check *content* against each canonical Git blob object, not just its
    # length. Same-size archive tampering must never pass this release gate.
    # Support Git repositories using either SHA-1 or SHA-256 object formats.
    object_format = _run(
        ["git", "rev-parse", "--show-object-format"], cwd=ROOT
    ).stdout.strip()
    if object_format not in ("sha1", "sha256"):
        raise RuntimeError("Unsupported Git object format for release audit")
    tracked: dict[str, tuple[int, str]] = {}
    tree_output = _run(["git", "ls-tree", "-r", "-l", "-z", candidate.commit], cwd=ROOT).stdout
    for line in tree_output.split("\0"):
        if not line:
            continue
        header, separator, filename = line.partition("\t")
        pieces = header.split()
        if separator != "\t" or len(pieces) != 4 or pieces[1] != "blob" or pieces[0] not in ("100644", "100755"):
            raise RuntimeError("Unexpected Git tree entry in source audit")
        if filename in tracked:
            raise RuntimeError(f"Duplicate Git tree entry: {filename}")
        tracked[filename] = (int(pieces[3]), pieces[2])
    if files.keys() != tracked.keys():
        raise RuntimeError("Source archive paths differ from the committed tree")
    # Git has no independent empty directory objects. Every exported
    # directory must be an ancestor of an actual tracked source file.
    expected_directories = {
        parent.as_posix() for filename in tracked
        for parent in PurePosixPath(filename).parents if parent != PurePosixPath(".")
    }
    if not entries.root_seen or entries.directories != expected_directories:
        raise RuntimeError("Source archive directories differ from the committed tree")
    for filename, data in files.items():
        expected_size, expected_oid = tracked[filename]
        if len(data) != expected_size:
            raise RuntimeError(f"Noncanonical exported source bytes: {filename}")
        blob = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
        digest = (hashlib.sha1(blob).hexdigest() if object_format == "sha1"
                  else hashlib.sha256(blob).hexdigest())
        if digest != expected_oid:
            raise RuntimeError(f"Source archive Git blob checksum mismatch: {filename}")
    missing = sorted(REQUIRED.difference(files))
    if missing:
        raise RuntimeError(f"Missing buildable-source entries: {missing}")
    project = tomllib.loads(files["pyproject.toml"].decode("utf-8"))["project"]
    if project.get("name") != "sonumio" or project.get("version") != candidate.version:
        raise RuntimeError("Python package metadata differs from pinned source candidate")
    if project.get("license") != "Apache-2.0":
        raise RuntimeError("Sonumio source-license metadata changed")
    package_init = files["src/sonumio/__init__.py"].decode("utf-8")
    declarations = [
        node.value for node in ast.parse(package_init).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]
    if (len(declarations) != 1 or not isinstance(declarations[0], ast.Constant)
            or declarations[0].value != candidate.version):
        raise RuntimeError("Python module version differs from pyproject.toml")
    api_source = files["src/sonumio/api.py"].decode("utf-8")
    if "version=__version__" not in api_source:
        raise RuntimeError("API version must follow the package version")
    web = json.loads(files["web/package.json"])
    if web.get("private") is not True:
        raise RuntimeError("Web package must remain private / unpublished")
    if web.get("version") != candidate.version:
        raise RuntimeError("Web package version differs from source candidate")
    if not any(key.startswith("tests/test_") and key.endswith(".py")
               for key in files):
        raise RuntimeError("Source archive has no tests")
    return {
        "tag_candidate": candidate.tag, "commit_candidate": candidate.commit,
        "software_version": candidate.version, "archive_format": format, "source_files": len(files),
        "source_bytes": total_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "canonical_blob_bytes_verified": True,
        "canonical_directory_tree_verified": True,
        "canonical_blob_hashes_verified": True,
        "published": False, "binary_or_model_artifacts": 0,
    }


def _smoke_archive(archive: Path, work: Path, candidate: SourceCandidate) -> dict[str, object]:
    version = candidate.version
    work.mkdir(parents=True, exist_ok=False)
    venv_dir = work / "isolated-env"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    if os.name == "nt":
        python = venv_dir / "Scripts" / "python.exe"
        console = venv_dir / "Scripts" / "sonumio.exe"
    else:
        python = venv_dir / "bin" / "python"
        console = venv_dir / "bin" / "sonumio"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    _run([str(python), "-m", "pip", "install", "--no-input",
          "--disable-pip-version-check", str(archive)], cwd=work, env=env)
    version_check = (
        "from importlib.metadata import version; "
        "import sonumio; from sonumio.api import app; "
        f"assert version('sonumio') == '{version}'; "
        f"assert sonumio.__version__ == '{version}'; "
        f"assert app.version == '{version}'"
    )
    _run([str(python), "-c", version_check], cwd=work, env=env)
    reported = _run([str(console), "--version"], cwd=work, env=env).stdout.strip()
    if reported != f"sonumio {version}":
        raise RuntimeError(f"Unexpected console-script version: {reported}")
    registered = _run([str(console), "models"], cwd=work, env=env).stdout
    if "procedural-sfx-v0" not in registered:
        raise RuntimeError("Installed source lacks procedural engine registration")
    destination = work / "source-built-smoke.wav"
    _run([str(console), "generate", "rain", "--seconds", "0.5",
          "--seed", "7", "--output", str(destination)], cwd=work, env=env)
    with wave.open(str(destination), "rb") as source:
        if (source.getframerate() != 24_000 or source.getnchannels() != 1
                or source.getsampwidth() != 2 or source.getnframes() != 12_000):
            raise RuntimeError("Installed source produced invalid PCM WAV")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    inspected = json.loads(_run([
        str(console), "wav-inspect", str(destination),
        "--expected-sha256", digest, "--timeline", "--continuity",
        "--quiet-intervals", "--channel-relationships",
    ], cwd=work, env=env).stdout)
    if (inspected.get("source_sha256") != digest
            or inspected.get("source_verified") is not True
            or inspected.get("frames") != 12000
            or inspected.get("signal_health", {}).get("clipping_fraction") != 0
            or not {"timeline", "frame_continuity", "quiet_intervals",
                    "channel_relationships"}.issubset(inspected)):
        raise RuntimeError("Installed source WAV inspection failed")
    spectrum = json.loads(_run([
        str(python), "-m", "sonumio.audio_core.spectrum_cli", str(destination),
        "--expected-sha256", digest, "--fft-frames", "256", "--include-bins",
    ], cwd=work, env=env).stdout)
    _check_spectrum_smoke(spectrum, digest, destination)
    demo_path = work / "source-built-demo.wav"
    demo = json.loads(_run([
        str(python), "-m", "sonumio.demo", "--output", str(demo_path),
    ], cwd=work, env=env).stdout)
    _check_demo_smoke(demo, demo_path, version)
    repeated = work / "source-built-demo-repeat.wav"
    repeat = json.loads(_run([
        str(python), "-m", "sonumio.demo", "--output", str(repeated),
        "--expected-sha256", demo["wav_sha256"],
    ], cwd=work, env=env).stdout)
    if (repeat != demo or repeated.read_bytes() != demo_path.read_bytes()):
        raise RuntimeError("Installed source procedural demo is not reproducible")
    sound_design = _smoke_sound_design(console, work, env, destination, digest)
    return {
        **sound_design,
        "installed_version": reported, "source_install": "isolated non-editable",
        "generated_wav_frames": 12_000,
        "verified_wav_inspection": True, "verified_spectrum": True,
        "verified_demo": True, "demo_repeat_identical": True,
        "demo_wav_sha256": demo["wav_sha256"], "uploaded_artifacts": 0,
    }


def _check_generated_pcm(report, destination, geometry):
    """Independently verify the installed CLI's encoded PCM file."""
    data = destination.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if report.get("output_sha256") != digest:
        raise RuntimeError("Installed sound design output digest mismatch")
    with wave.open(str(destination), "rb") as source:
        actual = (source.getframerate(), source.getnchannels(),
                  source.getsampwidth(), source.getnframes())
        if actual != geometry or source.getcomptype() != "NONE":
            raise RuntimeError("Installed sound design PCM geometry mismatch")
        payload = source.readframes(source.getnframes())
    if len(payload) != geometry[1]*geometry[2]*geometry[3] or not any(payload):
        raise RuntimeError("Installed sound design output is empty or silent")
    return digest


def _check_pack_smoke(created, verified, selected, archive, destination):
    """Cross-check the original library report with ZIP and WAV bytes."""
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if (created.get("archive_sha256") != digest
            or verified.get("archive_sha256") != digest
            or selected.get("archive_sha256") != digest
            or verified.get("replayed_assets") != 2
            or verified.get("recipe_replay_verified") is not True
            or verified.get("signature_verified") is not False):
        raise RuntimeError("Installed original sound pack verification mismatch")
    manifest = verified["manifest"]
    if (manifest.get("asset_count") != 2
            or manifest.get("total_frames") != 4800
            or manifest.get("model_inference") is not False
            or manifest.get("recording_sources_used") is not False
            or created.get("manifest") != manifest):
        raise RuntimeError("Installed original sound pack provenance mismatch")
    expected_names = ["manifest.json", "audio/rain-001.wav", "audio/rain-002.wav"]
    with zipfile.ZipFile(archive, "r") as zipped:
        if zipped.namelist() != expected_names:
            raise RuntimeError("Installed sound pack member mismatch")
        embedded = zipped.read("manifest.json")
        if (json.loads(embedded) != manifest
                or hashlib.sha256(embedded).hexdigest() != verified.get("manifest_sha256")):
            raise RuntimeError("Installed sound pack manifest mismatch")
        for record in manifest["assets"]:
            member = zipped.read(record["member"])
            if (hashlib.sha256(member).hexdigest() != record.get("wav_sha256")
                    or len(member) != record.get("bytes")):
                raise RuntimeError("Installed sound pack asset checksum mismatch")
        if destination.read_bytes() != zipped.read(expected_names[-1]):
            raise RuntimeError("Installed selected WAV differs from archive")
    if selected.get("asset") != manifest["assets"][1]:
        raise RuntimeError("Installed selected asset receipt mismatch")
    return _check_generated_pcm(selected, destination, (24000, 1, 2, 2400))


def _smoke_sound_design(console, work, env, source, source_digest):
    """Exercise new primary CLI workflows in each clean source installation.

    All generated files live in the disposable install workspace, never
    in the committed source archive. Exact hashes are compared within
    one environment; different numerical dependencies may differ.
    """
    def run(*args):
        return json.loads(_run([str(console), *map(str, args)], cwd=work, env=env).stdout)

    grains = work / "granular.wav"
    granular = run("granular", "--input", source, "--expected-sha256", source_digest,
                   "--frames", 2400, "--grain-frames", 128, "--hop-frames", 64,
                   "--stereo-spread", ".5", "--seed", 12, "--output", grains)
    grain_digest = _check_generated_pcm(granular, grains, (24000, 2, 2, 2400))
    if granular["source"].get("source_sha256") != source_digest:
        raise RuntimeError("Installed granular source digest mismatch")

    contact_path = work / "contact.wav"
    contact = run("contact-sound", "--material", "wood", "--excitation", "tap",
                  "--count", 2, "--event-ms", 80, "--interval-ms", 150,
                  "--sample-rate", 8000, "--output", contact_path)
    contact_digest = _check_generated_pcm(contact, contact_path, (8000, 2, 2, 1840))
    if len(contact["render"]["event_reports"]) != 2:
        raise RuntimeError("Installed contact event count mismatch")

    plan = work / "spatial-plan.json"
    plan.write_text(json.dumps({
        "sample_rate": 24000,
        "placements": [{"source_id": "rain", "trajectory": {
            "keys": [{"frame": 0, "position": {"x": 0, "y": 4}}]}}],
        "receiver": {"spacing_m": 0, "sound_speed": 320},
    }), encoding="utf-8")
    spatial_path = work / "spatial.wav"
    spatial = run("spatial-scene", "--plan", plan, "--source", f"rain={source}",
                  "--source-sha256", f"rain={source_digest}", "--output", spatial_path)
    spatial_digest = _check_generated_pcm(spatial, spatial_path, (24000, 2, 2, 12300))
    if (spatial["sources"]["rain"].get("source_sha256") != source_digest
            or any(p["truncated"] for p in spatial["render"]["paths"])):
        raise RuntimeError("Installed spatial propagation receipt mismatch")

    master_path = work / "master.wav"
    master = run("wav-master", grains, "--peak-only", "--peak-ceiling-dbfs", -3,
                 "--expected-sha256", grain_digest, "--output", master_path)
    master_digest = _check_generated_pcm(master, master_path, (24000, 2, 2, 2400))
    if (master.get("source_sha256") != grain_digest
            or master.get("encoded_peak_reverified") is not True
            or master["encoded_measurement"]["peak"]["estimated_peak"] > 10**(-3/20)):
        raise RuntimeError("Installed mastering encoded-peak verification mismatch")

    pack_plan = work / "pack-plan.json"
    pack_plan.write_text(json.dumps({
        "pack_id": "smoke-library", "seed": 7,
        "recipes": [{"asset_id": "rain", "frames": 2400, "variants": 2,
                     "tags": ["rain"]}],
    }), encoding="utf-8")
    pack_path = work / "library.zip"
    preview = run("sound-pack-plan", "--plan", pack_plan)
    if preview.get("asset_count") != 2 or preview.get("total_frames") != 4800:
        raise RuntimeError("Installed sound pack planning mismatch")
    created = run("sound-pack", "--plan", pack_plan, "--output", pack_path)
    verified = run("sound-pack-verify", pack_path, "--expected-sha256",
                   created["archive_sha256"], "--replay")
    listing = run("sound-pack-list", pack_path, "--tag", "rain")
    if listing.get("matched_assets") != 2:
        raise RuntimeError("Installed sound pack filtering mismatch")
    selected_path = work / "selected.wav"
    selected = run("sound-pack-select", pack_path, "--asset", "rain", "--variant", 2,
                   "--expected-sha256", created["archive_sha256"], "--output", selected_path)
    selected_digest = _check_pack_smoke(created, verified, selected, pack_path, selected_path)
    return {
        "granular_wav_sha256": grain_digest, "contact_wav_sha256": contact_digest,
        "spatial_wav_sha256": spatial_digest, "master_wav_sha256": master_digest,
        "sound_pack_sha256": created["archive_sha256"],
        "selected_wav_sha256": selected_digest, "sound_pack_replayed_assets": 2,
        "verified_installed_sound_design": True,
    }


def _check_spectrum_smoke(report: dict, digest: str, destination: Path) -> None:
    """Check installed FFT report independently against actual PCM16 bytes."""
    import struct
    spectral = report["spectrum"]
    if (report.get("source_verified") is not True
            or report.get("source_sha256") != digest
            or spectral.get("frames") != 12000
            or spectral.get("fft_frames") != 256
            or spectral.get("channels") != 1
            or spectral.get("complete_blocks") != 46
            or spectral.get("padded_tail_frames") != 224):
        raise RuntimeError("Installed spectrum geometry or provenance mismatch")
    with wave.open(str(destination), "rb") as source:
        data = source.readframes(source.getnframes())
    expected = math.fsum(
        (sample / 32768.0) ** 2 for (sample,) in struct.iter_unpack("<h", data)
    ) / 12000
    channel = spectral["channel_spectra"][0]
    bins = channel["bin_mean_square"]
    if (len(bins) != 129 or any(not math.isfinite(x) or x < 0 for x in bins)
            or not math.isclose(math.fsum(bins), expected, rel_tol=1e-8, abs_tol=1e-12)
            or not math.isclose(channel["mean_square"], expected,
                                rel_tol=1e-8, abs_tol=1e-12)):
        raise RuntimeError("Installed spectrum failed independent PCM energy check")


def _check_demo_smoke(report: dict, destination: Path, version: str) -> None:
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    inspection = report["inspection"]
    if (report.get("software_version") != version
            or report.get("engine_id") != "procedural-sfx-v0"
            or report.get("seed") != 42
            or report.get("duration_seconds") != 12.0
            or report.get("original_model_inference") is not False
            or report.get("third_party_recordings_used") is not False
            or report.get("wav_sha256") != digest
            or report.get("bytes_written") != 576044
            or inspection.get("source_sha256") != digest
            or inspection.get("source_verified") is not True
            or inspection.get("signal_health", {}).get("clipping_fraction") != 0):
        raise RuntimeError("Installed demo provenance or signal contract mismatch")
    with wave.open(str(destination), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(),
                source.getnframes()) != (24000, 1, 2, 288000):
            raise RuntimeError("Installed demo WAV geometry mismatch")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="Install and run the source candidate in a fresh venv")
    parser.add_argument("--expected-version", help="Require an exact numeric software version")
    args = parser.parse_args()
    candidate = resolve_candidate(args.expected_version)
    with tempfile.TemporaryDirectory(prefix="sonumio-source-gate-") as folder:
        work = Path(folder).resolve()
        archive = work / f"sonumio-{candidate.tag}-source.tar.gz"
        _make_archive(archive, candidate)
        result = _audit_archive(archive, candidate)
        zipped = work / f"sonumio-{candidate.tag}-source.zip"
        _make_archive(zipped, candidate, format="zip")
        zip_result = _audit_archive(zipped, candidate, format="zip")
        if (result['source_files'], result['source_bytes']) != (zip_result['source_files'], zip_result['source_bytes']):
            raise RuntimeError('Source archive formats disagree')
        result.update({'zip_archive_sha256': zip_result['archive_sha256'],
                       'both_source_formats_verified': True})
        if args.smoke:
            tar_smoke = _smoke_archive(archive, work / "tar-smoke", candidate)
            zip_smoke = _smoke_archive(zipped, work / "zip-smoke", candidate)
            result.update(tar_smoke)
            result["archive_install_smokes"] = {
                "tar.gz": tar_smoke, "zip": zip_smoke,
            }
            result["both_source_formats_installed"] = True
            # Dependency resolution is independent in each environment.
            # Do not assert identical hashes across different NumPy versions.
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
