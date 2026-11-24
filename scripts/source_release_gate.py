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
import os
import re
import stat
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import wave
import zipfile

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = frozenset({
    "pyproject.toml", "README.md", "LICENSE",
    "src/sonumio/__init__.py", "src/sonumio/cli.py",
    "src/sonumio/api.py", "src/sonumio/engines/procedural.py",
    "web/package.json", "scripts/source_release_gate.py",
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
})
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
                or any(":" in part or part.endswith((" ", ".")) for part in path.parts)):
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
        "canonical_blob_hashes_verified": True,
        "published": False, "binary_or_model_artifacts": 0,
    }


def _smoke_archive(archive: Path, work: Path, candidate: SourceCandidate) -> dict[str, object]:
    version = candidate.version
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
    return {
        "installed_version": reported, "source_install": "isolated non-editable",
        "generated_wav_frames": 12_000, "uploaded_artifacts": 0,
    }


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
            result.update(_smoke_archive(archive, work, candidate))
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
