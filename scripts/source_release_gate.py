"""Audit and smoke-test a v0.0.1 source archive without publishing it.

Run: python scripts/source_release_gate.py --smoke
Uses tracked Git source only, then a disposable virtual environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import wave

VERSION = "0.0.1"
TAG = "v0.0.1"
PREFIX = f"sonumio-{TAG}/"
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


def _make_archive(archive: Path) -> None:
    # On Windows, core.autocrlf=true can change bytes INSIDE git archive.
    # GitHub source snapshots must preserve committed blobs exactly.
    _run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
          "archive", "--format=tar.gz", f"--prefix={PREFIX}",
          f"--output={archive}", "HEAD"], cwd=ROOT)
    if not archive.is_file() or archive.stat().st_size == 0:
        raise RuntimeError("Git source archive was not created")


def _audit_archive(archive: Path) -> dict[str, object]:
    files: dict[str, bytes] = {}
    total_size = 0
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            name = member.name
            # tarfile normalizes the root directory name without its final '/'.
            if name == PREFIX.rstrip("/") and member.isdir():
                continue
            if not name.startswith(PREFIX):
                raise RuntimeError("Unexpected source archive prefix")
            relative = name[len(PREFIX):]
            if not relative:
                if not member.isdir():
                    raise RuntimeError("Archive root is not a directory")
                continue
            path = PurePosixPath(relative)
            if (path.is_absolute() or ".." in path.parts
                    or "\\" in relative or not path.parts):
                raise RuntimeError(f"Unsafe archive path: {relative}")
            if any(part in BANNED_PARTS for part in path.parts):
                raise RuntimeError(f"Private or generated directory: {relative}")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError(f"Non-regular archive entry: {relative}")
            if (path.suffix.lower() in BANNED_SUFFIXES
                    or path.name.lower() in {".env", ".npmrc", "id_rsa", "id_ed25519"}):
                raise RuntimeError(f"Binary or sensitive artifact: {relative}")
            if relative in files:
                raise RuntimeError(f"Duplicate source entry: {relative}")
            if member.size > MAX_FILE_BYTES:
                raise RuntimeError(f"Unexpectedly large source entry: {relative}")
            total_size += member.size
            if total_size > MAX_TOTAL_BYTES or len(files) >= MAX_FILES:
                raise RuntimeError("Source archive exceeds review limits")
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unreadable source entry: {relative}")
            data = stream.read(MAX_FILE_BYTES + 1)
            if len(data) != member.size:
                raise RuntimeError(f"Source archive entry changed: {relative}")
            files[relative] = data
    # Verify each emitted source file against the canonical Git object size.
    # This detects CRLF conversion and other host-dependent export filters.
    tracked: dict[str, int] = {}
    tree_output = _run(["git", "ls-tree", "-r", "-l", "HEAD"], cwd=ROOT).stdout
    for line in tree_output.splitlines():
        header, separator, filename = line.partition("\t")
        pieces = header.split()
        if separator != "\t" or len(pieces) != 4 or pieces[1] != "blob":
            raise RuntimeError("Unexpected Git tree entry in source audit")
        tracked[filename] = int(pieces[3])
    if files.keys() != tracked.keys():
        raise RuntimeError("Source archive paths differ from the committed tree")
    for filename, data in files.items():
        if len(data) != tracked[filename]:
            raise RuntimeError(f"Noncanonical exported source bytes: {filename}")
    missing = sorted(REQUIRED.difference(files))
    if missing:
        raise RuntimeError(f"Missing buildable-source entries: {missing}")
    project = tomllib.loads(files["pyproject.toml"].decode("utf-8"))["project"]
    if project.get("name") != "sonumio" or project.get("version") != VERSION:
        raise RuntimeError("Python package version is not exactly 0.0.1")
    if project.get("license") != "Apache-2.0":
        raise RuntimeError("Sonumio source-license metadata changed")
    package_init = files["src/sonumio/__init__.py"].decode("utf-8")
    if f'__version__ = "{VERSION}"' not in package_init:
        raise RuntimeError("Python module version differs from pyproject.toml")
    api_source = files["src/sonumio/api.py"].decode("utf-8")
    if "version=__version__" not in api_source:
        raise RuntimeError("API version must follow the package version")
    web = json.loads(files["web/package.json"])
    if web.get("private") is not True:
        raise RuntimeError("Web package must remain private / unpublished")
    if not any(key.startswith("tests/test_") and key.endswith(".py")
               for key in files):
        raise RuntimeError("Source archive has no tests")
    return {
        "tag_candidate": TAG, "source_files": len(files),
        "source_bytes": total_size,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "canonical_blob_bytes_verified": True,
        "published": False, "binary_or_model_artifacts": 0,
    }


def _smoke_archive(archive: Path, work: Path) -> dict[str, object]:
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
        f"assert version('sonumio') == '{VERSION}'; "
        f"assert sonumio.__version__ == '{VERSION}'; "
        f"assert app.version == '{VERSION}'"
    )
    _run([str(python), "-c", version_check], cwd=work, env=env)
    reported = _run([str(console), "--version"], cwd=work, env=env).stdout.strip()
    if reported != f"sonumio {VERSION}":
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
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sonumio-source-gate-") as folder:
        work = Path(folder).resolve()
        archive = work / f"sonumio-{TAG}-source.tar.gz"
        _make_archive(archive)
        result = _audit_archive(archive)
        if args.smoke:
            result.update(_smoke_archive(archive, work))
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
