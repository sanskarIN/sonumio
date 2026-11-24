# Sonumio v0.0.2 source-only candidate

Status: preparation only. The current published source pre-release is
v0.0.1. This checklist does not authorize creating a tag, publishing a
release, uploading model weights, or claiming a trained Sonumio-1 model.

## Scope and integration

- Merge only reviewed, passing research changes intended for v0.0.2.
- The reviewed-corpus readiness and multiscale waveform research are
  already integrated. Complete verification of the transform-codec,
  device-aware training and reversible-waveform research before deciding
  their inclusion.
- Preserve failed research findings in the record. Passing a regression
  suite is not equivalent to passing a model quality gate.
- Do not introduce new public model identifiers for experimental weights.
- Keep the existing procedural engine accurately identified as DSP.

## Prepare an exact candidate

Once the scope is frozen, update the package version consistently in
`pyproject.toml`, `src/sonumio/__init__.py` and `web/package.json`.
Review any other version assertions, installation examples and user-facing
metadata. Keep historical release references intact.

The reusable source gate reads the version from the committed Python
package metadata and pins HEAD to one full Git object ID before exporting.
It validates the same commit's file tree and every blob hash, then checks
the Python module and private web package versions. Duplicate or
case-colliding archive paths and nonregular entries are rejected.
The manual CI dispatch requires the intended numeric package version.
No source archive or generated audio is uploaded by the gate.

Run from the assembled, committed candidate:

```bash
python scripts/source_release_gate.py --expected-version 0.0.2 --smoke
```

The command deliberately fails on a current v0.0.1 checkout. Until the
version bump is reviewed, use the existing version (or omit the optional
expected-version argument for ordinary PR checks).

## Release acceptance

Require successful source-archive build and isolated installation on the
Linux and Windows matrix, general and relevant research tests, and any
affected web/CLI/API checks. Verify the CLI reports the intended version
and generates a valid short PCM WAV via the procedural engine. Review
the archive exclusions; this is a bounded source-artifact audit, not a
comprehensive secret scanner.

Record the exact main commit and workflow evidence. Present a short
source-only release note and request explicit owner approval tied to that
commit. Only after approval may the v0.0.2 GitHub pre-release be published.
No binary installers, generated audio or model weights belong in it.
