# v0.0.4 source-only target — 1 December 2026

Planning target, not a release approval. The latest published release
at planning time is v0.0.2; v0.0.3 is an optional intermediate source
candidate, not a published or authorized release.

## Scope

Prioritize the reviewed archive, streamed DSP and neural research
already implemented. Integrate the waveform checkpoint / conditioning
stack (#63–#66) and crop codec / latent training stack (#67–#69) only
after each candidate passes its relevant checks on the actual integration
base. Keep incomplete features out of the release rather than masking
errors or weakening test coverage.

## Sequence

- 25–26 November: resolve integration conflicts and synthetic source
  leakage; verify dedicated CPU research tests.
- 27–28 November: validate checkpoint and private corpus training,
  confirm no private artifacts appear in source exports.
- 29 November: freeze feature scope; inspect reproducibility and
  compatibility, CLI/API/web changes and known limitations.
- 30 November: align Python and private web metadata to 0.0.4 on the
  intended candidate, verify all required post-merge checks and the
  exact full commit.
- 1 December: owner review and explicit approval of the source-only
  candidate. Defer if any required gate fails.

## Release gates

- No open blocker from either dependency stack included in the release.
- All applicable general, DSP, corpus, CPU research and web tests pass.
- Dedicated codec tests exercise active source crops, split isolation,
  true optimizer updates and exact completed-stage handoff equivalence.
- Source-only build gate passes all Linux/Windows matrix jobs, both
  archive formats and clean installation with expected version 0.0.4.
- Archive gate rejects known private/binary formats by suffix and
  recognizable content signature, even when mislabeled. This is not
  a comprehensive secret scanner.
- No model weights, private data, generated audio, binary installers
  or model-quality claims are included. Experimental category generation
  is not an open-vocabulary Sonumio-1 model.

Run the existing gate from the exact committed candidate:

```bash
python scripts/source_release_gate.py --expected-version 0.0.4 --smoke
```

Record the full candidate SHA, workflow URLs, matrix results and known
limitations before requesting approval. This plan does not bump package
versions, create tags, publish releases or authorize checkpoints.
