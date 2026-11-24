# Reviewed corpus readiness

The rights-gated WAV loader checks declarations, actual bytes, recording
identities and cross-split overlap. A valid manifest is not necessarily a
sufficiently diverse or usable training corpus. This separate diagnostic
measures the actual resampled model-input clips before expensive training.

The loader retains per-clip asset, recording group, language, manifest line,
and crop index from its original validated manifest snapshot. The audit
reports aggregate counts only, with no paths, captions or recording IDs.
It neither verifies legal rights independently nor detects voices.

## Readiness policy

Defaults require at least two eligible recording groups per source category
in training, and one per category in validation. The optional test split is
reported but is not used for training sampling or the readiness gate.

A crop is rejected from the eligible set when centered RMS is below -55 dBFS,
fewer than 10% of its 4 ms frames meet that activity threshold, more than 1%
of samples have absolute amplitude at least 0.999, or absolute DC exceeds
0.05. Subsequent byte-identical resampled crops within the same split are
also excluded. These are conservative engineering heuristics; quiet,
transient or highly dynamic recordings can require manual review and a
different explicitly chosen policy. They are not judgments of sound quality
or semantic label correctness. Reports count all rejection reasons; a clip
can have multiple reasons.

Coverage is counted by declared independent recording groups, not by number
of clips. Categories and languages are reported separately, but language
coverage is not a default gate. The audit does not edit the corpus or alter
existing training defaults.

## Use

```bash
python -m research.corpus_readiness manifest.jsonl /path/to/reviewed/audio
python -m research.corpus_readiness manifest.jsonl /path/to/reviewed/audio --require-ready --min-train-groups 3
```

The second command exits with code 2 if the coverage policy fails. Malformed
or unverified audio fails in the underlying loader. JSON is written to stdout.
No audio, identifiers or weights are written by the audit.

For Python callers, use the same in-memory loaded splits with
`audit_loaded_splits` and `GroupBalancedSampler`. Sampling requires a passed
gate and verifies that the eligible training audio/provenance still matches
the audited index fingerprint at construction. Each cycle contains one
example from every category in shuffled order. Within the category, sample
uniformly among eligible groups, then assets, then crops. Sampling is with
replacement and state is preserved across calls. This prevents a group
from dominating simply because it has more crops, but does not establish
that declared groups are genuinely independent. Keep loaded arrays unchanged
after constructing the sampler; it is not a tamper-proof security boundary.

The class-conditioned waveform DDPM now has an explicit opt-in mode:

```bash
python -m research.wave_diffusion train --manifest manifest.jsonl --audio-root /path/to/reviewed/audio --group-balanced --steps 50
```

It requires the default readiness gate to pass before any optimizer update,
samples only eligible training crops, and evaluates eligible validation
crops in bounded minibatches. The report records available versus eligible
example counts, the aggregate readiness audit, eligible split fingerprints
and the deterministic sampler index trace. No test crop is used for
optimization or held-out validation. The training checkpoint, if requested,
remains local and is not an original Sonumio-1 model. The ordinary legacy
sampling path is unchanged when the option is absent. The filtered validation
metric is not directly comparable to an unfiltered evaluation with different
examples. Do not claim data readiness guarantees model quality.
