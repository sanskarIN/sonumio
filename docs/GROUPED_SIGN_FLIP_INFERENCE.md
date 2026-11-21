# Exact small-group paired inference (research only)

Sonumio numerical audio evaluations compare source-aligned candidates using
the **recording/source group**, not a clip, as the independent unit. Recording
several crops from one source does not multiply the number of independent
observations. These tests are research diagnostics, not human listening
scores, real-world fidelity evidence, or model-release authorization.

## Statistical behavior

\`paired_sign_flip_test\` aggregates each group's candidate-minus-reference
scores into one mean, then tests all sign assignments under the paired
exchangeability null.

- **2–16 groups**: enumerate every \`2**groups\` assignment. The reported
  \`method="exact"\` and \`permutations\` give the effective count. This is
  independent of random seed and requested Monte Carlo budget, although
  both inputs remain validated for reproducible experimental metadata.
- **17+ groups**: enumerate no more than the requested \`100..100000\` random
  sign assignments using the explicit seed. The reported
  \`method="monte_carlo"\` uses a plus-one correction to avoid fabricated
  zero p-values. Simulation error remains.
- **All paths**: preserve one-sided and two-sided direction choices, retain
  source-group aggregation, and count tied/extreme effects with the same
  bounded floating-point comparison tolerance.

The exact two-sided smallest p-value with consistent nonzero effects is
\`2 / (2**N)\`. With four independent groups, the minimum is **0.125**,
even if each group favors the candidate. Adding correlated clips cannot
improve this resolution. Predeclare the experimental hypothesis before
looking at acoustic results.

The report schemas are versioned \`sonumio-grouped-paired-test-v2\` and
\`sonumio-group-level-audio-metric-effect-v2\` because switching from
stochastic to exhaustive inference changes reported p-values. The
\`requested_permutations\` field records the caller's budget; \`permutations\`
counts actually evaluated assignments. Existing \`v1\` reports must not be
silently compared with \`v2\` numbers.

## Limits

The sign-flip test assumes independent, exchangeable **source groups**.
Exactness is conditional on those assumptions; the test cannot detect
unrecognized duplicates, hidden leakage, or nonrandom sampling. Bootstrap
confidence bounds still use their distinct seeded resampling procedure.
Multiple hypotheses require planned multiplicity correction (the existing
Holm helper); these reports do not declare a perceptual winner.

To run the focused tests from source:

\`\`\`bash
python -m pytest -q tests/test_audio_eval_exact_sign_flip.py tests/test_audio_eval_statistics.py
\`\`\`
