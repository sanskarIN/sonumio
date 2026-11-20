# Ambient sine version-drift investigation and isolated alternatives

**17 November 2026 · DSP numerical research, NOT generative model training**

## Verified cause of current NumPy 1.26 divergence

The [controlled single-runner stage-difference job #44077066392](https://github.com/sanskarIN/sonumio/actions/runs/44077066392)
installed NumPy 1.26.4, generated first-party stage fingerprints,
reinstalled NumPy 2.4.6 on the **same host** and regenerated those
fingerprints. Its host reported Intel Xeon 6975P-C. This avoids the
different-CPU confounding in the first NumPy compatibility matrix.

Out of 40 first-party source/seed diagnostic recordings, **28
complete recordings had identical stage hashes** and **12
differed**. All twelve differing cases included the ambient
sound layer (ambient alone, rain+ambient or ocean+ambient
over the four original seeds).

The stage trace is now explicitly ordered by actual execution,
not alphabetically sorted JSON object keys. Before this fix,
an earlier report could call the final crop the “first
difference” because `json.dumps(sort_keys=True)` had
reordered stage-map keys. The underlying per-stage hashes
were still useful; the label **was not reliable until
execution order was stored separately**.

The corrected [comparison logs](https://github.com/sanskarIN/sonumio/actions/runs/44076922419)
show that the first actual divergent DSP stage in each
of those twelve recordings is **`layer_drone`**. All
measured rain, thunder, wind, water and forest layer
aggregates matched between the two versions. The 24-kHz
time grid, fade ramp and FIR kernel also matched.

The focused original-expression probe
`research/numpy_ambient_probe.py` verified exact
reconstruction of the unchanged `dsp._drone`
implementation on both versions.
Its [same-runner logs](https://github.com/sanskarIN/sonumio/actions/runs/44077066392)
pinpointed the earliest expression mismatch to
**`sin_110`**, with differences also recorded at
**`sin_220`**, the scaled 220-Hz component, the
combined oscillator wave and the final original
drone output. The original float32 **phase inputs**
matched across versions.

This is reproducible evidence that the original
NumPy float32 sine evaluation contributed to the
observed NumPy 1.26.4 versus 2.4.6 complete-DSP
fingerprint difference. It is **not** evidence
that changing only the sine function is sufficient
to guarantee universal future hardware portability.

## Separately implemented oscillator candidates

`research/ambient_sine_candidate.py` now implements
two **candidate-only** alternatives:

- `numpy_float64`: promote the unchanged float32
  phase angles to float64, evaluate NumPy sine,
  then round sine results once to float32.
- `python_math_sin`: promote the same float32
  phase values to Python float (binary64),
  evaluate scalar `math.sin`, then round sine
  results once to float32.

Both preserve the original synth's envelope
constants, gains and arithmetic combination order.
The original `sonumio.engines.procedural._drone`
is never modified, patched or called to
produce the candidate's output. It remains the
independent numerical reference for comparison.

Each candidate is checked for finite 36,000-frame
float32 output, bounded input, exact SHA256,
maximum/RMS error against the unchanged original,
and timing. Only **JSON numerical diagnostics**
are written. The separate same-runner workflow
now probes both candidates under NumPy 1.26.4
and 2.4.6 on the same CPU, while the independent
four-runner matrix records their candidate
hashes at the CI NumPy version.

The new hypotheses should not be considered
cross-version reproducible until **actual
completed CI logs show identical candidate
SHA256s**, and cross-CPU matching in the same
experiment is also verified. Output from one
seed/one 1.5-second ambient source is not
universal numerical reproducibility.

## Research commands

```bash
python -m research.numpy_ambient_probe \
  --output checkpoints/private-research/ambient-original.json
python -m research.ambient_sine_candidate \
  --output checkpoints/private-research/ambient-alternatives.json
python -m pytest -q tests/test_numpy_ambient_probe.py \
  tests/test_ambient_sine_candidate.py
```

A new full-corpus versioned candidate for corrected
ambient synthesis, if justified, must still pass:
full source SHA controls, 84 original and 105
held-out sound cases, exact cross-host and
cross-NumPy final 8-kHz crops, anti-leakage
checks, independent performance and memory
benchmarks, human listening and licensed
real-audio evaluations. No historical data
fingerprints should be silently overwritten.

**Draft PR #1 stays unmerged. No releases,
pre-releases, tags, WAV files, public model
weights or neural training data were published
or modified by these oscillator experiments.**
