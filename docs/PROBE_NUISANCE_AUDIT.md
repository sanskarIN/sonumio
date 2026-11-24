# Frozen source-probe nuisance audit

The open-set source probe is an experimental numerical similarity measure.
An apparently failed audio reconstruction may reflect poor source
preservation, sensitivity of the probe to a nuisance transformation, or
both. This experiment measures that distinction without altering the
existing probe or fitting thresholds on the new audit panel.

## Independent recordings

Use four fresh recording seeds for each of the six synthetic source
categories (24 recordings) and four distinct seeds for each of the seven
noise families (28 recordings). Check seed and exact waveform hash overlap
against reference fit/calibration/holdout and all synthetic training and
validation combinations. The fitted probe is reused unchanged for all
controls; no audit result selects a new probe.

## Controls

- Half and quarter gain, and polarity inversion: decision-invariance
  diagnostics. These operations do not change the source recipe.
- Three bounded circular shifts: temporal-alignment stress tests. Wrapping
  can introduce a seam; these are not assumed perceptually equivalent.
- Symmetric 16-, 8- and 4-bit amplitude grids, two windowed-sinc low-pass
  filters and clipping: controlled distortions. Output limiting of the FIR
  control is explicit; no gain normalization is applied.

Every control receives the same original waveform. Counts are paired
observations, not 624 independent source/noise recordings. Report correct
accepted sources, new/lost acceptances, label changes, raw nearest-class
changes, false accepts among noise controls, and per-category transition
counts. A source that was originally rejected is not silently treated as
a correct baseline. Silence is explicitly recorded, and undefined
zero-energy numerical ratios are represented by null.

## Execution

```bash
python -m research.probe_nuisance_audit
```

Optionally pass `--controls gain_half polarity_flip shift_128`.
The report contains numeric metrics and seed identifiers, not audio or
weights. Results cannot certify perceptual quality, codec transparency,
general source identity, or Sonumio-1 readiness. A failure of scale or
polarity invariance is an evaluator limitation worth investigating; a
degradation failure is not by itself proof of poor perceptual quality.
