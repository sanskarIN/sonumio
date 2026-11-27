# Offline mastering signal measurements and peak guard

The new modules are deterministic NumPy DSP references. They do not
implement a certified ITU-R BS.1770 true-peak meter, calibrated LUFS,
K weighting, a broadcast compliance guarantee, listening tests, or a
trained sound-generation model.

## Finite-sinc intersample estimate

The estimator evaluates the original integer samples and equally spaced
fractional phases (2, 4 or 8 subdivisions) in each source interval. Each
fractional phase uses a normalized finite Kaiser-windowed sinc kernel
with a declared radius. It retains only the halo required for a future
packet, processes a noninteger interval when its context is available,
and flushes the remaining intervals at verified EOF with zero extension.
There is no interpolated interval beyond the last original source frame.

The report exposes sample peak, estimated peak, estimated overshoot,
per-channel locations and signed values. Its work is bounded by the
declared tap-product count, packet size and duration.

## Unweighted gated RMS

A separate accumulator averages channel mean-square values, constructs
exact fixed-clock energy windows, and applies absolute and relative
thresholds in linear energy. The default window is 400 ms and the hop is
100 ms. A source shorter than a window uses one short window. Otherwise
an incomplete trailing window is excluded from gating, and its frames
remain included in overall RMS. Silence is represented by null dB values
rather than fake finite silence loudness.

This is an unweighted signal-level proxy expressed in dBFS, not LUFS.

## Gain staging and limiting

Static mode computes the target gated RMS difference, caps positive gain
by an explicit boost budget, and prioritizes the finite-sinc peak ceiling.
It never changes duration or the sample clock. By default no boost is
permitted. Peak-only mode skips the RMS target.

Lookahead mode optionally applies linked sample-peak limiting. The reverse
gain envelope anticipates peaks with a bounded linear attack slope and
the forward pass smooths release while respecting each frame's safe gain.
A common gain preserves channel ratios. The resulting PCM is remeasured
and, if necessary, uniformly attenuated to satisfy the chosen finite-sinc
estimate. No hard clip or analog ceiling is claimed.

## Verified file workflow

The streaming measurement command reads canonical PCM packets, feeds
both accumulators and reports only after the complete source digest
is verified. The export command takes a bounded verified PCM snapshot,
masters it, encodes integer PCM and measures the decoded encoded output.
Quantization can change an interpolated peak even when the floating
source is within the requested limit. A failed encoded peak can either
be rejected or trigger one conservative attenuation attempt using the
maximum absolute interpolation-kernel sum and a PCM quantization-error
reserve. A second failure rejects publication. Explicit dither remains
enabled throughout both attempts.

```bash
sonumio master-inspect generated.wav
sonumio wav-master generated.wav --output mastered.wav --target-rms-dbfs -18 --max-boost-db 6 --peak-ceiling-dbfs -1
sonumio wav-master generated.wav --output limited.wav --peak-only --mode lookahead --output-bits 24
```

The report distinguishes pre-encoding and encoded measurements, source and
destination SHA256, requested and achieved RMS, gain reductions and
encoding attempts. No-clobber publication is atomic. Source/destination
aliases and digest mismatches fail before changing the destination.

The test suite includes a direct sinc interpolation reference, exact
packet partitions and locations, analytic high-frequency overshoot,
energy-gating reference, limiter ceiling invariants, linked-channel
ratios, deterministic integer output, encoded peak correction, source
integrity failures and local CLI commands.
