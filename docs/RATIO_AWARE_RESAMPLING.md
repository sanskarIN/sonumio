# Ratio-aware offline resampling

The source-level NumPy resampler uses a centered, DC-normalized windowed sinc.
For a target/source rate ratio below one, the cutoff is reduced to that ratio
and the window radius is expanded to `lobes / cutoff` input samples. This
preserves the requested number of sinc lobes at extreme supported ratios
(e.g. 192 kHz to 8 kHz). Finite support still means a transition band; no
claim of perfect bandlimiting or perceptual transparency is made.

The requested chunk size is an upper bound: an internal 262,144-cell cap
limits each two-dimensional kernel matrix, independent of the clock ratio.
Several matrices and buffers coexist, so this is not a bound on process RSS.
Linear interpolation similarly processes output in bounded batches and still
requires explicit aliasing permission for downsampling. The float64 working
input allocation is validated before widening, and values not representable
in finite float32 are rejected rather than silently returning infinities.

The tests cover interior tone passband/stopband behavior at 24:1 downsampling,
DC at clip boundaries, channel independence, fractional sample clocks,
batch partition independence, memory preflight, and overflow rejection.
These are synthetic numerical tests, not listening tests or training evidence.
The change is confined to the independent offline audio core.
