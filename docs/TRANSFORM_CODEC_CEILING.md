# Transform codec reconstruction ceiling

The prior neural codec experiment (PR #43, two initializations, 64 codec
updates each) recovered zero accepted source matches even when decoding
encoded held-out original audio. Channel calibration improved latent moments
but could not recover detail the codec discarded. This motivates a
non-neural reconstruction benchmark at comparable scalar budgets.

Five representations are compared: contiguous orthonormal DCT-II with
low-frequency coefficients, the same DCT with a fixed training-ranked
coefficient mask, block PCA fitted on training recordings, sine-windowed
lapped MDCT with low-frequency coefficients, and lapped MDCT with a fixed
training-ranked mask. MDCT uses a 128-sample window and a 64-sample hop, with
one half-window of zero padding at each boundary; overlap-add cancels
time-domain aliasing at full coefficient count. Full transforms must
reconstruct all samples to numerical precision, including boundary impulses.

The study tests 8, 16, 32 and 64 retained coefficients per transform frame.
The 16-coefficient DCT/PCA representation has 2048 scalars per 8192 samples.
MDCT uses 129 rather than 128 frames, so its corresponding count is 2064.
These are floating-point scalar counts, NOT bitrate or compression of a
quantized bitstream. PCA basis/center and coefficient masks are shared
across every recording and fitted on 36 first-party training originals.
PCA uses equal category weighting in covariance estimation; ranked frequency
masks use equal category weighting of relative coefficient energy.

Validation uses 12 separate source recordings; the independent probe
holdout contains 18 additional recordings. Exact hashes and seed
separation are checked. The open-set source probe is frozen before codec
evaluation; if its reference gate fails, source scores remain descriptive.
Reconstruction measures include MAE, MSE, relative squared error, SNR
with a disclosed error floor, waveform cosine, amplitude range, RMS and
DC bias. Source decisions operate on clipped float32 reconstructions while
numerical error is measured before clipping. Every tested configuration
is reported; selecting a winner using these holdouts requires a subsequent
untouched confirmation panel.

Run locally with research dependencies:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m research.transform_codec_study
```

No model training, quantization, checkpoint, waveform publication or
Sonumio-1 readiness claim is part of this experiment. The purpose is to
separate representation loss from failure of the neural denoising model
and obtain a reproducible reconstruction reference.


## Quantized payload and validation-only selection

The optional quantized study adds a symmetric uniform quantizer per retained
coefficient coordinate. Ranges are fitted only on the training panel, using
maximum absolute coefficient value with a 1e-7 floor. Values outside the
range saturate; their fraction is reported. Rounding is to nearest with
ties to even. Supported widths are 4, 8, 12 and 16 bits. A code uses
two's-complement representation with the most negative code reserved.
Codes are serialized in row-major order, least-significant bit first,
into canonical bytes; unused final padding bits must be zero. Decoding
validates shape, payload length, padding and reserved codes before
dequantizing. No pickle or general object deserialization is involved.

The payload requires the transform, dimensions, shared coordinate mask or
PCA basis, quantizer ranges and bit width out of band. It is not a complete
audio file, and payload bits per second do not include these shared
parameters or container overhead. Each recording is packed independently,
and the report measures actual payload bytes, not a hypothetical entropy
estimate.

The validation grid evaluates five methods, three coefficient budgets
(8/16/32) and four bit widths, for 60 configurations. At each predeclared
payload cap (8/16/32/64 kb/s), candidates are selected first by ascending
false noise acceptance count on 21 independent validation noise controls,
then descending correct accepted validation source count and represented
category count, then ascending mean relative squared reconstruction error,
payload rate and stable configuration ID. If the independent probe fails
its reference gate or accepts any original validation noise, the selection
uses only numerical distortion, rate and ID. All candidates are still
reported, even if no candidate achieves zero reconstructed noise acceptance.

All choices are fixed before evaluating any selected configuration on
24 new confirmation recordings (four previously unused seeds per source).
The same selected transform without quantization is scored alongside its
quantized counterpart to separate loss of discarded coordinates from
quantization loss. Selection and confirmation panels have distinct exact
waveform hashes and recording seeds from the fit panel and probe references.
The confirmation panel is still from the same synthetic DSP recipes, not
a test of real-world sound generalization.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m research.quantized_transform_study
```

The study prints aggregate and per-recording numeric diagnostics in JSON,
without saving audio, coefficient payload files or model weights.
No production bitrate, perceptual codec quality or model readiness is claimed.


## Continuous transform latent diffusion bridge

The separate frozen-transform study uses exactly 16 coefficients per
64-sample block, giving the existing 16 x 128 latent layout. It considers
DCT low-frequency, ranked DCT and block PCA, rather than the lapped
129-frame MDCT representation. This is a continuous float32 latent model,
not diffusion over the quantized payload codes.

The transform basis remains frozen, with zero trainable codec parameters.
A second adapter applies per-channel train-only means and standard
deviations with a 0.001 scale floor. Both adapters reconstruct the same
projection (within numerical precision) and the decoder does not silently
clip waveform amplitude. The existing DDIM sampler can use the fixed
decoder without changing production APIs or checkpoint formats.

Selection uses only the 12 source validation recordings. It ranks the
three candidates by correct accepted source count, represented category
count, mean relative squared reconstruction error and stable method name.
If the independent probe gate fails, source scores are not used for
selection. The selected method then undergoes a new confirmation panel
of 24 synthetic sources and 28 artificial noise controls, with seeds
separate from the quantization study and the fixed probe references.

The predeclared training gate requires:

- The independent reference probe gate passes.
- At least 16 of 24 original and reconstructed confirmation source
  recordings are correctly accepted, and reconstruction preserves at
  least one correct accepted recording per category.
- None of the 28 original or reconstructed artificial noise controls is
  accepted as a source.

A failed gate is reported and **no denoiser optimization or generation**
is performed in the study. These are conservative experiment heuristics,
not a general-purpose audio-quality certification. The regression suite
separately executes a tiny optimizer/DDIM smoke test on fabricated
numerical inputs to verify the adapter even when the gate fails.

If the gate passes, native and calibrated arms share identical denoiser
initialization and stochastic minibatch inputs. The original 42 unique
training recordings and 84 bilingual caption rows are used, with the
fixed validation corpus for teacher-forced diagnostics. Training and
transform fit may share originals, and validation and method selection
may share originals; the reconstruction confirmation and probe reference
domains are disjoint. Held-out diagnostics report both latent velocity
MSE and decoded waveform MAE at three timesteps; MSE across different
latent scales is not directly comparable. Real DDIM generation is
evaluated under matched, rotated-label and null controls using paired
sampling seeds, with pre-clipping numerical diagnostics.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m research.frozen_transform_study --seed 17 --train-steps 96 --evaluations 8 --sampling-seeds 301 302
```

The CPU workflow runs the same protocol for seeds 17 and 29. A green
workflow verifies execution and fail-closed controls, not that the
reconstruction gate passed or that a useful model exists. No audio,
checkpoint, coefficient payload or model weights are saved.


## Codec-induced noise confusion

The quantized selection protocol v2 adds seven artificial noise families
with three fresh seeds each for validation and four additional seeds each
for confirmation. These 49 recordings do not enter transform fitting,
quantizer calibration or probe fitting. Exact hashes and seed isolation
are checked against every source and fixed probe panel. The same validation
noises are used for every candidate; they are not independent new samples
per configuration.

For each codec, the frozen probe evaluates the original noise and its
reconstructed waveform. Reports distinguish new false acceptance (an
original abstention becomes an accepted category), removed false acceptance,
and changes between accepted categories. Aggregate and family-level counts
are checked for consistency. After selection is frozen, the same checks
are applied to the selected quantized transform and its unquantized
counterpart on 28 new noise recordings. A narrow confirmation noise
screen passes only if the reference/validation probe controls were usable
and neither original nor reconstructed confirmation noise is accepted.
A passing screen is not a claim about real-world false-positive rates,
audio quality, semantic accuracy, or readiness for release.
