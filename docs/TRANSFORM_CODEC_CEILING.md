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
