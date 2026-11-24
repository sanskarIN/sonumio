# Open-set synthetic source probe

The earlier six-source spectral probe accepted both white and downsampled
noise as rain. Consequently, a nearest reference or even an accepted reference
match cannot be treated as evidence of prompt grounding. This study adds an
explicit artificial-noise rejection model and evaluates the old and new probes
on the same untouched panel. It is not a real-world sound classifier.

## Reference design

| Domain | Fit | Calibration | Holdout |
| --- | ---: | ---: | ---: |
| Six DSP source categories | 24 | 12 | 18 |
| Seven artificial noise families | 28 | 21 | 28 |

The seven noise families are white, downsampled white, pink, brown, short
moving-average, long moving-average and amplitude-modulated Gaussian noise.
Each has its own fixed recipe. Recording seeds are separate between splits and
from the generator corpus; exact waveform hashes are checked across all panels
and generator train/validation recordings. Noise families sharing a seed
are related controls, not independent population samples.

The feature vector extends the earlier spectral signature with gain-normalized
amplitude quantiles, fourth moment, short-lag autocorrelation, frame-energy
dynamics, spectral flux and flatness. Positive and negative centroids are fit
using fit data only. Distances are RMS feature errors scaled by pooled
within-group variation with a floor of 2 in the chosen feature units.

A waveform is accepted as its nearest source only if it passes both tests:

1. Its distance to the source centroid is at most the source radius, derived
   from positive calibration distances.
2. Its distance to the nearest noise centroid minus its source distance is
   strictly greater than the source-specific threshold. The threshold is
   the maximum of zero and the largest such margin among calibration
   negatives whose nearest source is that category.

This excludes every observed calibration negative by construction. It is not
a guarantee about new noises. The untouched holdout is evaluated once with the
fixed design, without changing thresholds based on its outcomes.

The predeclared gate requires at least 12 of 18 correctly accepted holdout
sources, at least one in each category, and zero accepted holdout noise
controls. If it fails, the optional neural sampling step is skipped; the
report records the failure and category/family breakdown. Passing a small
synthetic holdout is not proof of generalization, safety, or model readiness.

## Phase challenge

The study also creates phase-randomized versions of all 18 source holdout
recordings. Relative full-clip power is preserved, while temporal phase
relationships are disrupted. These are reported separately and are not
automatically labeled wrong: some stationary tonal sounds retain their
identity after phase randomization. No phase challenge enters calibration or
the strict noise gate.

## Run

```bash
python -m pip install -e '.[research,test]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.open_set_source_study
# Request actual DDIM training/sampling only after the fixed gate passes:
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m research.open_set_source_study --train --seed 17 --train-steps 32 --codec-steps 16 --evaluations 8 --sampling-seeds 301 302
```

The CPU workflow tests deterministic controls, data separation, calibration
and fail-closed behavior and runs two configurations with the same reference
panel. Successful CI means the experiment ran correctly; it does not mean the
probe passed the gate or that generated audio is good. Results are numeric
JSON only. No model parameters, audio files or external datasets are published.
