# Score-clock audio diagnostics

Measure a rendered canonical stereo PCM16 WAV against the exact musical
clock of its score. Per-bar and section reports include sample peaks,
RMS, DC, centered stereo correlation, mono-fold retention, side-energy
fraction, rail samples, activity and bar-boundary discontinuities. Exact
integer accumulators merge across regions without block-boundary loss.

Hann-window FFT power is summed in six frequency bands. Both channels'
power is combined before computing fractions, avoiding false cancellation
from a mono downmix. Windows align to each bar, and section/global spectra
sum those same windows. These are signal diagnostics, not LUFS, true peak,
broadcast conformance, perceptual quality or proof of origin.

    python -m sonumio.music_render_analysis_cli inspect score.json mix.wav
    python -m sonumio.music_render_analysis_cli compare score.json baseline.wav candidate.wav

Comparison uses identical sample clocks and no time alignment or gain
application. It reports exact integer error sums, relative and absolute
waveform MSE, cosine, RMS ratio and a separate least-squares gain-fit
residual. Silence produces null ratios instead of misleading infinities.
Threshold findings are explicit and configurable; quiet bars can be
intentional. A score/WAV geometry match does not verify provenance.
