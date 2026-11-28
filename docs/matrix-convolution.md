# Matrix convolution reference

This opt-in NumPy DSP engine accepts an impulse response matrix with shape
`(receivers, sources, taps)`. Each receiver is the sum of causal FIR responses
from all source channels. It does not load external impulse files or models.

```python
import numpy as np
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.matrix_convolution import ImpulseMatrix, convolve_matrix

source = make_buffer(np.ones((1, 800), dtype=np.float32) * 0.1, 8000)
kernel = np.zeros((2, 1, 240), dtype=np.float32)
kernel[0, 0, 0] = 0.8
kernel[1, 0, 120] = 0.4
stereo = convolve_matrix(source, ImpulseMatrix(kernel, 8000))
assert stereo.samples.shape == (2, 1039)
```

For streaming use, instantiate `MatrixConvolver`, call `push` on each
nonempty source chunk, and concatenate its returned arrays plus the result of
`finish`. Push emits complete fixed-clock FFT blocks; finish emits the final
partial block and, by default, the remaining causal tail. Use
`finish(include_tail=False)` to omit the tail. Chunk boundaries do not change
the FFT block clock. The streaming output is float64; the offline wrapper
returns the normal immutable float32 AudioBuffer.

Normalization is explicit: none preserves coefficients; receiver_peak uses
the receiver's aggregate L1 gain, while receiver_energy uses its L2 norm.
Silent receivers remain silent. No normalization is applied by default.

Limits include channel count, sample rate, duration, block power-of-two,
impulse taps, estimated FFT working memory and cumulative spectral products.
The engine is not a real-time processing or measured room acoustics claim.
Existing short mono convolution reverb behavior is unchanged.
