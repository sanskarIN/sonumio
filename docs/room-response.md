# Static shoebox room response

The local room-response reference builds receiver/source impulse matrices
from a rectangular three-dimensional room. Coordinates are metres and points
must lie strictly inside the room. Each of six walls has a separate amplitude
reflection coefficient. Image paths of order zero through four are supported.

```python
import numpy as np
from sonumio.audio_core.contracts import make_buffer
from sonumio.audio_core.room_response import (
    Point3D, ShoeboxRoom, RoomConfig, render_room,
)

audio = make_buffer(np.array([1., 0., 0.], dtype=np.float32), 8000)
room = ShoeboxRoom(5, 4, 3)
source = (Point3D(1, 2, 1),)
receivers = (Point3D(3, 1.9, 1), Point3D(3, 2.1, 1))
result = render_room(
    audio, room, source, receivers,
    RoomConfig(sample_rate=8000, reflection_order=2),
)
assert result.channels == 2
```

The response includes propagation time and inverse-distance gain. Distance is
floored explicitly to avoid a singularity at coincident positions. Every path
uses a two-tap fractional delay, preserving coefficient sum and centroid.
The arrival cutoff, bounded reflection order, matrix memory and convolution
work limits are explicit. This is a finite-order static reference, not a
measured acoustic simulation or neural model. No normalization, clipping,
late diffuse tail, frequency-dependent absorption or binaural head model is
implicitly added.

## Command-line render

The example plan expects a mono 24 kHz PCM WAV. For other sample rates,
edit the plan clock explicitly; no implicit resampling occurs.

```sh
sonumio room-response --plan examples/static-room-plan.json --source input.wav --output room.wav
```

The command emits a JSON receipt with source WAV SHA256, canonical plan
SHA256, output WAV SHA256, pre-encoding PCM SHA256, geometry and peak.
Use optional `--source-sha256` and `--expected-output-sha256` to pin replay.
Existing files are preserved unless `--overwrite` is explicitly provided.
Reduce the plan's output_gain if the combined signal exceeds normalized PCM.

## Explicit partitioned FFT backend

The default matrix engine remains unchanged, including its v1 export receipt
and canonical plan hash. To use the experimental uniform partitioned FIR engine:

```sh
sonumio room-response --plan examples/static-room-plan.json --source input.wav --output room.wav --engine partitioned
```

This backend splits each impulse into B-sample segments and uses a 2B-point
FFT with a frequency-delay ring. The FFT size does not grow with impulse length.
The partition count, spectral products, initialization work, output memory,
and stream duration are independently bounded. No automatic backend fallback
is performed on a limit error.

The partitioned backend returns a `sonumio-verified-room-wav-v2` receipt
with an explicit `engine: partitioned` field. Replay with the same engine,
plan, source, quantization settings, and dither seed. A pinned output SHA-256
is still verified before publication. The plan schema and its digest remain
unchanged. Different engines may differ in floating-point rounding, and
cross-engine byte-identical output is not promised.

The Python API accepts `render_room(..., engine="partitioned")` or
`export_room_plan(..., engine="partitioned")`. Generic FIR callers can use
`convolve_partitioned(audio, impulse)` or `PartitionedConvolver` from
`sonumio.audio_core.partitioned_convolution`. The latter emits full blocks
from push and the exact remaining frames and optional tail from finish.

This is a CPU offline reference backend, not a measured performance guarantee,
a real-time callback API, or a learned audio model.
