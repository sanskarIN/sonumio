# Moving 3D acoustic scenes through static room snapshots

Sonumio can render a bounded sequence of source/receiver positions inside
one rectangular shoebox room. Each keyframe is a physically static image-
source impulse response, rendered with the existing matrix or partitioned
convolution engine. Adjacent **complete renders** are blended with
sample-clock linear crossfade weights.

This is a first-order approximation for gradually changing acoustic
scenes. It does *not* compute a retarded-time moving-source wave equation,
Doppler shift, diffraction, boundary scattering, or measured HRTFs.
Users should not interpret the result as physical ground truth.

## Position timeline

A `MotionRoomPlan` contains one room, one fixed `RoomConfig`, and 1..16
strictly ordered `SceneKeyframe` values. The first frame must be zero;
all frames must belong to the actual source duration. Every keyframe has
the same number of source and receiver points, each strictly within
the room's walls.

The weights form a nonnegative triangular basis. For each audio frame,
the sum of all active snapshot weights equals one. Before the first
transition, the first geometry is active; after the last keyframe,
the last geometry remains active through the optional late reverb tail.
Each rendered response is on the source sample clock and is never
sample-rate converted.

## Python example

```python
from sonumio.audio_core.moving_room import (
    SceneKeyframe, MotionRoomPlan, render_motion_room
)
from sonumio.audio_core.room_response import (
    Point3D, ShoeboxRoom, RoomConfig
)

room = ShoeboxRoom(6., 5., 3.)
config = RoomConfig(
    sample_rate=48000,
    reflection_order=2,
    max_seconds=1.,
    block_frames=1024,
)
listener = (Point3D(3., 2., 1.5),)
keys = (
    SceneKeyframe(0, (Point3D(1., 1.5, 1.5),), listener),
    SceneKeyframe(48000, (Point3D(2., 2., 1.5),), listener),
    SceneKeyframe(96000, (Point3D(4., 2., 1.5),), listener),
)
plan = MotionRoomPlan(room, config, keys, engine="partitioned")
outcome = render_motion_room(audio, plan, include_tail=True)
processed = outcome.audio
report = outcome.report.as_dict()
```

The source `audio` must have sample rate 48 kHz, at least 96001 frames,
and exactly one channel for this example. `MotionRoomPlan` does not open
files, read network locations, load models or change the audio clock.

## Resource and output contracts

- Each static response is bound by Sonumio's maximum reflection count,
  channel topology, impulse length and individual spectral-work budget.
- The aggregate impulse memory, cumulative processing work, output
  geometry and accumulation workspace are checked before processing.
- A result has exactly `source_frames + max_impulse_taps - 1` frames
  with `include_tail=True`, or exactly the input duration otherwise.
- The output receiver layout is explicit, not inherited from unrelated
  source-channel labels.
- Every scene report records the keyframe frame number, path count,
  direct paths and impulse length. Floating-point peak is reported
  without implicit clipping, limiting or normalization.

## Accuracy and limitations

Static geometry and a one-keyframe plan should reproduce the existing
static room renderer within float32 arithmetic tolerance. Repeating
identical keyframe positions must be approximately invariant regardless
of transition times. Keyframe transitions interpolate *rendered signal*;
they do not interpolate propagation distance for each individual
sound emission, so moving transient pitch and reverberant continuity
can differ from a physical continuous-motion model.

The output is suitable for deterministic offline research and prototypes,
with evaluation required before use in perceptual or production claims.
