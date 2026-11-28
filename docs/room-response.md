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
