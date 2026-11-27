# Geometric moving-source stereo audio

This local DSP reference places mono sounds on a two-dimensional,
metre-based trajectory. It computes a delay from every source emission
to two separated receivers and maps it onto the output sample clock.
Receiver heading defines the lateral direction, with an adjustable
simple equal-power directional mix and distance attenuation.

Optional rectangular rooms use four first-order image-source paths.
Wall gains are explicit, and zero-gain walls are omitted. The model
does not use measured HRTFs, calibrated room acoustics, occlusion,
diffraction, or frequency-dependent ear filtering. Linear interpolation
of the arrival map may introduce artifacts; a Doppler anti-alias
guarantee is not provided.

Motion starts at source-local frame zero, uses piecewise-linear keys,
and holds the final position. Each segment is limited to 40 m/s so
that receiver arrivals are strictly increasing. An automatically sized
scene includes the last direct and reflected arrival; an explicit
scene duration can truncate paths, which the report records.

Example plan with one named source:

```json
{
  "sample_rate": 24000,
  "placements": [
    {
      "source_id": "rain",
      "trajectory": {
        "keys": [
          {"frame": 0, "position": {"x": -3, "y": 4}},
          {"frame": 23999, "position": {"x": 3, "y": 4}}
        ]
      }
    }
  ],
  "tail_frames": 1000
}
```

```bash
python -m sonumio.audio_core.spatial_cli spatial-scene --plan scene.json --source rain=rain.wav --output moving-rain.wav
```

The WAV must be mono at the declared sample rate and contain the
declared motion-key frames. Sources are bound explicitly by identifier;
a JSON plan cannot select a filesystem path. One source may be placed
multiple times. Source SHA256 pins can be provided separately, and all
source snapshots must be verified before any render or output publication.
The output is one canonical stereo PCM file, with optional explicit bit
depth and seeded dither.

Reports include the plan digest, source file/PCM digests, direct and
reflection arrival ranges, distances, rendered frame counts, truncation,
output peak and WAV digest. Numerical tests cover static delays, moving
arrival clocks, image-source geometry, mirror symmetry, bounds and
deterministic verified file export. The feature is not a neural model.
