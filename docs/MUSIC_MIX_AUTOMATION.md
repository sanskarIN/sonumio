# Score-clock music mixer automation

The experimental local original-music engine can render explicit gain and
stereo-balance lanes against an existing structured arrangement. No neural
weights, samples, external music or public hosting are required.

## Compose and render

~~~bash
python -m sonumio.arrangement_cli compose --bars 4 --bpm 96 --seed 17 --output score.json
sonumio music-automate score.json examples/music_mix_automation.json --output automated.wav
~~~

The same operation is available through the standalone module:

~~~bash
python -m sonumio.music_automation_cli score.json examples/music_mix_automation.json --output automated.wav
~~~

The score is the existing verified arrangement interchange, not arbitrary
MIDI or a prompt. The example automation fades the pad in, raises the lead
over the middle of the arrangement and moves its stereo balance.

## JSON format

An automation document has a format identifier, an array of lanes and
optional static mix controls:

~~~json
{
  "format": "sonumio-mix-automation-v1",
  "lanes": [
    {
      "track": "lead",
      "parameter": "gain",
      "keys": [
        {"tick": 0, "value": 0.2, "law": "smoothstep"},
        {"tick": 32, "value": 1.0},
        {"tick": 128, "value": 0.5}
      ]
    }
  ],
  "mix": {"room": 0.15, "peak_ceiling": 0.9}
}
~~~

There are eight tracks: keys, pad, bass, lead, pluck, kick, snare and hat.
Each track may have one gain lane and one balance lane. Each lane must
contain 1–64 strictly increasing integer score ticks. A key at the final
score tick is valid, but the final tick itself is an exclusive audio
endpoint. The interpolation law on the left key governs its outgoing
segment. Available laws are linear, smoothstep and step. All curves hold
their first and last values outside the keyed interval. A one-key lane
is a constant setting. Step interpolation can intentionally produce an
abrupt transition; linear or smoothstep is recommended for fades.

One beat equals eight ticks and one 4/4 bar equals 32 ticks. Tick positions
are converted through the existing arrangement swing clock. Automation
does not alter score notes, timing, harmony or instrument synthesis.

Gain is a linear multiplier from 0 to 2, multiplied by the existing static
track gain. Balance ranges from -1 (right channel attenuated to zero) to
+1 (left channel attenuated to zero); zero is unchanged. Balance does not
re-pan individual notes, introduce crossfeed, or boost a stereo channel.
Dry track automation precedes room reflections and the final edge fades.
Consequently a room response may carry energy into a later silent interval
or the other channel even if its dry signal is muted by automation.
Static mute/solo takes precedence over all lanes.

The optional mix object supports gains (eight values in the track order
above), mute, solo, room and peak_ceiling with the same restrictions as
the existing MixPlan. Unknown fields, duplicate keys, nonfinite numbers,
duplicate lanes and score-exceeding ticks are rejected.

## Reproducibility and safety limits

- Up to 16 lanes, 64 keys per lane, 128 KiB of input JSON and the existing
  arrangement frame/stem limits. Control values are evaluated in blocks
  of at most 16,384 frames. The existing full dry stems and final stereo
  mix are still held in memory; this is not a constant-memory renderer.
- The mixer preserves the input dry stems. Peak attenuation can only
  reduce amplitude; it is not a true-peak limiter or LUFS measurement.
- The receipt reports canonical score and automation SHA256 values,
  actual WAV SHA256, output bytes, geometry and per-track automated
  signal statistics. Automation identity normalizes numeric spellings
  and lane/selection order; it is not a hash of the original JSON bytes.
- Pin the score with --expected-score-sha256 and the canonical automation
  plan with --expected-automation-sha256. Pin the WAV with
  --expected-output-sha256 if exact output bytes are required.
- Existing files require --overwrite. The output may not alias the
  score or automation document. Existing local atomic WAV publication
  and its filesystem limitations apply.
- This remains a procedural instrument engine. No perceptual-quality,
  cross-platform floating-point identity or neural-generation claim
  is made.
