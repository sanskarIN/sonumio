# Offline music processing rack

The rack provides score-pinned instrument processing without third-party
plugins or audio files. All rendering remains original procedural synthesis.
Its fixed topology is:

1. Dry instrument stems.
2. Ordered per-track insert chains.
3. Static faders and score-clock gain/balance automation.
4. Parallel post-fader auxiliary buses, then summation into the main mix.
5. Existing finite room response, ordered master inserts, endpoint fades,
   and sample-peak ceiling.

A muted track contributes neither direct audio nor auxiliary sends. Returns
cannot feed other returns or themselves. Each effect preserves duration.
Delay tails outside the score are discarded. There is no latency compensation,
live monitoring, LUFS claim, or true-peak certification.

A minimal rack document:

    {
      "format": "sonumio-music-processing-rack-v1",
      "automation": {
        "format": "sonumio-mix-automation-v1",
        "lanes": []
      },
      "tracks": [
        {"track":"lead","inserts":[
          {"kind":"eq","parameters":{"kind":"lowpass","cutoff_hz":3800}},
          {"kind":"compressor","wet":0.5}
        ]}
      ],
      "aux": [
        {"name":"aux1","sends":[{"track":"lead","level":0.2}],
         "inserts":[{"kind":"delay","parameters":{"delay_seconds":0.12,"taps":3}}],
         "return_gain":0.5}
      ],
      "master":[{"kind":"width","parameters":{"width":1.1}}]
    }

The eight supported insert kinds are gain, width, eq, compressor, gate,
delay, tremolo and saturation. Every insert supports a wet fraction (0–1)
and boolean bypass. Wet zero and bypass preserve the incoming samples exactly.
The delay's wet branch is silent if none of its taps fits within the score.

Inspect before rendering:

    python -m sonumio.music_rack_cli inspect score.json rack.json

Render with an optional expected rack digest:

    python -m sonumio.music_rack_cli render score.json rack.json --output mix.wav

The receipt includes score, rack and automation hashes, per-stage input/output
hashes, sample geometry, signal statistics, and the final WAV hash. Stage
hashes use canonical little-endian float64 stereo samples and a domain prefix.
These hashes verify reproducibility; they do not authenticate a creator.

The offline rack supports at most 60 seconds, six inserts per chain, two
auxiliary buses, and a conservative processing-cost cap. Intermediate
absolute sample amplitudes above 32 fail rather than clipping silently.
The command never overwrites an input document and uses the existing atomic
WAV publication path. The rack format is separate from native project v1.
