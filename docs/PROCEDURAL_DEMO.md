# Sonumio procedural listening demo

After installing the source checkout with Python 3.11+:

```bash
python -m pip install -e .
python -m sonumio.demo --output outputs/forest-storm.wav
```

Open the generated WAV in any ordinary media player. The preset is a
12-second mix of rain, thunder, wind and birds, using the existing
`procedural-sfx-v0` DSP engine and seed 42. It is 24 kHz, mono, signed
16-bit PCM. It does not use external recordings, trained weights or neural
inference, and is not a Sonumio-1 demonstration.

The command emits a JSON report containing the engine, prompt, seed, software
and NumPy versions, actual WAV SHA256 and a full-file verified signal-health
inspection. Output publication is atomic and no-clobber by default. Pass
`--overwrite` to explicitly replace an existing ordinary WAV file. An
optional `--expected-sha256` enforces the expected output bytes before
publication. A subsequent inspection error may leave an already published
WAV but never emits a verified JSON report.

The same preset can also be rendered using the original CLI:

```bash
sonumio generate "Rain, thunder, wind and birds in a forest" --seconds 12 --seed 42 --output outputs/forest-storm.wav
```

In the 24 November 2026 demonstration, using NumPy 2.3.5 and verified
procedural/encoder source files from commit
`84006ed13b901829d5060715a4df0429c2fa2ff2`, two runs produced the
same 576044-byte WAV with SHA256
`1e7bdaf71106bdc31b8b83c5029b95c7992d165a815ae65f12eff67b1f0062f9`.
The in-memory peak was -5.067 dBFS, RMS -20.116 dBFS, with zero full-scale
PCM endpoints. This is a measured example, not a cross-platform bitwise
guarantee: library versions or hardware may affect floating-point results.
No generated WAVs, trained weights, or release assets are added to Git.
