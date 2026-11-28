# Complete sound-design recipes

The extended sound recipe embeds the existing render recipe plus optional
pitched and drum patch banks. The original project and v1 recipe formats
are not changed. Both patch banks must address the exact embedded score.

The compiler counts custom oscillator work using actual performed note
lengths, not the unmodified score geometry. It rejects unsupported cost
or stem allocation before synthesis. Custom patches preserve the original
event's tick and pitch for deterministic noise. Performance changes affect
the actual note sample interval, velocity and stereo pan, and all unassigned
tracks continue to use the existing original instrument engine.

    python -m sonumio.music_sound_recipe_cli create recipe.json --pitched patches.json --drums drums.json --output sound.json
    python -m sonumio.music_sound_recipe_cli inspect sound.json
    python -m sonumio.music_sound_recipe_cli render sound.json --output song.wav
    python -m sonumio.music_sound_recipe_cli bundle sound.json --output sound.zip
    python -m sonumio.music_sound_recipe_cli verify sound.zip --replay

The fixed-member stored ZIP includes a canonical sound recipe, WAV,
receipt and manifest. Verification checks exact member names, bounds,
identities, WAV geometry and canonical archive bytes. Replay repeats
the full local synthesis. Without replay, consistency checks do not
establish that the engine generated the supplied WAV. No external
recordings, plugins or learned weights are loaded.
