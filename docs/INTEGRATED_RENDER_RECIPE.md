# Integrated native project rendering

A render recipe embeds a canonical native project with optional sample-clock
performance and processing-rack documents. The rack must use the identical
automation plan already in the project. This avoids two contradictory
mixer configurations. No external files or plugins are referenced.

    python -m sonumio.music_render_recipe_cli create song.sonumio --performance take.json --rack rack.json --output recipe.json
    python -m sonumio.music_render_recipe_cli inspect recipe.json
    python -m sonumio.music_render_recipe_cli render recipe.json --output song.wav
    python -m sonumio.music_render_recipe_cli bundle recipe.json --output render.zip
    python -m sonumio.music_render_recipe_cli verify render.zip --replay

Four render paths reuse the existing project, performance and rack
engines. With both optional inputs, performance changes are applied
before the rack's track inserts and automation. The score and project
revision are unchanged.

The deterministic archive contains exactly manifest.json, recipe.json,
receipt.json and audio.wav. Its reader validates bounded stored members,
canonical metadata, all hashes and independent stereo PCM16 geometry.
No member is extracted to a filesystem path. Explicit extract commands
publish only the requested verified artifact atomically.

Verification without replay establishes internal content consistency,
not that a particular renderer produced the audio. Replay repeats the
synthesis and compares every byte and the receipt in the current runtime.
Neither method authenticates a creator or proves perceptual quality.
The existing project, performance and rack size limits continue to apply.
