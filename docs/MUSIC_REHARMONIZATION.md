# Melody-conditioned diatonic harmony

This offline transformation chooses one diatonic triad per selected bar.
It minimizes a deterministic integer objective combining duration-weighted
lead nonchord tones, movement between chord degrees, deviations from the
original progression, and total accompaniment pitch displacement.

Lead and drums remain byte-equivalent as events. Accompaniment event
timing, velocity, pan and duration stay unchanged. A changed bar maps the
distinct pitches of each accompaniment track to strictly increasing
distinct pitches in the new triad. This is conservative and may exclude
some otherwise playable voicings. Bars with accompaniment crossing a
bar boundary may keep their original chord but cannot be changed.

A plan can restrict allowed degrees, the selected bar span, maximum
pitch shift, objective weights, edge preservation and dominant cadence.
The final chord must always be tonic. Infeasible constraints fail
instead of relaxing them silently. No learned model or external audio
is involved.

Example plan (substitute the actual source digest):

    {"format":"sonumio-melody-reharmonization-v1","score_sha256":"SOURCE_SHA256","first_bar":0,"bars":4,"degrees":[0,3,4,5],"maximum_shift":12}

Preview and apply:

    python -m sonumio.music_reharmonize_cli inspect score.json harmony.json
    python -m sonumio.music_reharmonize_cli apply score.json harmony.json --output new-score.json

Both commands report the original and result chord degrees, movement,
objective cost, candidate counts and source/result digests. Output
publication is atomic and cannot overwrite an input.
