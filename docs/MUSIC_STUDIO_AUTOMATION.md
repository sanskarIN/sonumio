# Studio mixer automation

The music studio now exposes editable gain and stereo-balance lanes,
each containing up to 64 score-tick keys, with linear, smoothstep or
step interpolation. The graph uses the actual swung sample clock,
not uniform tick spacing. Keys before and after the defined interval
hold the nearest endpoint. Gain multiplies the existing track level;
balance attenuates one channel without crossfeed.

The local API provides `POST /v1/music/automation/plan` and
`POST /v1/music/automation/export`. Both accept the existing score
and mix request with an optional automation object. The planner
returns the score digest, normalized automation digest, key positions
in frames and the normalized static mix. The exporter returns the
same canonical JSON used by the offline CLI and binds the score,
automation and byte-level digests in response headers.

The browser invalidates automation receipts after changes to score,
note edits, mixer settings or lanes. Rendering pins the verified
automation digest and uses the shared music job queue. An empty
automation draft retains the original static renderer. Imported
canonical automation JSON is validated against the active score
before replacing the current controls. Changing composition clears
the old lanes. No external audio, paths, weights or network models
are accepted. This is procedural synthesis, not a trained generator.
