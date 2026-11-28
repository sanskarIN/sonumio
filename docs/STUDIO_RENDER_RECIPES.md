# Integrated render recipes in the local studio

Open an existing render recipe or create one from a native project,
optional performance JSON and optional rack JSON. The local API validates
each document and the project's declared identity. The rack must contain
the same automation as the project. Raw source document strings are
preserved until server-side canonicalization.

The recipe inspector reports all component digests, engine, revision,
sample geometry and whether the preview duration is supported. Rendering
uses the existing bounded music queue, with an idempotency identity
containing the recipe and component digests. Browser WAV downloads check
the result identity, exact byte count, PCM16 header and SHA256.

The browser stores only the last job identifier and recipe digest in
session storage. To recover a render, reopen its exact recipe; recovery
fails if the queue no longer retains the complete receipt. Full offline
render receipts and canonical archive replay remain available via
music_render_recipe_cli. No archive generation is added to the HTTP
request path. No external audio or model weights are required.

Endpoints:
- POST /v1/music/render-recipe/create
- POST /v1/music/render-recipe/inspect
- POST /v1/jobs/music-recipe

The render endpoint requires X-Sonumio-Recipe-SHA256. Browser rendering
is limited to 30 seconds and 16 bars.
