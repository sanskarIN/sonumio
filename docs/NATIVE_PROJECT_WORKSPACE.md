# Native project workspace

Open a saved .sonumio document in the local browser workspace or create
one from a verified exported score and an optional automation JSON
document. The workspace retains original UTF-8 document strings and
never reconstructs digest-bearing nested scores with JavaScript
JSON.stringify. The server validates every source document and pin.

The browser can edit a project's symbolic notes and static mixer,
preserving its existing automation lanes. Saving produces a verified
immutable revision linked to its parent. Render and export operate on
the saved project only, preventing unsaved drafts from being mistaken
for the rendered revision. Parent identity is not authentication.

Project inspection supports full bounded CLI scores. The browser
restricts note editing and rendering to 16 bars and 30 seconds.
Longer projects can be inspected, exported and revised without audio
rendering. For full-length audio use the native project CLI.

The API reads project requests as bounded streams (2 MiB project,
5 MiB mutation envelope), checks content type and declared length,
rejects compressed requests and duplicate JSON fields, and accepts no
filesystem paths. Render jobs share the existing local queue.
Project files are not written to server storage; only generated audio
uses the existing local output store. Browser recovery persists a job
identifier and project digest, not the project contents. The optional
job journal may retain only an audio reference.

Routes:
- POST /v1/music/project/inspect
- POST /v1/music/project/inspect-score
- POST /v1/music/project/inspect-automation
- POST /v1/music/project/create
- POST /v1/music/project/revise
- POST /v1/music/project/score
- POST /v1/music/project/midi
- POST /v1/jobs/music-project

The project score exports and render endpoint require
X-Sonumio-Project-SHA256. The create/revise JSON envelopes contain raw
document strings and semantic digest pins. All downloaded file bytes
carry a SHA-256 receipt. This remains a local-only development API.
