# Bounded local audio-generation jobs (experimental)

Sonumio's development-only generation queue is designed for **single-user,
loopback-only** operation. It runs one background worker in the FastAPI process,
with at most **4 active or queued jobs**. Only the local procedural DSP engine is
accepted; third-party MusicGen and original neural research checkpoints are not
routed through this queue.

## Endpoints

| Method | Route | Meaning |
| --- | --- | --- |
| `POST` | `/v1/jobs/generations` | Submit the same JSON as `/v1/generations`; returns HTTP `202` |
| `POST` | `/v1/jobs/scenes` | Submit the same JSON as `/v1/scenes`; returns HTTP `202` |
| `GET` | `/v1/jobs/{uuid}` | Read state, metadata and eventual generated-audio result |
| `DELETE` | `/v1/jobs/{uuid}` | Cancel a pending job or request cancellation of a running job |

```sh
curl -s http://127.0.0.1:8000/v1/jobs/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"rain and thunder","duration_seconds":3,"seed":7}'
```

Response contains `id`, `status` (`queued` initially), `status_url`,
`cancel_url`, and the local development disclaimer. Poll the status URL until
`succeeded`, `failed`, or `cancelled`. On success, `result.audio_url` is an
existing `/v1/generations/{uuid}/audio` endpoint. State may briefly be
`running` or `cancelling`.

## Guarantees and limitations

- Jobs are in memory, not durable. Restarting the application loses job state.
- One worker executes tasks sequentially. This prevents unlimited parallel
  local generation but is not GPU scheduling or process isolation.
- Additional submissions beyond the 4-job active budget return HTTP `429`.
- Pending jobs can be cancelled before starting. Once running, cancellation
  means **discarding the result when the current renderer returns**; it does
  not interrupt a model, stop a GPU kernel, or guarantee immediate release
  of memory. The returned state may temporarily be `cancelling`.
- Invalid prompts, unsupported models and impossible scene plans are rejected
  **before enqueueing**, with HTTP `422`.
- Unexpected exception messages are not exposed over HTTP; they remain in
  server logs. The single worker continues after a failed job.
- Terminal records are retained for up to 30 minutes while the process is
  active; their audio files are removed on expiry or history eviction.
  A background idle sweep and API reads can trigger expiry. On shutdown,
  best-effort cleanup occurs, but crashes/restarts can leave orphan WAV files
  under `outputs/`; users must review and remove them manually.
- Existing synchronous endpoints remain available for development callers.
  They do **not** use job retention or cleanup, so their files persist until
  manually deleted.
- This service has **no user authentication, per-user quotas, abuse prevention,
  multi-process persistence, encrypted storage, or production security audit**.
  Do not expose the API to a public network or deploy this as a shared service.

Do not merge or publish Sonumio releases, pre-releases, tags or model weights
without explicit owner approval. Original Sonumio-1 remains research-only.
