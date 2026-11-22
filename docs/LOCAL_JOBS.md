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

- By default, jobs remain in memory and restarting the application loses
  their state. An **opt-in local status ledger** can retain bounded metadata:
  set `SONUMIO_LOCAL_JOB_JOURNAL=1` before starting the API. The ledger file is
  `<SONUMIO_OUTPUT_DIR>/.sonumio-api-jobs.json` (or the corresponding default
  output directory). It is intended for one loopback-only API process.
- Journaled jobs are **never automatically resumed** after a crash. An
  interrupted `queued`, `running` or `cancelling` job becomes `failed`
  with the generic message `Generation interrupted by local server restart`.
  Terminal states survive within the configured retention period. Successful
  jobs retain only a validated local WAV ID and audio URL; metrics and other
  result fields are not restored. WAV availability still depends on storage
  retention, so a recovered success may refer to an expired audio file.
- The journal uses bounded, fsynced temporary snapshots and atomic rename.
  It stores no prompts, waveforms, arbitrary renderer results or checkpoint
  data. Corrupted, oversized, symlinked or incompatible journal files cause
  fail-closed startup rather than silent data discard. Back up and inspect
  such a file before changing it. If journal writes fail at submission, new
  work is rejected instead of receiving an untracked accepted job. Journal
  writes during ongoing work are best-effort and can leave last-known states
  after a storage error; this is **not** a transactional job queue.
- One worker executes tasks sequentially. This prevents unlimited parallel
  local generation but is not GPU scheduling or process isolation.
- Additional submissions beyond the 4-job active budget return HTTP `429`.
  A worker that cannot start, an exhausted unique job-ID reservation, or a
  queue already shutting down returns HTTP `503` instead of accepting a
  job that cannot run. Job IDs are reserved without overwriting existing
  queued or terminal history, even if a collision is injected.
- Nonblocking shutdown immediately marks all still-queued jobs
  `cancelled` (including a finish timestamp) and clears pending work. An
  in-progress render is marked `cancelling` until it returns, when its output
  is discarded and best-effort cleaned. Shutdown does not preempt the renderer.
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
  A background idle sweep and API reads can trigger expiry. With the default
  RAM queue, shutdown attempts to clean up its completed audio files. With
  the opt-in journal, graceful shutdown preserves terminal metadata and its
  WAV references for later cleanup at the original TTL. A crash can still
  leave orphan WAVs if audio was committed immediately before a metadata
  update; the output store's own separate retention cleanup remains necessary.
- Existing synchronous endpoints remain available for development callers.
  They do **not** use job retention or cleanup, so their files persist until
  manually deleted.
- This service has **no user authentication, per-user quotas, abuse prevention,
  multi-process coordination, encrypted storage, or production security audit**.
  Do not expose the API to a public network or deploy this as a shared service.

Do not merge or publish Sonumio releases, pre-releases, tags or model weights
without explicit owner approval. Original Sonumio-1 remains research-only.
