# Sonumio web interface (experimental)

Local-first React/TypeScript frontend for the existing **procedural DSP** backend.
It is **not an AI-trained sound generator** and does not load Sonumio-1 weights.

## Run locally

Start the Python API in one terminal:

```sh
python -m pip install -e '.[test]'
sonumio serve --port 8000
```

In another terminal (Node 22+):

```sh
cd web
npm install
npm run dev
```

Visit **http://127.0.0.1:5173**. Vite proxies only `/health` and `/v1` to
`127.0.0.1:8000`; cross-origin backend access and public hosting are not
required. The app can generate a single WAV or mix up to 8 procedural scene
layers. Use `npm run build`, `npm run typecheck`, and `npm test` to validate.

The interface makes all supported model origins explicit. Optional third-party
MusicGen is listed for transparency but not selectable while its actual runtime
is unverified. The neural diffusion experiment is research-only and NOT
available as an API engine. Voice imitation/cloning is not supported.

This local prototype still lacks auth, quotas, retention controls, async
GPU jobs, accessibility audits and production deployment hardening.
The generated audio API keeps WAV files in `outputs/`; remove them manually
when finished. Do not expose the API or Vite server on a public interface.

No GitHub pre-releases, releases, tags or model weight publication are approved.

## Background jobs in the web studio

The interface now submits the sound/scene form through the bounded local
`/v1/jobs` queue. It polls job status, reports queued/running states and
requests server-side cancellation via `DELETE /v1/jobs/{id}`. Running work
is non-preemptive, so cancellation is best effort. Job audio and records
expire after 30 minutes while the local API process is alive. The previous
synchronous `/v1/generations` and `/v1/scenes` endpoints remain supported.

Frontend job/client validation has automated Vitest coverage; see
[queue safeguards](../docs/LOCAL_JOBS.md). No public deployment or
release is authorized.
