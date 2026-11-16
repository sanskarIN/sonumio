# Sonumio

**Imagine it. Hear it.**

Sonumio (pronounced **so-NOO-mee-oh**, **सो-नू-मी-ओ**) is a long-term, open-source project researching and building tools for generative audio: **sound effects, music, and speech**.

> **Development status — research prototype, not released.** The `main` branch is currently a minimal public project home. The experimental application, tests, local audio studio, and research implementation are being developed in [draft pull request #1](https://github.com/sanskarIN/sonumio/pull/1) on [`feat/initial-audio-platform`](https://github.com/sanskarIN/sonumio/tree/feat/initial-audio-platform). Those features are **not yet available on `main`**.

## Project direction

- **Local-first development:** make early experimentation possible on a developer's machine.
- **Original audio research:** investigate text-conditioned neural audio generation, evaluation, and eventually larger models.
- **Creator tooling:** develop practical interfaces, APIs, and reproducible workflows.
- **Transparent provenance:** distinguish original research, procedural DSP audio, and third-party model integrations.
- **Responsible datasets:** require human-reviewed rights, permissions, and consent before model training or redistribution.

## What is being developed?

The **unmerged draft branch** currently includes:

- A procedural sound-effects baseline for English/Hindi prompts that writes **24 kHz WAV audio**. This is **DSP, not a trained AI model**.
- A Python CLI, local FastAPI service, bounded asynchronous generation jobs, and a React/TypeScript audio studio.
- Small **experimental** neural diffusion and bilingual text-conditioning research with synthetic training examples.
- Automated Python, web, and CPU research checks, plus documentation of limitations.

These are research and engineering prototypes, **not** evidence of production-quality speech/music generation, broad prompt understanding, or a released original neural model.

To inspect code, setup instructions, documentation, or experiments, see the **[development-branch README](https://github.com/sanskarIN/sonumio/blob/feat/initial-audio-platform/README.md)**. Commands in that README apply to the development branch, **not** to `main`.

## Original model generations

**Sonumio-1, Sonumio-2, Sonumio-3, ...** are names reserved for future original model generations. They are **not software version numbers**. **Sonumio-1 is not trained, validated, or published**.

No stable release, pre-release, release tag, downloadable model weights, or public service is announced or authorized at this stage.

## Contributing

This is early-stage research. Follow [draft PR #1](https://github.com/sanskarIN/sonumio/pull/1) for active development, or use [GitHub Issues](https://github.com/sanskarIN/sonumio/issues) to suggest improvements, report reproducible defects, or discuss research directions. Please do not submit private recordings, copyrighted audio without permission, or model weights without a documented distribution license.

## License

Sonumio's **source code** is licensed under [Apache License 2.0](LICENSE). Third-party model weights, audio datasets, and dependencies may have **different licenses and use restrictions**; Apache-2.0 does not automatically apply to them.

---

**Sonumio — Imagine it. Hear it.**
