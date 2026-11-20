import { afterEach, describe, expect, it, vi } from 'vitest';
import { awaitJob, cancelJob, getJob, submitGenerationJob, submitSceneJob } from './api';

const id = '3415208b-a921-4fcc-8f33-aef223eab207';
const path = '/v1/jobs/' + id;
const audio = { id, audio_url: '/v1/generations/' + id + '/audio',
  sample_rate: 24000, duration_seconds: 1, disclaimer: 'DSP, not AI', metrics: {} };
const fake = (status: number, body: unknown): Response =>
  ({ ok: status >= 200 && status < 300, status, json: async () => body } as Response);
const job = (status: string, result: unknown = null, error: unknown = null) =>
  ({ id, kind: 'generation', status, status_url: path, cancel_url: path, result, error });
const input = { prompt: 'rain', duration_seconds: 1, seed: 0, model_id: 'procedural-sfx-v0' } as const;

afterEach(() => vi.unstubAllGlobals());

describe('development audio jobs', () => {
  it('creates jobs via fixed same-origin routes and no credentials', async () => {
    const mock = vi.fn().mockResolvedValueOnce(fake(202, job('queued')))
      .mockResolvedValueOnce(fake(202, { ...job('queued'), kind: 'scene' }));
    vi.stubGlobal('fetch', mock);
    expect((await submitGenerationJob(input)).status).toBe('queued');
    expect((await submitSceneJob({ duration_seconds: 2, layers: [] })).status).toBe('queued');
    expect(mock.mock.calls[0][0]).toBe('/v1/jobs/generations');
    expect(mock.mock.calls[1][0]).toBe('/v1/jobs/scenes');
    expect(mock.mock.calls[0][1].credentials).toBe('omit');
  });
  it('returns validated WAV result for completed jobs', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(200, job('succeeded', audio))));
    const statuses: string[] = [];
    expect(await awaitJob(id, new AbortController().signal, status => statuses.push(status))).toEqual(audio);
    expect(statuses).toEqual(['succeeded']);
  });
  it('reports failed jobs without polling indefinitely', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(200, job('failed', null, 'unsupported prompt'))));
    await expect(awaitJob(id, new AbortController().signal)).rejects.toThrow('unsupported prompt');
  });
  it('rejects malicious job URLs and result URLs', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(200, { ...job('succeeded', audio), status_url: 'https://evil.example' })));
    await expect(getJob(id)).rejects.toThrow('invalid local job URL');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(200, job('succeeded', { ...audio, audio_url: 'https://evil.example/a.wav' }))));
    await expect(awaitJob(id, new AbortController().signal)).rejects.toThrow('invalid local audio URL');
  });
  it('rejects invalid UUIDs locally without calling server', async () => {
    const mock = vi.fn();
    vi.stubGlobal('fetch', mock);
    await expect(getJob('../secret')).rejects.toThrow('Invalid local job identifier');
    expect(mock).not.toHaveBeenCalled();
  });
  it('calls DELETE on a known queued job', async () => {
    const mock = vi.fn().mockResolvedValue(fake(200, job('cancelled')));
    vi.stubGlobal('fetch', mock);
    expect((await cancelJob(id)).status).toBe('cancelled');
    expect(mock.mock.calls[0][0]).toBe(path);
    expect(mock.mock.calls[0][1].method).toBe('DELETE');
  });
  it('honors abort before polling', async () => {
    const controller = new AbortController();
    controller.abort();
    const mock = vi.fn(); vi.stubGlobal('fetch', mock);
    await expect(awaitJob(id, controller.signal)).rejects.toThrow('Cancelled');
    expect(mock).not.toHaveBeenCalled();
  });
});
