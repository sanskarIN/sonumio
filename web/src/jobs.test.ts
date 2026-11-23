import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  awaitJob, cancelJob, getJob, submitGenerationJob, submitSceneJob,
  JOB_ACTION_TIMEOUT_MS, JOB_POLL_TIMEOUT_MS,
} from './api';

const id = '3415208b-a921-4fcc-8f33-aef223eab207';
const path = '/v1/jobs/' + id;
const audio = { id, audio_url: '/v1/generations/' + id + '/audio',
  sample_rate: 24000, duration_seconds: 1, disclaimer: 'DSP, not AI', metrics: {} };
const fake = (status: number, body: unknown): Response =>
  ({ ok: status >= 200 && status < 300, status, json: async () => body } as Response);
const job = (status: string, result: unknown = null, error: unknown = null) =>
  ({ id, kind: 'generation', status, status_url: path, cancel_url: path, result, error });
const input = { prompt: 'rain', duration_seconds: 1, seed: 0, model_id: 'procedural-sfx-v0' } as const;

afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

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
  it('sends the client ID with both job forms and rejects an acknowledgement for another ID', async () => {
    const mock = vi.fn().mockResolvedValue(fake(202, job('queued')));
    vi.stubGlobal('fetch', mock);
    expect((await submitGenerationJob(input, { id })).id).toBe(id);
    await submitSceneJob({ duration_seconds: 2, layers: [] }, { id });
    expect(mock.mock.calls.map(([, options]) => options.headers['X-Sonumio-Job-Id'])).toEqual([id, id]);
    expect(mock.mock.calls.every(([, options]) => options.redirect === 'error')).toBe(true);
    await expect(submitGenerationJob(input, { id: '0415208b-a921-4fcc-8f33-aef223eab207' }))
      .rejects.toThrow('could not be confirmed');
  });
  it('recovers an accepted render with GET after a lost acknowledgement, without reposting', async () => {
    const mock = vi.fn().mockRejectedValueOnce(new TypeError('Connection dropped'))
      .mockResolvedValueOnce(fake(200, job('succeeded', audio)));
    vi.stubGlobal('fetch', mock);
    await expect(submitGenerationJob(input, { id })).rejects.toThrow('Check last job');
    await expect(awaitJob(id, new AbortController().signal)).resolves.toEqual(audio);
    expect(mock.mock.calls).toHaveLength(2);
    expect(mock.mock.calls[0][0]).toBe('/v1/jobs/generations');
    expect(mock.mock.calls[0][1].method).toBe('POST');
    expect(mock.mock.calls[1][0]).toBe(path);
    expect(mock.mock.calls[1][1].method).toBeUndefined();
  });
  it('bounds a stalled submission and retains the same recovery ID without another attempt', async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn((_url: string, options: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        options.signal?.addEventListener('abort', () => reject(options.signal?.reason), { once: true });
      }));
    vi.stubGlobal('fetch', fetcher);
    const assertion = expect(submitGenerationJob(input, { id })).rejects.toThrow('Check last job');
    await vi.advanceTimersByTimeAsync(JOB_ACTION_TIMEOUT_MS);
    await assertion;
    expect(fetcher).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  });
  it('preserves an explicit queue rejection and never submits a pre-aborted job', async () => {
    const mock = vi.fn().mockResolvedValue(fake(429, { detail: 'Queue full' }));
    vi.stubGlobal('fetch', mock);
    await expect(submitGenerationJob(input, { id })).rejects.toMatchObject({
      status: 429, message: 'Queue full',
    });
    const controller = new AbortController();
    controller.abort();
    await expect(submitGenerationJob(input, { id, signal: controller.signal }))
      .rejects.toMatchObject({ name: 'AbortError' });
    expect(mock).toHaveBeenCalledOnce();
  });
  it('returns validated WAV result for completed jobs', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(200, job('succeeded', audio))));
    const statuses: string[] = [];
    expect(await awaitJob(id, new AbortController().signal, status => statuses.push(status))).toEqual(audio);
    expect(statuses).toEqual(['succeeded']);
  });
  it('returns a journal-recovered audio reference without inventing WAV metadata', async () => {
    const reference = { id, audio_url: audio.audio_url };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(200, job('succeeded', reference))));
    await expect(awaitJob(id, new AbortController().signal)).resolves.toEqual(reference);
  });
  it('rejects valid local URLs that belong to a different job or audio ID', async () => {
    const other = '0415208b-a921-4fcc-8f33-aef223eab207';
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(fake(200, {
      ...job('running'), id: other, status_url: '/v1/jobs/' + other, cancel_url: '/v1/jobs/' + other,
    })).mockResolvedValueOnce(fake(200, job('succeeded', { ...audio, id: other }))));
    await expect(getJob(id)).rejects.toThrow('different local job');
    await expect(awaitJob(id, new AbortController().signal)).rejects.toThrow('different local audio URL');
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
  it('requests fresh job status rather than a possibly cached queue snapshot', async () => {
    const mock = vi.fn().mockResolvedValue(fake(200, job('running')));
    vi.stubGlobal('fetch', mock);
    expect((await getJob(id)).status).toBe('running');
    expect(mock.mock.calls[0][1].cache).toBe('no-store');
    expect(mock.mock.calls[0][1].credentials).toBe('omit');
  });

  it('enforces the overall five-minute deadline even when HTTP polling hangs', async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn((_url: string, options: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        options.signal?.addEventListener('abort', () => {
          reject(new DOMException('Cancelled', 'AbortError'));
        }, { once: true });
      }));
    vi.stubGlobal('fetch', fetcher);
    const pending = awaitJob(id, new AbortController().signal);
    const assertion = expect(pending).rejects.toThrow('timed out after five minutes');
    await vi.advanceTimersByTimeAsync(JOB_POLL_TIMEOUT_MS);
    await assertion;
    expect(fetcher).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('immediately interrupts an in-flight status request when the user cancels', async () => {
    vi.useFakeTimers();
    const controller = new AbortController();
    const fetcher = vi.fn((_url: string, options: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        options.signal?.addEventListener('abort', () => {
          reject(new DOMException('Cancelled', 'AbortError'));
        }, { once: true });
      }));
    vi.stubGlobal('fetch', fetcher);
    const pending = awaitJob(id, controller.signal);
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetcher).toHaveBeenCalledOnce();
    expect(vi.getTimerCount()).toBe(0);
  });

  it('cleans up the timeout after a successful job without waiting five minutes', async () => {
    vi.useFakeTimers();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(200, job('succeeded', audio))));
    await expect(awaitJob(id, new AbortController().signal)).resolves.toEqual(audio);
    expect(vi.getTimerCount()).toBe(0);
  });
});
