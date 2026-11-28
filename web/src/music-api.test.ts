import { afterEach, describe, expect, it, vi } from 'vitest';
import { createHash, webcrypto } from 'node:crypto';
import { exportMusic, submitMusicJob } from './api';
import { defaultComposition, defaultMusicMix } from './music-contract';

const digest = 'a'.repeat(64);
const id = 'f7db519f-14ba-4caf-a297-3ff567d17d99';
const input = { composition: defaultComposition(), expected_score_sha256: digest };
const bytes = new Uint8Array([77, 84, 104, 100, 0, 0, 0, 6]);
const checksum = createHash('sha256').update(bytes).digest('hex');

function response(overrides: Record<string, string> = {}, body = bytes) {
  return new Response(body, { status: 200, headers: {
    'content-type': 'audio/midi',
    'content-length': String(bytes.length),
    'X-Sonumio-Score-SHA256': digest,
    'X-Sonumio-Content-SHA256': checksum,
    ...overrides,
  } });
}
afterEach(() => vi.unstubAllGlobals());

describe('music export downloads', () => {
  it('downloads bounded, digest-verified bytes from the fixed local endpoint', async () => {
    vi.stubGlobal('crypto', webcrypto);
    const fetcher = vi.fn().mockResolvedValue(response());
    vi.stubGlobal('fetch', fetcher);
    const blob = await exportMusic(input, 'midi');
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(bytes);
    expect(blob.type).toBe('audio/midi');
    expect(fetcher.mock.calls[0][0]).toBe('/v1/music/midi');
    expect(fetcher.mock.calls[0][1]).toMatchObject({
      method: 'POST', credentials: 'omit', redirect: 'error', cache: 'no-store',
    });
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual(input);
  });
  it.each([
    { 'X-Sonumio-Score-SHA256': 'b'.repeat(64) },
    { 'X-Sonumio-Content-SHA256': 'b'.repeat(64) },
    { 'X-Sonumio-Content-SHA256': 'not-a-hash' },
    { 'content-length': '1048577' }, { 'content-length': '0' },
    { 'content-length': '7' }, { 'content-length': '9' },
    { 'content-type': 'text/html' },
  ])('rejects inconsistent download metadata %j', async headers => {
    vi.stubGlobal('crypto', webcrypto);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response(headers)));
    await expect(exportMusic(input, 'midi')).rejects.toThrow();
  });
  it('rejects missing score pin without contacting the server', async () => {
    const fetcher = vi.fn();
    vi.stubGlobal('fetch', fetcher);
    await expect(exportMusic({ composition: defaultComposition() }, 'json')).rejects.toThrow('Plan');
    expect(fetcher).not.toHaveBeenCalled();
  });
  it('detects changed bytes despite matching receipt headers', async () => {
    vi.stubGlobal('crypto', webcrypto);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(response({}, new Uint8Array(bytes.length))));
    await expect(exportMusic(input, 'midi')).rejects.toThrow('checksum mismatch');
  });
});

describe('music queue submission', () => {
  it('submits a pinned score and mix with a client-generated identifier', async () => {
    const item = { id, kind: 'music', status: 'queued', result: null, error: null,
      status_url: '/v1/jobs/' + id, cancel_url: '/v1/jobs/' + id };
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(item), { status: 202 }));
    vi.stubGlobal('fetch', fetcher);
    const renderInput = { ...input, mix: defaultMusicMix() };
    expect(await submitMusicJob(renderInput, { id })).toEqual(item);
    expect(fetcher.mock.calls[0][0]).toBe('/v1/jobs/music');
    expect(fetcher.mock.calls[0][1].headers['X-Sonumio-Job-Id']).toBe(id);
    expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual(renderInput);
  });
  it('rejects invalid mix before fetching', () => {
    const fetcher = vi.fn();
    vi.stubGlobal('fetch', fetcher);
    expect(() => submitMusicJob({ ...input, mix: { ...defaultMusicMix(), gains: [] } })).toThrow('eight gains');
    expect(fetcher).not.toHaveBeenCalled();
  });
});
