import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, generate, getHealth, getModels, generateScene, validateAudioPath, validateScene } from './api';

const ID = 'f7db519f-14ba-4caf-a297-3ff567d17d99';
const audio = { id: ID, audio_url: '/v1/generations/' + ID + '/audio', sample_rate: 24000, duration_seconds: 3, metrics: {}, disclaimer: 'DSP' };
const fake = (status: number, body: unknown): Response => ({
  ok: status >= 200 && status < 300, status, json: async () => body,
} as Response);
const layer = { prompt: 'rain', start_seconds: 0, duration_seconds: 2, gain_db: 0, fade_in_seconds: 0.02, fade_out_seconds: 0.05, seed: 0 };

afterEach(() => vi.unstubAllGlobals());
describe('same-origin API client', () => {
  it('loads health and registry without using arbitrary origins', async () => {
    const mock = vi.fn().mockResolvedValueOnce(fake(200, { status: 'ok' })).mockResolvedValueOnce(fake(200, []));
    vi.stubGlobal('fetch', mock);
    expect(await getHealth()).toBe(true);
    expect(await getModels()).toEqual([]);
    expect(mock.mock.calls.map(args => args[0])).toEqual(['/health', '/v1/models']);
    expect(mock.mock.calls[0][1].credentials).toBe('omit');
  });
  it('POSTs a prompt to local generation endpoint', async () => {
    const mock = vi.fn().mockResolvedValue(fake(201, audio)); vi.stubGlobal('fetch', mock);
    await expect(generate({ prompt: 'rain', duration_seconds: 3, seed: 1, model_id: 'procedural-sfx-v0' })).resolves.toEqual(audio);
    expect(mock.mock.calls[0][0]).toBe('/v1/generations');
    expect(JSON.parse(mock.mock.calls[0][1].body).seed).toBe(1);
  });
  it('preserves server validation messages', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(422, { detail: 'Unsupported sound' })));
    await expect(generate({ prompt: 'x', duration_seconds: 1, model_id: 'procedural-sfx-v0', seed: 0 }))
      .rejects.toEqual(new ApiError('Unsupported sound', 422));
  });
  it('rejects unexpected audio URLs from API response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fake(201, { ...audio, audio_url: 'https://elsewhere.example/audio.wav' })));
    await expect(generate({ prompt: 'rain', duration_seconds: 1, model_id: 'procedural-sfx-v0', seed: 0 }))
      .rejects.toThrow('invalid local audio URL');
    expect(() => validateAudioPath('/v1/generations/../../secrets/audio')).toThrow();
    expect(() => validateAudioPath(audio.audio_url)).not.toThrow();
  });
  it('POSTs procedural scene plans', async () => {
    const mock = vi.fn().mockResolvedValue(fake(201, audio));
    vi.stubGlobal('fetch', mock);
    const positioned = { ...layer, pan: -0.5, pan_end: 0.75, gain_end_db: -12 };
    await expect(generateScene({ duration_seconds: 3, layers: [positioned] })).resolves.toEqual(audio);
    const submitted = JSON.parse(mock.mock.calls[0][1].body);
    expect(submitted.layers[0].pan).toBe(-0.5);
    expect(submitted.layers[0].pan_end).toBe(0.75);
    expect(submitted.layers[0].gain_end_db).toBe(-12);
  });
});
describe('scene client-side guardrails', () => {
  it('validates bounds and timing before sending', () => {
    expect(validateScene({ duration_seconds: 3, layers: [layer] })).toBe(null);
    expect(validateScene({ duration_seconds: 1, layers: [layer] })).toContain('exceeds');
    expect(validateScene({ duration_seconds: 4, layers: [] })).toContain('1–8');
    expect(validateScene({ duration_seconds: 31, layers: [layer] })).toContain('30 seconds');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, fade_in_seconds: 1.5, fade_out_seconds: 1 }] })).toContain('fades');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, prompt: '' }] })).toContain('prompt');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: -1 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: 1 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: 0 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: 1.01 }] })).toContain('Pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: -1.01 }] })).toContain('Pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: Number.NaN }] })).toContain('Pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: Number.POSITIVE_INFINITY }] })).toContain('Pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: 0, pan_end: 1 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: -1, pan_end: 1 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: 1, pan_end: -1 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan: 0, pan_end: null }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan_end: 1.1 }] })).toContain('End pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan_end: -1.1 }] })).toContain('End pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan_end: Number.NaN }] })).toContain('End pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, pan_end: Number.POSITIVE_INFINITY }] })).toContain('End pan');
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, gain_end_db: -60 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, gain_end_db: 12 }] })).toBe(null);
    expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, gain_end_db: null }] })).toBe(null);
    for (const invalid of [-60.01, 12.01, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(validateScene({ duration_seconds: 3, layers: [{ ...layer, gain_end_db: invalid }] })).toContain('End gain');
    }
  });
});
