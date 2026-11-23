import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  inspectPcm16Wav, loadLocalWaveform,
  MAX_PREVIEW_BYTES, MAX_PREVIEW_SECONDS, WAVEFORM_BINS,
} from './waveform';

const id = '3415208b-a921-4fcc-8f33-aef223eab207';
const audioPath = '/v1/generations/' + id + '/audio';

function pcm(
  samples: number[], channels: 1 | 2 = 1, rate = 24000,
): ArrayBuffer {
  if (samples.length % channels !== 0) throw new Error('bad test fixture');
  const data = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(data);
  const marker = (at: number, name: string) => {
    for (let index = 0; index < name.length; index++) {
      view.setUint8(at + index, name.charCodeAt(index));
    }
  };
  marker(0, 'RIFF'); view.setUint32(4, data.byteLength - 8, true);
  marker(8, 'WAVE'); marker(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * channels * 2, true);
  view.setUint16(32, channels * 2, true);
  view.setUint16(34, 16, true);
  marker(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let index = 0; index < samples.length; index++) {
    view.setInt16(44 + index * 2, samples[index], true);
  }
  return data;
}

function modified(
  original: ArrayBuffer, action: (view: DataView) => void,
): ArrayBuffer {
  const result = original.slice(0);
  action(new DataView(result));
  return result;
}

const streamResponse = (bytes: ArrayBuffer, status = 200): Response => new Response(
  new Uint8Array(bytes), {
    status,
    headers: {
      'Content-Type': 'audio/wav',
      'Content-Length': String(bytes.byteLength),
    },
  },
);

afterEach(() => vi.unstubAllGlobals());

describe('sample-accurate local WAV waveform parsing', () => {
  it('reports a mono PCM16 peak envelope from actual samples, not decorative bars', () => {
    const fixture = pcm([0, 8192, 16384, 32767, -32768, 0, -8192, 0]);
    const parsed = inspectPcm16Wav(fixture, 8);
    expect(parsed.protocol).toBe('sonumio-canonical-pcm16-peak-envelope-v1');
    expect(parsed.channels).toBe(1);
    expect(parsed.sampleRate).toBe(24000);
    expect(parsed.frames).toBe(8);
    expect(parsed.peaks).toHaveLength(8);
    expect(parsed.peaks[0]).toBe(0);
    expect(parsed.peaks[1]).toBe(0.25);
    expect(parsed.peaks[2]).toBe(0.5);
    expect(parsed.peaks[3]).toBeCloseTo(32767 / 32768, 8);
    expect(parsed.peaks[4]).toBe(1);
    expect(parsed.peaks[6]).toBe(0.25);
    expect(parsed.highestPeak).toBe(1);
    expect(parsed.durationSeconds).toBeCloseTo(8 / 24000);
  });

  it('uses both stereo channels rather than silently downmixing opposite polarity', () => {
    const interleaved = pcm([
      16384, -16384, 0, 32767, -32768, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
    ], 2);
    const parsed = inspectPcm16Wav(interleaved, 8);
    expect(parsed.channels).toBe(2);
    expect(parsed.frames).toBe(8);
    expect(parsed.peaks[0]).toBe(0.5);
    expect(parsed.peaks[1]).toBeCloseTo(32767 / 32768);
    expect(parsed.peaks[2]).toBe(1);
    expect(parsed.peaks[3]).toBe(0);
    expect(parsed.highestPeak).toBe(1);
  });

  it('reports silence as zero, without inventing visible audio energy', () => {
    const source = pcm(Array<number>(8192).fill(0));
    const result = inspectPcm16Wav(source);
    expect(result.peaks).toHaveLength(WAVEFORM_BINS);
    expect(result.peaks.every(peak => peak === 0)).toBe(true);
    expect(result.highestPeak).toBe(0);
  });

  it('uses deterministic time buckets independent of local source frequency', () => {
    const frames = 8192;
    const samples = Array<number>(frames).fill(0);
    samples[0] = 32767;
    samples[4000] = -32768;
    samples[frames - 1] = 16384;
    const result = inspectPcm16Wav(pcm(samples), 32);
    expect(result.peaks).toHaveLength(32);
    expect(result.peaks[0]).toBeCloseTo(32767 / 32768);
    expect(result.peaks[Math.floor(4000 * 32 / frames)]).toBe(1);
    expect(result.peaks[31]).toBe(0.5);
    expect(result.peaks.filter(value => value > 0)).toHaveLength(3);
  });

  it('handles one valid mono clip as 64 bounded peak bins', () => {
    const source = pcm(Array.from({ length: 200 }, (_, i) =>
      i % 2 ? -Math.min(i * 60, 32768) : Math.min(i * 60, 32767),
    ));
    const result = inspectPcm16Wav(source, 64);
    expect(result.peaks).toHaveLength(64);
    expect(result.peaks.every(x => Number.isFinite(x) && x >= 0 && x <= 1)).toBe(true);
  });

  it.each([0, 1, 7, 65, 128, -1, 8.5, NaN, Infinity])(
    'rejects unsupported bin count %s', bins => {
      expect(() => inspectPcm16Wav(pcm(Array(64).fill(0)), bins)).toThrow('bin count');
    },
  );

  it('rejects short, absent, or greater-than-8MiB WAV bytes', () => {
    expect(() => inspectPcm16Wav(new ArrayBuffer(0))).toThrow('limits');
    expect(() => inspectPcm16Wav(new ArrayBuffer(44))).toThrow('limits');
    expect(() => inspectPcm16Wav(new ArrayBuffer(MAX_PREVIEW_BYTES + 1))).toThrow('limits');
  });

  it.each([
    ['wrong RIFF', (v: DataView) => v.setUint8(0, 'X'.charCodeAt(0))],
    ['wrong WAVE', (v: DataView) => v.setUint8(8, 'X'.charCodeAt(0))],
    ['wrong fmt', (v: DataView) => v.setUint8(12, 'X'.charCodeAt(0))],
    ['wrong data', (v: DataView) => v.setUint8(36, 'X'.charCodeAt(0))],
    ['bad RIFF length', (v: DataView) => v.setUint32(4, 10, true)],
    ['unsupported noncanonical fmt', (v: DataView) => v.setUint32(16, 18, true)],
    ['float WAV', (v: DataView) => v.setUint16(20, 3, true)],
    ['three channels', (v: DataView) => v.setUint16(22, 3, true)],
    ['zero rate', (v: DataView) => v.setUint32(24, 0, true)],
    ['byte rate mismatch', (v: DataView) => v.setUint32(28, 100, true)],
    ['frame block mismatch', (v: DataView) => v.setUint16(32, 8, true)],
    ['wrong bits', (v: DataView) => v.setUint16(34, 24, true)],
    ['declared audio length mismatch', (v: DataView) => v.setUint32(40, 10, true)],
  ])('rejects malformed PCM metadata: %s', (_name, action) => {
    const input = pcm(Array(24).fill(0));
    expect(() => inspectPcm16Wav(modified(input, action))).toThrow();
  });

  it('rejects truncated payload, extra trailing bytes and bad channel alignment', () => {
    const original = pcm(Array(24).fill(0));
    expect(() => inspectPcm16Wav(original.slice(0, -2))).toThrow();
    const extra = new Uint8Array(original.byteLength + 2);
    extra.set(new Uint8Array(original), 0);
    expect(() => inspectPcm16Wav(extra.buffer)).toThrow();
    expect(() => inspectPcm16Wav(modified(original, v =>
      v.setUint32(40, 3, true)))).toThrow();
  });

  it('rejects an otherwise canonical 31-second recording', () => {
    const sampleRate = 8000;
    const seconds = MAX_PREVIEW_SECONDS + 1;
    const signal = Array(sampleRate * seconds).fill(0);
    expect(() => inspectPcm16Wav(pcm(signal, 1, sampleRate))).toThrow('30-second');
  });
});

describe('private same-origin waveform preview transport', () => {
  it('uses one validated local URL, no cookies, and no cache', async () => {
    const bytes = pcm(Array(4096).fill(0).map((_, i) => i % 2 ? 6000 : -6000));
    const fetcher = vi.fn().mockResolvedValue(streamResponse(bytes));
    vi.stubGlobal('fetch', fetcher);
    const preview = await loadLocalWaveform(audioPath);
    expect(preview.peaks).toHaveLength(WAVEFORM_BINS);
    expect(preview.highestPeak).toBeGreaterThan(0);
    expect(fetcher).toHaveBeenCalledOnce();
    expect(fetcher.mock.calls[0][0]).toBe(audioPath);
    const options = fetcher.mock.calls[0][1] as RequestInit;
    expect(options.credentials).toBe('omit');
    expect(options.cache).toBe('no-store');
    expect(options.headers).toEqual({ Accept: 'audio/wav' });
  });

  it('rejects remote URLs and arbitrary local paths before calling fetch', async () => {
    const fetcher = vi.fn();
    vi.stubGlobal('fetch', fetcher);
    for (const invalid of [
      'https://evil.example/a.wav', '//evil.example/test.wav',
      '/api/something', '/v1/generations/../../secret/audio',
    ]) {
      await expect(loadLocalWaveform(invalid)).rejects.toThrow('invalid local audio URL');
    }
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('honors an already-aborted preview request before network access', async () => {
    const fetcher = vi.fn();
    vi.stubGlobal('fetch', fetcher);
    const controller = new AbortController();
    controller.abort();
    await expect(loadLocalWaveform(audioPath, controller.signal))
      .rejects.toThrow('cancelled');
    expect(fetcher).not.toHaveBeenCalled();
  });

  it('rejects missing preview audio and leaves player behavior unchanged', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('missing', { status: 404 }),
    ));
    await expect(loadLocalWaveform(audioPath)).rejects.toThrow('HTTP 404');
  });

  it('refuses oversized Content-Length before consuming the body', async () => {
    const fetcher = vi.fn().mockResolvedValue(
      new Response('tiny', {
        status: 200,
        headers: { 'Content-Length': String(MAX_PREVIEW_BYTES + 1) },
      }),
    );
    vi.stubGlobal('fetch', fetcher);
    await expect(loadLocalWaveform(audioPath)).rejects.toThrow('oversized');
  });

  it('refuses bogus non-numeric Content-Length', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response('tiny', {
        headers: { 'Content-Length': 'untrusted-length' },
      }),
    ));
    await expect(loadLocalWaveform(audioPath)).rejects.toThrow('invalid');
  });

  it('rejects oversized streaming payload when server omits length', async () => {
    const bytes = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(MAX_PREVIEW_BYTES));
        controller.enqueue(new Uint8Array(1));
        controller.close();
      },
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(bytes)));
    await expect(loadLocalWaveform(audioPath)).rejects.toThrow('size limit');
  });

  it('refuses servers with no readable stream body', async () => {
    const fakeResponse = {
      ok: true, status: 200,
      headers: new Headers(),
      body: null,
    } as Response;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(fakeResponse));
    await expect(loadLocalWaveform(audioPath)).rejects.toThrow('cannot stream');
  });

  it('rejects malformed WAV received from an otherwise valid API URL', async () => {
    const wrong = new TextEncoder().encode('not audio bytes');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(wrong, { status: 200 }),
    ));
    await expect(loadLocalWaveform(audioPath)).rejects.toThrow();
  });

  it('honors an aborted signal during streamed WAV reading', async () => {
    const controller = new AbortController();
    let enqueued = false;
    const body = new ReadableStream<Uint8Array>({
      pull(stream) {
        if (!enqueued) {
          enqueued = true;
          stream.enqueue(new Uint8Array(44));
          controller.abort();
        } else {
          stream.close();
        }
      },
    });
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body)));
    await expect(loadLocalWaveform(audioPath, controller.signal))
      .rejects.toThrow('cancelled');
  });
});
