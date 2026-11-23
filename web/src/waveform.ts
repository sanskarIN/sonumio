/** Actual local PCM peak envelope for the Sonumio studio preview.

No Web Audio autoplay, no external model, no remote provider, no decoded
recording storage and no fake procedural bars after rendering. Fetches only
validated same-origin Sonumio generation IDs, reads no more than 8 MiB,
and rejects malformed/noncanonical RIFF/WAVE before rendering any peaks.

The envelope measures source PCM peak amplitude in fixed temporal buckets.
It is NOT a loudness (LUFS), perceptual-quality, or speech/music score.
*/

import { ApiError, validateAudioPath } from './api';

export const MAX_PREVIEW_BYTES = 8 * 1024 * 1024;
export const WAVEFORM_BINS = 32;
export const MAX_PREVIEW_SECONDS = 30;

export type WaveformPreview = {
  peaks: number[];
  channels: 1 | 2;
  sampleRate: number;
  frames: number;
  durationSeconds: number;
  highestPeak: number;
  protocol: 'sonumio-canonical-pcm16-peak-envelope-v1';
};

function requireInt(value: number, min: number, max: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    throw new Error('Invalid WAV ' + name);
  return value;
}

function fourCC(view: DataView, offset: number, expected: string): void {
  for (let index = 0; index < 4; index++) {
    if (view.getUint8(offset + index) !== expected.charCodeAt(index))
      throw new Error('Unsupported or malformed WAV ' + expected + ' chunk');
  }
}

/** Decode the exact 44-byte interleaved integer PCM format Sonumio emits. */
export function inspectPcm16Wav(bytes: ArrayBuffer, bins = WAVEFORM_BINS): WaveformPreview {
  if (!(bytes instanceof ArrayBuffer) || bytes.byteLength < 46 ||
      bytes.byteLength > MAX_PREVIEW_BYTES)
    throw new Error('WAV preview exceeds byte limits or is incomplete');
  requireInt(bins, 8, 64, 'waveform bin count');
  const view = new DataView(bytes);
  fourCC(view, 0, 'RIFF');
  fourCC(view, 8, 'WAVE');
  fourCC(view, 12, 'fmt ');
  fourCC(view, 36, 'data');

  if (view.getUint32(4, true) !== bytes.byteLength - 8 ||
      view.getUint32(16, true) !== 16 ||
      view.getUint16(20, true) !== 1)
    throw new Error('WAV must have an exact canonical integer PCM header');

  const channels = requireInt(view.getUint16(22, true), 1, 2, 'channel count') as 1 | 2;
  const sampleRate = requireInt(view.getUint32(24, true), 8000, 192000, 'sample rate');
  const bitsPerSample = view.getUint16(34, true);
  const bytesPerFrame = channels * 2;
  if (bitsPerSample !== 16 ||
      view.getUint16(32, true) !== bytesPerFrame ||
      view.getUint32(28, true) !== sampleRate * bytesPerFrame)
    throw new Error('WAV preview supports canonical interleaved PCM16 only');

  const length = view.getUint32(40, true);
  if (length <= 0 || length % bytesPerFrame !== 0 ||
      length + 44 !== bytes.byteLength)
    throw new Error('WAV preview payload and RIFF file sizes disagree');
  const frames = length / bytesPerFrame;
  const durationSeconds = frames / sampleRate;
  if (!Number.isFinite(durationSeconds) || durationSeconds > MAX_PREVIEW_SECONDS)
    throw new Error('WAV preview duration exceeds the local 30-second limit');

  const peaks = Array<number>(bins).fill(0);
  let highestPeak = 0;
  for (let bin = 0; bin < bins; bin++) {
    const first = Math.floor(bin * frames / bins);
    const stop = Math.floor((bin + 1) * frames / bins);
    let maximum = 0;
    for (let frame = first; frame < stop; frame++) {
      const offset = 44 + frame * bytesPerFrame;
      for (let channel = 0; channel < channels; channel++) {
        const value = Math.abs(view.getInt16(offset + channel * 2, true)) / 32768;
        if (value > maximum) maximum = value;
      }
    }
    peaks[bin] = maximum;
    if (maximum > highestPeak) highestPeak = maximum;
  }
  return {
    peaks, channels, sampleRate, frames, durationSeconds, highestPeak,
    protocol: 'sonumio-canonical-pcm16-peak-envelope-v1',
  };
}

/** Bounded stream read: never materialize an unbounded server response. */
async function readBoundedWav(response: Response, signal?: AbortSignal): Promise<ArrayBuffer> {
  const stated = response.headers.get('content-length');
  if (stated !== null && (!/^\d+$/.test(stated) ||
      !Number.isSafeInteger(Number(stated)) ||
      Number(stated) > MAX_PREVIEW_BYTES))
    throw new Error('Audio preview declares an invalid or oversized WAV');

  if (!response.body) throw new Error('Browser cannot stream a WAV preview');
  const reader = response.body.getReader();
  const parts: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      if (signal?.aborted) throw new Error('Audio waveform preview cancelled');
      const { done, value } = await reader.read();
      if (done) break;
      if (!(value instanceof Uint8Array) ||
          length + value.byteLength > MAX_PREVIEW_BYTES)
        throw new Error('Audio waveform preview exceeds the local size limit');
      parts.push(value);
      length += value.byteLength;
    }
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally {
    reader.releaseLock();
  }
  if (signal?.aborted) throw new Error('Audio waveform preview cancelled');
  const combined = new Uint8Array(length);
  let offset = 0;
  for (const part of parts) {
    combined.set(part, offset);
    offset += part.byteLength;
  }
  return combined.buffer;
}

/** Fetch same-origin generated WAV with no credentials or browser cache. */
export async function loadLocalWaveform(
  path: string, signal?: AbortSignal,
): Promise<WaveformPreview> {
  const safePath = validateAudioPath(path);
  if (signal?.aborted) throw new Error('Audio waveform preview cancelled');
  const response = await fetch(safePath, {
    signal,
    credentials: 'omit',
    cache: 'no-store',
    redirect: 'error',
    headers: { Accept: 'audio/wav' },
  });
  if (!response.ok)
    throw new ApiError('Local audio waveform unavailable (HTTP ' + response.status + ')', response.status);
  return inspectPcm16Wav(await readBoundedWav(response, signal));
}
