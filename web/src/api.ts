/** Typed same-origin client for the Sonumio local audio API. */
export type ModelEntry = {
  id: string; origin: string; model_kind: string; tasks: string[];
  license: string; enabled_by_default: boolean; description: string;
  upstream_url: string | null; original_sonumio_model: boolean;
};
export type GenerationInput = {
  prompt: string; duration_seconds: number; model_id: 'procedural-sfx-v0'; seed: number;
};
export type SceneLayer = {
  prompt: string; start_seconds: number; duration_seconds: number; gain_db: number;
  fade_in_seconds: number; fade_out_seconds: number; seed: number;
  gain_end_db?: number | null; // Optional layer end level, interpolated in decibels
  pan?: number; // -1 left, 0 centered/legacy mono, +1 right
  pan_end?: number | null; // optional end position; omitted means constant pan
};
export type SceneInput = { duration_seconds: number; layers: SceneLayer[] };
export type GeneratedAudio = {
  id: string; audio_url: string; sample_rate: number; duration_seconds: number;
  disclaimer: string; metrics: Record<string, number>; model_id?: string; engine_id?: string;
  channels?: 1 | 2;
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message); this.name = 'ApiError';
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init, credentials: 'omit',
    headers: { 'Content-Type': 'application/json', ...init.headers },
  });
  if (!response.ok) {
    let detail = 'Request failed (HTTP ' + response.status + ')';
    try {
      const body: unknown = await response.json();
      if (body && typeof body === 'object' && 'detail' in body) {
        const reason = (body as { detail: unknown }).detail;
        if (typeof reason === 'string') detail = reason;
        else if (Array.isArray(reason))
          detail = reason.map((item: { msg?: string }) => item.msg ?? 'Invalid input').join('; ');
      }
    } catch { /* Preserve HTTP status if error has no JSON. */ }
    throw new ApiError(detail, response.status);
  }
  return await response.json() as T;
}

export async function getModels(signal?: AbortSignal): Promise<ModelEntry[]> {
  return request<ModelEntry[]>('/v1/models', { signal });
}
export async function getHealth(signal?: AbortSignal): Promise<boolean> {
  const result = await request<{ status: string }>('/health', { signal });
  return result.status === 'ok';
}
export function validateAudioPath(path: string): string {
  if (!/^\/v1\/generations\/[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\/audio$/.test(path)) {
    throw new Error('API returned an invalid local audio URL');
  }
  return path;
}
export async function generate(input: GenerationInput, signal?: AbortSignal): Promise<GeneratedAudio> {
  const result = await request<GeneratedAudio>('/v1/generations', {
    method: 'POST', body: JSON.stringify(input), signal,
  });
  validateAudioPath(result.audio_url);
  return result;
}
export async function generateScene(input: SceneInput, signal?: AbortSignal): Promise<GeneratedAudio> {
  const result = await request<GeneratedAudio>('/v1/scenes', {
    method: 'POST', body: JSON.stringify(input), signal,
  });
  validateAudioPath(result.audio_url);
  return result;
}
export function validateScene(scene: SceneInput): string | null {
  if (!Number.isFinite(scene.duration_seconds) || scene.duration_seconds < 0.5 || scene.duration_seconds > 30)
    return 'Scene duration must be between 0.5 and 30 seconds.';
  if (!scene.layers.length || scene.layers.length > 8) return 'A scene needs 1–8 layers.';
  for (const [i, layer] of scene.layers.entries()) {
    const n = i + 1;
    if (!layer.prompt.trim() || layer.prompt.length > 800) return 'Layer ' + n + ' needs a prompt (max 800 characters).';
    if (!Number.isFinite(layer.start_seconds) || layer.start_seconds < 0 || layer.start_seconds > 30)
      return 'Invalid start time in layer ' + n + '.';
    if (!Number.isFinite(layer.duration_seconds) || layer.duration_seconds < 0.5 || layer.duration_seconds > 15)
      return 'Invalid duration in layer ' + n + '.';
    if (layer.start_seconds + layer.duration_seconds > scene.duration_seconds + 1e-7)
      return 'Layer ' + n + ' exceeds the scene timeline.';
    if (!Number.isFinite(layer.gain_db) || layer.gain_db < -60 || layer.gain_db > 12)
      return 'Invalid gain in layer ' + n + '.';
    if (layer.gain_end_db != null &&
        (!Number.isFinite(layer.gain_end_db) || layer.gain_end_db < -60 || layer.gain_end_db > 12))
      return 'End gain in layer ' + n + ' must be between -60 and +12 dB.';
    const pan = layer.pan ?? 0;
    if (!Number.isFinite(pan) || pan < -1 || pan > 1)
      return 'Pan in layer ' + n + ' must be between -1 (left) and +1 (right).';
    if (layer.pan_end != null &&
        (!Number.isFinite(layer.pan_end) || layer.pan_end < -1 || layer.pan_end > 1))
      return 'End pan in layer ' + n + ' must be between -1 (left) and +1 (right).';
    if (!Number.isInteger(layer.seed) || layer.seed < 0 || layer.seed >= 2 ** 32)
      return 'Invalid seed in layer ' + n + '.';
    if (!Number.isFinite(layer.fade_in_seconds) || !Number.isFinite(layer.fade_out_seconds) ||
      layer.fade_in_seconds < 0 || layer.fade_out_seconds < 0 ||
      layer.fade_in_seconds + layer.fade_out_seconds > layer.duration_seconds)
      return 'Layer ' + n + ' fades exceed its duration.';
  }
  return null;
}

/** Bounded, development-only background audio tasks. */
export type JobStatus = 'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled';
export type AudioJob = {
  id: string; kind: 'generation' | 'scene'; status: JobStatus;
  status_url: string; cancel_url: string;
  result: GeneratedAudio | null; error: string | null;
};

function jobPath(jobId: string): string {
  if (!/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/.test(jobId)) {
    throw new Error('Invalid local job identifier');
  }
  return '/v1/jobs/' + jobId;
}
function validateJob(job: AudioJob): AudioJob {
  if (job.status_url !== jobPath(job.id) || job.cancel_url !== jobPath(job.id)) {
    throw new Error('API returned an invalid local job URL');
  }
  if (!['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled'].includes(job.status)) {
    throw new Error('API returned an unknown job status');
  }
  return job;
}
export async function submitGenerationJob(input: GenerationInput, signal?: AbortSignal): Promise<AudioJob> {
  return validateJob(await request<AudioJob>('/v1/jobs/generations', {
    method: 'POST', body: JSON.stringify(input), signal,
  }));
}
export async function submitSceneJob(input: SceneInput, signal?: AbortSignal): Promise<AudioJob> {
  return validateJob(await request<AudioJob>('/v1/jobs/scenes', {
    method: 'POST', body: JSON.stringify(input), signal,
  }));
}
export async function getJob(jobId: string, signal?: AbortSignal): Promise<AudioJob> {
  return validateJob(await request<AudioJob>(jobPath(jobId), { signal }));
}
export async function cancelJob(jobId: string): Promise<AudioJob> {
  return validateJob(await request<AudioJob>(jobPath(jobId), { method: 'DELETE' }));
}
function pause(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('Cancelled', 'AbortError')); return; }
    const onAbort = () => { clearTimeout(timer); reject(new DOMException('Cancelled', 'AbortError')); };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal.addEventListener('abort', onAbort, { once: true });
  });
}
/** Poll at most five minutes, providing transparent queue status to the UI. */
export async function awaitJob(jobId: string, signal: AbortSignal,
                               onUpdate?: (status: JobStatus) => void): Promise<GeneratedAudio> {
  for (let attempt = 0; attempt < 600; attempt++) {
    if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
    const job = await getJob(jobId, signal);
    onUpdate?.(job.status);
    if (job.status === 'succeeded') {
      if (!job.result) throw new Error('Completed job has no audio result');
      validateAudioPath(job.result.audio_url);
      return job.result;
    }
    if (job.status === 'failed') throw new Error(job.error || 'Audio generation failed.');
    if (job.status === 'cancelled') throw new Error('Audio generation was cancelled.');
    await pause(500, signal);
  }
  throw new Error('Job status timed out after five minutes. The server may still be working.');
}
