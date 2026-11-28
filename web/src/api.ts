import { validateAutomation, parseAutomationReceipt, MAX_AUTOMATION_BYTES, type AutomationReceipt } from './music-automation-contract';
import { validateScoreInput, validateMusicMix, parseMusicPlan, isDigest, type MusicScoreInput, type MusicRenderInput, type MusicPlan } from './music-contract';
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


/** Human-editable, local-only scene plans. Never reads external URLs. */
export const MAX_SCENE_JSON_BYTES = 16 * 1024;
const SCENE_FORMAT = 'sonumio-scene-v1';

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function sceneNumber(row: Record<string, unknown>, key: string, fallback: number): number {
  const value = row[key] === undefined ? fallback : row[key];
  if (typeof value !== 'number') throw new Error('Scene field "' + key + '" must be a number.');
  return value;
}

function sceneOptionalNumber(row: Record<string, unknown>, key: string): number | undefined {
  const value = row[key];
  if (value == null) return undefined;
  if (typeof value !== 'number') throw new Error('Scene field "' + key + '" must be a number.');
  return value;
}

/** Parse both Sonumio JSON exports and simple CLI scene JSON. Never coerce types. */
export function parseScenePlan(text: string): SceneInput {
  if (text.length > MAX_SCENE_JSON_BYTES)
    throw new Error('Scene JSON exceeds the 16 KiB local import limit.');
  let raw: unknown;
  try { raw = JSON.parse(text); }
  catch { throw new Error('Scene file is not valid JSON.'); }
  if (!isRecord(raw) || !Array.isArray(raw.layers))
    throw new Error('Scene JSON must contain an object with a layers array.');
  if (raw.format !== undefined && raw.format !== SCENE_FORMAT)
    throw new Error('Unsupported Sonumio scene file format.');
  if (raw.layers.length < 1 || raw.layers.length > 8)
    throw new Error('A scene needs 1–8 layers.');
  const layers: SceneLayer[] = raw.layers.map((item: unknown, index: number) => {
    if (!isRecord(item) || typeof item.prompt !== 'string')
      throw new Error('Layer ' + (index + 1) + ' needs a string prompt.');
    return {
      prompt: item.prompt,
      start_seconds: sceneNumber(item, 'start_seconds', 0),
      duration_seconds: sceneNumber(item, 'duration_seconds', 3),
      gain_db: sceneNumber(item, 'gain_db', 0),
      gain_end_db: sceneOptionalNumber(item, 'gain_end_db'),
      fade_in_seconds: sceneNumber(item, 'fade_in_seconds', 0.02),
      fade_out_seconds: sceneNumber(item, 'fade_out_seconds', 0.05),
      seed: sceneNumber(item, 'seed', 0),
      pan: sceneNumber(item, 'pan', 0),
      pan_end: sceneOptionalNumber(item, 'pan_end'),
    };
  });
  const scene = { duration_seconds: sceneNumber(raw, 'duration_seconds', 5), layers };
  const error = validateScene(scene);
  if (error) throw new Error(error);
  return scene;
}

/** Export only supported fields; ignore any unrelated object properties. */
export function serializeScenePlan(scene: SceneInput): string {
  const error = validateScene(scene);
  if (error) throw new Error(error);
  return JSON.stringify({
    format: SCENE_FORMAT,
    duration_seconds: scene.duration_seconds,
    layers: scene.layers.map(layer => ({
      prompt: layer.prompt, start_seconds: layer.start_seconds,
      duration_seconds: layer.duration_seconds, gain_db: layer.gain_db,
      ...(layer.gain_end_db == null ? {} : { gain_end_db: layer.gain_end_db }),
      fade_in_seconds: layer.fade_in_seconds,
      fade_out_seconds: layer.fade_out_seconds, seed: layer.seed,
      pan: layer.pan ?? 0,
      ...(layer.pan_end == null ? {} : { pan_end: layer.pan_end }),
    })),
  }, null, 2) + '\n';
}

/** Bounded, development-only background audio tasks. */
export type JobStatus = 'queued' | 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled';
// An optional journal restores the audio reference without renderer metadata.
export type JobAudio = Pick<GeneratedAudio, 'id' | 'audio_url'> & Partial<GeneratedAudio>;
export type AudioJob = {
  id: string; kind: 'generation' | 'scene' | 'music'; status: JobStatus;
  status_url: string; cancel_url: string;
  result: JobAudio | null; error: string | null;
};

export function isClientJobId(value: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(value);
}

function jobPath(jobId: string): string {
  if (!/^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/.test(jobId)) {
    throw new Error('Invalid local job identifier');
  }
  return '/v1/jobs/' + jobId;
}
export function validateJob(job: AudioJob, expectedId?: string): AudioJob {
  const path = jobPath(job.id);
  if (job.status_url !== path || job.cancel_url !== path) {
    throw new Error('API returned an invalid local job URL');
  }
  if (expectedId !== undefined && job.id !== expectedId)
    throw new Error('API returned a different local job');
  if (!['queued', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled'].includes(job.status)) {
    throw new Error('API returned an unknown job status');
  }
  return job;
}

export const JOB_ACTION_TIMEOUT_MS = 10_000;

export async function withJobDeadline<T>(
  signal: AbortSignal | undefined, action: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  signal?.throwIfAborted();
  const pending = new AbortController();
  const scoped = signal ? AbortSignal.any([signal, pending.signal]) : pending.signal;
  const timer = setTimeout(() => pending.abort(new Error('Local job request timed out.')), JOB_ACTION_TIMEOUT_MS);
  try {
    const result = await action(scoped);
    scoped.throwIfAborted();
    return result;
  } finally {
    clearTimeout(timer);
  }
}

type JobSubmissionOptions = { id?: string; signal?: AbortSignal };

export function isRejectedJobSubmission(error: unknown): error is ApiError {
  return error instanceof ApiError && error.status >= 400 && error.status < 500 && error.status !== 408;
}

async function submitJob(
  kind: 'generations' | 'scenes' | 'music', input: GenerationInput | SceneInput | MusicRenderInput,
  { id, signal }: JobSubmissionOptions,
): Promise<AudioJob> {
  if (id !== undefined && !isClientJobId(id)) throw new Error('Invalid client job identifier');
  try {
    const task = await withJobDeadline(signal, pending => request<AudioJob>('/v1/jobs/' + kind, {
      method: 'POST', body: JSON.stringify(input), signal: pending,
      redirect: 'error',
      headers: id === undefined ? undefined : { 'X-Sonumio-Job-Id': id },
    }));
    return validateJob(task, id);
  } catch (error) {
    if (id === undefined || signal?.aborted || isRejectedJobSubmission(error))
      throw error;
    throw new Error('Submission could not be confirmed. Use Check last job to look up the same render.', { cause: error });
  }
}

export function submitGenerationJob(input: GenerationInput, options: JobSubmissionOptions = {}): Promise<AudioJob> {
  return submitJob('generations', input, options);
}
export function submitSceneJob(input: SceneInput, options: JobSubmissionOptions = {}): Promise<AudioJob> {
  return submitJob('scenes', input, options);
}

export async function getJob(jobId: string, signal?: AbortSignal): Promise<AudioJob> {
  return validateJob(await request<AudioJob>(jobPath(jobId), { signal, cache: 'no-store', redirect: 'error' }), jobId);
}
export async function cancelJob(jobId: string, signal?: AbortSignal): Promise<AudioJob> {
  const path = jobPath(jobId);
  const task = await withJobDeadline(signal, pending => request<AudioJob>(
    path, { method: 'DELETE', signal: pending, redirect: 'error' },
  ));
  return validateJob(task, jobId);
}
function pause(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) { reject(new DOMException('Cancelled', 'AbortError')); return; }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('Cancelled', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal.addEventListener('abort', onAbort, { once: true });
  });
}

export const JOB_POLL_TIMEOUT_MS = 5 * 60 * 1000;
const JOB_POLL_INTERVAL_MS = 500;
const JOB_POLL_TIMEOUT_MESSAGE =
  'Job status timed out after five minutes. The server may still be working.';

/**
 * Bound the entire polling operation, including a stalled fetch, to five minutes.
 * User cancellation has precedence over deadline expiry. Each invocation owns
 * one timer and one derived abort signal, both cleaned up on every exit path.
 */
export async function awaitJob(jobId: string, signal: AbortSignal,
                               onUpdate?: (status: JobStatus) => void): Promise<JobAudio> {
  if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
  const pending = new AbortController();
  let timedOut = false;
  const onAbort = () => pending.abort();
  signal.addEventListener('abort', onAbort, { once: true });
  const deadline = setTimeout(() => {
    timedOut = true;
    pending.abort();
  }, JOB_POLL_TIMEOUT_MS);
  try {
    while (true) {
      if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      if (timedOut) throw new Error(JOB_POLL_TIMEOUT_MESSAGE);
      const job = await getJob(jobId, pending.signal);
      if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      if (timedOut) throw new Error(JOB_POLL_TIMEOUT_MESSAGE);
      onUpdate?.(job.status);
      if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
      if (job.status === 'succeeded') {
        if (!job.result) throw new Error('Completed job has no audio result');
        validateAudioPath(job.result.audio_url);
        if (job.result.audio_url !== '/v1/generations/' + job.result.id + '/audio')
          throw new Error('Completed job returned a different local audio URL');
        return job.result;
      }
      if (job.status === 'failed') throw new Error(job.error || 'Audio generation failed.');
      if (job.status === 'cancelled') throw new Error('Audio generation was cancelled.');
      await pause(JOB_POLL_INTERVAL_MS, pending.signal);
    }
  } catch (error) {
    if (signal.aborted) throw new DOMException('Cancelled', 'AbortError');
    if (timedOut) throw new Error(JOB_POLL_TIMEOUT_MESSAGE);
    throw error;
  } finally {
    clearTimeout(deadline);
    signal.removeEventListener('abort', onAbort);
    pending.abort();
  }
}

export function submitMusicJob(input: MusicRenderInput, options: JobSubmissionOptions = {}): Promise<AudioJob> {
  validateScoreInput(input);
  const error = validateMusicMix(input.mix);
  if (error) throw new Error(error);
  validateAutomation(input);
  return submitJob('music', input, options);
}

export async function planMusic(input: MusicScoreInput, signal?: AbortSignal): Promise<MusicPlan> {
  validateScoreInput(input);
  return withJobDeadline(signal, async pending => {
    const value = await request<unknown>('/v1/music/plan', {
      method: 'POST', body: JSON.stringify(input), signal: pending, redirect: 'error', cache: 'no-store',
    });
    return parseMusicPlan(value, input);
  });
}

/** A digest-pinned binary download, never a server-provided URL. */
export async function exportMusic(input: MusicScoreInput, format: 'json' | 'midi', signal?: AbortSignal): Promise<Blob> {
  validateScoreInput(input);
  if (!input.expected_score_sha256) throw new Error('Plan the music before exporting.');
  if (format !== 'json' && format !== 'midi') throw new Error('Unsupported music export.');
  return withJobDeadline(signal, async pending => {
    const response = await fetch('/v1/music/' + (format === 'json' ? 'score' : 'midi'), {
      method: 'POST', body: JSON.stringify(input), signal: pending, redirect: 'error', cache: 'no-store',
      credentials: 'omit', headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) throw new ApiError('Music export failed (HTTP ' + response.status + ')', response.status);
    const type = format === 'json' ? 'application/json' : 'audio/midi';
    if (response.headers.get('content-type')?.split(';')[0] !== type ||
        response.headers.get('X-Sonumio-Score-SHA256') !== input.expected_score_sha256 ||
        !isDigest(response.headers.get('X-Sonumio-Content-SHA256')))
      throw new Error('Invalid music export receipt.');
    const limit = 1024 * 1024;
    const size = Number(response.headers.get('content-length'));
    if (!Number.isInteger(size) || size < 1 || size > limit || !response.body)
      throw new Error('Invalid music export length.');
    const reader = response.body.getReader();
    const buffer = new Uint8Array(size);
    let offset = 0;
    try {
      while (true) {
        pending.throwIfAborted();
        const { value, done } = await reader.read();
        if (done) break;
        if (offset + value.byteLength > size) throw new Error('Music export exceeds its declared size.');
        buffer.set(value, offset);
        offset += value.byteLength;
      }
    } finally {
      try { await reader.cancel(); } catch { /* stream may have already failed */ }
      reader.releaseLock();
    }
    if (offset !== size) throw new Error('Truncated music export.');
    const digest = [...new Uint8Array(await crypto.subtle.digest('SHA-256', buffer))]
      .map(byte => byte.toString(16).padStart(2, '0')).join('');
    if (digest !== response.headers.get('X-Sonumio-Content-SHA256'))
      throw new Error('Music export checksum mismatch.');
    return new Blob([buffer], { type });
  });
}


export async function planMusicAutomation(input:MusicRenderInput,signal?:AbortSignal):Promise<AutomationReceipt> {
  validateScoreInput(input);
  const error=validateMusicMix(input.mix);
  if(error)throw new Error(error);
  validateAutomation(input);
  if(!input.expected_score_sha256)throw new Error('Validate the score before planning automation.');
  return withJobDeadline(signal,async pending=>{
    const raw=await request<unknown>('/v1/music/automation/plan',{
      method:'POST',body:JSON.stringify(input),signal:pending,redirect:'error',cache:'no-store',
    });
    return parseAutomationReceipt(raw,input);
  });
}
export async function exportMusicAutomation(input:MusicRenderInput,signal?:AbortSignal):Promise<Blob> {
  validateScoreInput(input);
  const error=validateMusicMix(input.mix);
  if(error)throw new Error(error);
  validateAutomation(input);
  if(!input.expected_score_sha256||!input.automation?.expected_automation_sha256)
    throw new Error('Validate score and automation before exporting.');
  return withJobDeadline(signal,async pending=>{
    const response=await fetch('/v1/music/automation/export',{
      method:'POST',body:JSON.stringify(input),signal:pending,redirect:'error',cache:'no-store',
      credentials:'omit',headers:{'Content-Type':'application/json'},
    });
    if(!response.ok)throw new ApiError('Automation export failed (HTTP '+response.status+')',response.status);
    if(response.headers.get('content-type')?.split(';')[0]!=='application/json'||
      response.headers.get('X-Sonumio-Score-SHA256')!==input.expected_score_sha256||
      response.headers.get('X-Sonumio-Automation-SHA256')!==input.automation?.expected_automation_sha256||
      !isDigest(response.headers.get('X-Sonumio-Content-SHA256')))
      throw new Error('Invalid automation export receipt.');
    const size=Number(response.headers.get('content-length'));
    if(!Number.isInteger(size)||size<1||size>MAX_AUTOMATION_BYTES||!response.body)
      throw new Error('Invalid automation export length.');
    const reader=response.body.getReader(), buffer=new Uint8Array(size);
    let offset=0;
    try {
      while(true){
        pending.throwIfAborted();
        const {value,done}=await reader.read();
        if(done)break;
        if(offset+value.byteLength>size)throw new Error('Automation export exceeds declared length.');
        buffer.set(value,offset);offset+=value.byteLength;
      }
    }finally{
      try{await reader.cancel();}catch{/* already closed */}
      reader.releaseLock();
    }
    if(offset!==size)throw new Error('Truncated automation export.');
    const digest=[...new Uint8Array(await crypto.subtle.digest('SHA-256',buffer))]
      .map(x=>x.toString(16).padStart(2,'0')).join('');
    if(digest!==response.headers.get('X-Sonumio-Content-SHA256'))throw new Error('Automation export checksum mismatch.');
    return new Blob([buffer],{type:'application/json'});
  });
}
