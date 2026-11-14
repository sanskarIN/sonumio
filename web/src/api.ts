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
};
export type SceneInput = { duration_seconds: number; layers: SceneLayer[] };
export type GeneratedAudio = {
  id: string; audio_url: string; sample_rate: number; duration_seconds: number;
  disclaimer: string; metrics: Record<string, number>; model_id?: string; engine_id?: string;
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
    if (!Number.isInteger(layer.seed) || layer.seed < 0 || layer.seed >= 2 ** 32)
      return 'Invalid seed in layer ' + n + '.';
    if (!Number.isFinite(layer.fade_in_seconds) || !Number.isFinite(layer.fade_out_seconds) ||
      layer.fade_in_seconds < 0 || layer.fade_out_seconds < 0 ||
      layer.fade_in_seconds + layer.fade_out_seconds > layer.duration_seconds)
      return 'Layer ' + n + ' fades exceed its duration.';
  }
  return null;
}
