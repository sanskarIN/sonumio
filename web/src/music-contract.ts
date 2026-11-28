/** Local original-music controls shared by the studio and its API client. */
export const TRACKS = ['keys', 'pad', 'bass', 'lead', 'pluck', 'kick', 'snare', 'hat'] as const;
export type Track = typeof TRACKS[number];
export const STYLES = ['calm', 'cinematic', 'rhythmic'] as const;
export const MODES = ['major', 'minor'] as const;
export const MUSIC_ENGINE = 'original-procedural-arrangement-v1';
export const MAX_PREVIEW_SECONDS = 30;
export const SAMPLE_RATE = 24000;
export const DEFAULT_GAINS = [.20, .14, .26, .25, .16, .23, .18, .12] as const;

export type Composition = {
  style: typeof STYLES[number]; root: number; mode: typeof MODES[number];
  bpm: number; bars: number; seed: number; swing: number;
};
export type MusicMix = {
  gains: number[]; mute: Track[]; solo: Track[]; room: number; peak_ceiling: number;
};
export type MusicScoreInput = { composition: Composition; expected_score_sha256?: string };
export type MusicRenderInput = MusicScoreInput & { mix: MusicMix };
export type Section = { kind: string; first_bar: number; bars: number };
export type TrackSummary = { events: number; active_fraction: number; pitch_min: number | null; pitch_max: number | null; max_polyphony: number };
export type MusicPlan = {
  engine_id: string; learned_model: false; external_recordings_used: false;
  disclaimer: string; composition: Composition; score_sha256: string; sample_rate: number;
  frames: number; duration_seconds: number; bars: number; events: number;
  sections: Section[]; harmony: number[]; tracks: Record<Track, TrackSummary>;
};

export function defaultComposition(): Composition {
  return { style: 'cinematic', root: 60, mode: 'minor', bpm: 120, bars: 4, seed: 0, swing: 0 };
}
export function defaultMusicMix(): MusicMix {
  return { gains: [...DEFAULT_GAINS], mute: [], solo: [], room: .15, peak_ceiling: .9 };
}
export const isDigest = (value: unknown): value is string =>
  typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

const bounded = (value: unknown, lo: number, hi: number): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= lo && value <= hi;
const integer = (value: unknown, lo: number, hi: number): value is number =>
  bounded(value, lo, hi) && Number.isInteger(value);
const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export function previewFrames(composition: Composition): number {
  return Math.floor((2 * composition.bars * 4 * 60 * SAMPLE_RATE + composition.bpm) / (2 * composition.bpm));
}

export function validateComposition(c: Composition): string | null {
  if (!STYLES.includes(c.style)) return 'Unknown composition style.';
  if (!MODES.includes(c.mode)) return 'Unknown musical mode.';
  if (!integer(c.root, 48, 71)) return 'Root must be a MIDI note from 48 to 71.';
  if (!integer(c.bpm, 60, 160)) return 'Tempo must be an integer from 60 to 160 BPM.';
  if (!integer(c.bars, 4, 16)) return 'Choose 4–16 bars.';
  if (!integer(c.seed, 0, 2 ** 32 - 1)) return 'Seed must be a 32-bit unsigned integer.';
  if (!bounded(c.swing, 0, .6)) return 'Swing must be between 0 and 0.6.';
  if (previewFrames(c) > SAMPLE_RATE * MAX_PREVIEW_SECONDS) return 'Preview cannot exceed 30 seconds. Reduce bars or increase tempo.';
  return null;
}
function validTracks(names: unknown): names is Track[] {
  return Array.isArray(names) && names.length <= TRACKS.length &&
    names.every(name => TRACKS.includes(name as Track)) && new Set(names).size === names.length;
}
export function validateMusicMix(m: MusicMix): string | null {
  if (!Array.isArray(m.gains) || m.gains.length !== TRACKS.length ||
      !m.gains.every(gain => bounded(gain, 0, 2))) return 'Provide eight gains between 0 and 2.';
  if (!validTracks(m.mute) || !validTracks(m.solo)) return 'Invalid or repeated track selection.';
  if (m.mute.some(track => m.solo.includes(track))) return 'A track cannot be both muted and soloed.';
  if (!bounded(m.room, 0, 1)) return 'Room amount must be between 0 and 1.';
  if (!bounded(m.peak_ceiling, .05, .99)) return 'Peak ceiling must be between 0.05 and 0.99.';
  return null;
}
export function validateScoreInput(input: MusicScoreInput): void {
  const error = validateComposition(input.composition);
  if (error) throw new Error(error);
  if (input.expected_score_sha256 !== undefined && !isDigest(input.expected_score_sha256))
    throw new Error('Invalid expected score digest.');
}
export function compositionKey(c: Composition): string {
  return JSON.stringify([c.style, c.root, c.mode, c.bpm, c.bars, c.seed, c.swing]);
}

export function parseMusicPlan(value: unknown, input: MusicScoreInput): MusicPlan {
  validateScoreInput(input);
  if (!record(value) || value.engine_id !== MUSIC_ENGINE || value.learned_model !== false ||
      value.external_recordings_used !== false || typeof value.disclaimer !== 'string' ||
      !isDigest(value.score_sha256) || !record(value.composition))
    throw new Error('Invalid music plan provenance.');
  const plan = value as MusicPlan;
  if (validateComposition(plan.composition) ||
      compositionKey(plan.composition) !== compositionKey(input.composition) ||
      (input.expected_score_sha256 !== undefined && plan.score_sha256 !== input.expected_score_sha256))
    throw new Error('Music plan does not match the requested composition.');
  if (plan.sample_rate !== SAMPLE_RATE || plan.frames !== previewFrames(input.composition) ||
      plan.duration_seconds !== plan.frames / SAMPLE_RATE || plan.bars !== input.composition.bars ||
      !integer(plan.events, 1, 4096))
    throw new Error('Invalid music plan geometry.');
  if (!Array.isArray(plan.harmony) || plan.harmony.length !== plan.bars ||
      !plan.harmony.every(degree => integer(degree, 0, 6)) || plan.harmony.at(-1) !== 0)
    throw new Error('Invalid music harmony.');
  if (!Array.isArray(plan.sections) || !plan.sections.length || plan.sections.length > 16)
    throw new Error('Invalid music sections.');
  let cursor = 0;
  for (const section of plan.sections) {
    if (!record(section) || !['intro', 'theme', 'variation', 'break', 'finale'].includes(section.kind) ||
        section.first_bar !== cursor || !integer(section.bars, 1, plan.bars))
      throw new Error('Invalid music section geometry.');
    cursor += section.bars;
  }
  if (cursor !== plan.bars || plan.sections.at(-1)?.kind !== 'finale')
    throw new Error('Music sections do not cover the arrangement.');
  if (!record(plan.tracks) || Object.keys(plan.tracks).length !== TRACKS.length)
    throw new Error('Invalid music track summaries.');
  let total = 0;
  for (const track of TRACKS) {
    const summary = plan.tracks[track];
    if (!record(summary) || !integer(summary.events, 0, plan.events) ||
        !bounded(summary.active_fraction, 0, 1) ||
        !integer(summary.max_polyphony, 0, 4))
      throw new Error('Invalid music track summary.');
    if (summary.events === 0 ? summary.pitch_min !== null || summary.pitch_max !== null :
        !integer(summary.pitch_min, 24, 96) || !integer(summary.pitch_max, 24, 96) || summary.pitch_min > summary.pitch_max)
      throw new Error('Invalid music track pitch range.');
    total += summary.events;
  }
  if (total !== plan.events) throw new Error('Music track event counts do not match.');
  return plan;
}
