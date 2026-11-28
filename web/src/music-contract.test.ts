import { describe, expect, it } from 'vitest';
import {
  TRACKS, MUSIC_ENGINE, defaultComposition, defaultMusicMix,
  compositionKey, previewFrames, validateComposition, validateMusicMix, parseMusicPlan,
  type Composition, type MusicPlan, type MusicMix,
} from './music-contract';

const digest = 'a'.repeat(64);
function fixture(): MusicPlan {
  const composition = defaultComposition();
  const frames = previewFrames(composition);
  return {
    engine_id: MUSIC_ENGINE, learned_model: false, external_recordings_used: false,
    disclaimer: 'Procedural', composition, score_sha256: digest, sample_rate: 24000,
    frames, duration_seconds: frames / 24000, bars: 4, events: 8,
    sections: [{ kind: 'intro', first_bar: 0, bars: 1 },
      { kind: 'theme', first_bar: 1, bars: 2 }, { kind: 'finale', first_bar: 3, bars: 1 }],
    harmony: [0, 4, 4, 0],
    tracks: Object.fromEntries(TRACKS.map(track => [track, {
      events: 1, active_fraction: .2, pitch_min: 60, pitch_max: 60, max_polyphony: 1,
    }])) as MusicPlan['tracks'],
  };
}

describe('music composition controls', () => {
  it('accepts default and exact 30-second preview', () => {
    expect(validateComposition(defaultComposition())).toBeNull();
    const exact = { ...defaultComposition(), bars: 15 };
    expect(previewFrames(exact)).toBe(720000);
    expect(validateComposition(exact)).toBeNull();
    expect(validateComposition({ ...exact, bpm: 119 })).toContain('30 seconds');
  });
  it.each<Partial<Composition>>([
    { bpm: 59 }, { bpm: 161 }, { bpm: 120.5 }, { bpm: NaN },
    { seed: 2 ** 32 }, { seed: -1 }, { seed: .1 }, { root: 72 },
    { root: 47 }, { bars: 3 }, { bars: 17 }, { swing: Infinity },
    { swing: -.1 }, { swing: .61 },
  ])('rejects invalid composition %j', override => {
    expect(validateComposition({ ...defaultComposition(), ...override })).not.toBeNull();
  });
  it('creates independent defaults and stable keys', () => {
    const a = defaultComposition();
    const b = defaultComposition();
    expect(compositionKey(a)).toBe(compositionKey(b));
    a.seed++;
    expect(compositionKey(a)).not.toBe(compositionKey(b));
    expect(defaultMusicMix().gains).not.toBe(defaultMusicMix().gains);
  });
});

describe('music mix validation', () => {
  it('supports mute, solo and independent gain values', () => {
    expect(validateMusicMix(defaultMusicMix())).toBeNull();
    expect(validateMusicMix({ ...defaultMusicMix(), mute: ['pad'], solo: ['lead'] })).toBeNull();
    expect(validateMusicMix({ ...defaultMusicMix(), mute: [...TRACKS] })).toBeNull();
  });
  it.each<Partial<MusicMix>>([
    { gains: [] }, { gains: Array(8).fill(3) }, { gains: Array(8).fill(-1) },
    { gains: Array(8).fill(NaN) }, { mute: ['keys', 'keys'] },
    { mute: ['lead'], solo: ['lead'] }, { room: Infinity },
    { peak_ceiling: .01 }, { peak_ceiling: 1 },
  ])('rejects invalid mix %j', override => {
    expect(validateMusicMix({ ...defaultMusicMix(), ...override })).not.toBeNull();
  });
});

describe('music plan response validation', () => {
  it('checks score digest, composition, section coverage and track totals', () => {
    const data = fixture();
    expect(parseMusicPlan(data, { composition: data.composition, expected_score_sha256: digest })).toBe(data);
    expect(() => parseMusicPlan(data, { composition: data.composition, expected_score_sha256: 'b'.repeat(64) })).toThrow('match');
    expect(() => parseMusicPlan(data, { composition: { ...data.composition, seed: 2 } })).toThrow('match');
  });
  it.each([
    (p: MusicPlan) => ({ ...p, engine_id: 'third-party' }),
    (p: MusicPlan) => ({ ...p, learned_model: true }),
    (p: MusicPlan) => ({ ...p, external_recordings_used: true }),
    (p: MusicPlan) => ({ ...p, score_sha256: 'bad' }),
    (p: MusicPlan) => ({ ...p, frames: p.frames + 1 }),
    (p: MusicPlan) => ({ ...p, sample_rate: 44100 }),
    (p: MusicPlan) => ({ ...p, duration_seconds: -1 }),
    (p: MusicPlan) => ({ ...p, harmony: [0, 4, 4, 4] }),
    (p: MusicPlan) => ({ ...p, sections: p.sections.slice(0, 2) }),
    (p: MusicPlan) => ({ ...p, sections: [{ kind: 'unknown', first_bar: 0, bars: 4 }] }),
    (p: MusicPlan) => ({ ...p, sections: [{ kind: 'finale', first_bar: 1, bars: 4 }] }),
    (p: MusicPlan) => ({ ...p, tracks: { ...p.tracks, fake: p.tracks.keys } }),
    (p: MusicPlan) => ({ ...p, tracks: { ...p.tracks, lead: { ...p.tracks.lead, events: 2 } } }),
    (p: MusicPlan) => ({ ...p, tracks: { ...p.tracks, lead: { ...p.tracks.lead, active_fraction: 2 } } }),
    (p: MusicPlan) => ({ ...p, tracks: { ...p.tracks, lead: { ...p.tracks.lead, pitch_min: 90, pitch_max: 60 } } }),
  ])('rejects corrupt response %s', mutate => {
    const data = fixture();
    expect(() => parseMusicPlan(mutate(data), { composition: data.composition })).toThrow();
  });
});
