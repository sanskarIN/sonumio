import { describe, expect, it } from 'vitest';
import { MAX_SCENE_JSON_BYTES, parseScenePlan, serializeScenePlan, type SceneInput } from './api';

const scene: SceneInput = {
  duration_seconds: 4,
  layers: [
    { prompt: 'rain and wind', start_seconds: 0, duration_seconds: 3,
      gain_db: -4, gain_end_db: -26, fade_in_seconds: 0.02,
      fade_out_seconds: 0.05, seed: 12, pan: -0.5, pan_end: 0.9 },
    { prompt: 'birds', start_seconds: 0.5, duration_seconds: 2,
      gain_db: -12, fade_in_seconds: 0, fade_out_seconds: 0.1,
      seed: 32, pan: 0 },
  ],
};

describe('local scene JSON import/export', () => {
  it('roundtrips precise timing, seeds, pans and end gain without changes', () => {
    const serialized = serializeScenePlan(scene);
    expect(JSON.parse(serialized).format).toBe('sonumio-scene-v1');
    expect(parseScenePlan(serialized)).toEqual(scene);
    expect(serialized.length).toBeLessThan(MAX_SCENE_JSON_BYTES);
  });

  it('imports minimal backwards-compatible CLI scene JSON with defaults', () => {
    const imported = parseScenePlan(JSON.stringify({
      duration_seconds: 2, layers: [{ prompt: 'rain', duration_seconds: 1 }],
    }));
    expect(imported).toEqual({ duration_seconds: 2, layers: [{
      prompt: 'rain', start_seconds: 0, duration_seconds: 1, gain_db: 0,
      gain_end_db: undefined, fade_in_seconds: 0.02, fade_out_seconds: 0.05,
      seed: 0, pan: 0, pan_end: undefined,
    }] });
  });

  it('rejects malformed, oversized, unsupported, and unrelated payloads', () => {
    expect(() => parseScenePlan('not JSON')).toThrow('not valid JSON');
    expect(() => parseScenePlan('x'.repeat(MAX_SCENE_JSON_BYTES + 1))).toThrow('16 KiB');
    expect(() => parseScenePlan('[]')).toThrow('layers array');
    expect(() => parseScenePlan('{"format":"unknown","layers":[]}')).toThrow('Unsupported');
    expect(() => parseScenePlan('{"duration_seconds":3,"layers":[]}')).toThrow('1–8');
  });

  it('rejects invalid primitives and bad timing instead of coercing', () => {
    for (const bad of [
      { ...scene, duration_seconds: '4' },
      { ...scene, layers: [{ ...scene.layers[0], gain_end_db: '12' }] },
      { ...scene, layers: [{ ...scene.layers[0], seed: true }] },
      { ...scene, layers: [{ ...scene.layers[0], prompt: {} }] },
      { ...scene, layers: [{ ...scene.layers[0], pan_end: 9 }] },
      { ...scene, layers: [{ ...scene.layers[0], start_seconds: 3.5 }] },
    ]) {
      expect(() => parseScenePlan(JSON.stringify(bad))).toThrow();
    }
  });

  it('exports only supported schema keys and rejects invalid editor scenes', () => {
    const payload = JSON.parse(serializeScenePlan({
      ...scene,
      layers: [{ ...scene.layers[0], unexpected: 'not exported' } as SceneInput['layers'][number]],
    }));
    expect(payload.layers[0]).not.toHaveProperty('unexpected');
    expect(() => serializeScenePlan({ ...scene, duration_seconds: -1 })).toThrow();
  });
});
