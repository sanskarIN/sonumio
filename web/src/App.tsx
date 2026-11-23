import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react';
import {
  submitGenerationJob, submitSceneJob, awaitJob, cancelJob, getHealth, getModels, validateScene,
  parseScenePlan, serializeScenePlan, MAX_SCENE_JSON_BYTES,
  type JobStatus,
  type GeneratedAudio, type ModelEntry, type SceneLayer,
} from './api';
import { loadLocalWaveform, type WaveformPreview } from './waveform';

const presets = [
  'Gentle rain in a peaceful forest',
  'Distant thunder and heavy rain',
  'Waves washing onto the ocean shore',
  'Soft wind and birds in a forest',
];
function makeLayer(index = 0): SceneLayer {
  return { prompt: index === 0 ? 'rain' : 'wind', start_seconds: index === 0 ? 0 : 1,
    duration_seconds: 3, gain_db: index === 0 ? 0 : -6,
    fade_in_seconds: 0.02, fade_out_seconds: 0.05, seed: index, pan: 0 };
}
const numberValue = (value: string) => Number(value);
export function App() {
  const [mode, setMode] = useState<'sound' | 'scene'>('sound');
  const [prompt, setPrompt] = useState(presets[0]);
  const [duration, setDuration] = useState(5);
  const [seed, setSeed] = useState(42);
  const [sceneDuration, setSceneDuration] = useState(6);
  const [layers, setLayers] = useState<SceneLayer[]>([makeLayer()]);
  const [sceneNotice, setSceneNotice] = useState('');
  const sceneFileInput = useRef<HTMLInputElement | null>(null);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [connected, setConnected] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [jobStatus, setJobStatus] = useState<JobStatus>('queued');
  const [error, setError] = useState('');
  const [result, setResult] = useState<GeneratedAudio | null>(null);
  const [waveform, setWaveform] = useState<WaveformPreview | null>(null);
  const [waveformStatus, setWaveformStatus] = useState<'idle' | 'loading' | 'unavailable'>('idle');
  const controller = useRef<AbortController | null>(null);
  const jobId = useRef<string | null>(null);

  useEffect(() => {
    const pending = new AbortController();
    Promise.all([getHealth(pending.signal), getModels(pending.signal)])
      .then(([healthy, entries]) => { setConnected(healthy); setModels(entries); })
      .catch(() => { if (!pending.signal.aborted) setConnected(false); });
    return () => { pending.abort(); controller.current?.abort(); };
  }, []);

  // Always dispose of the previous fetch when the rendered audio changes,
  // is cancelled, or the studio unmounts. Never retain raw PCM waveform bytes.
  const previewUrl = result?.audio_url ?? null;
  useEffect(() => {
    if (!previewUrl) {
      setWaveform(null);
      setWaveformStatus('idle');
      return;
    }
    const pending = new AbortController();
    setWaveform(null);
    setWaveformStatus('loading');
    void loadLocalWaveform(previewUrl, pending.signal)
      .then(preview => {
        if (!pending.signal.aborted) {
          setWaveform(preview);
          setWaveformStatus('idle');
        }
      })
      .catch(() => {
        if (!pending.signal.aborted) setWaveformStatus('unavailable');
      });
    return () => pending.abort();
  }, [previewUrl]);

  const activeModel = models.find(model => model.id === 'procedural-sfx-v0');
  const sceneError = validateScene({ duration_seconds: sceneDuration, layers });
  const sceneExportHref = sceneError ? '' :
    'data:application/json;charset=utf-8,' +
    encodeURIComponent(serializeScenePlan({ duration_seconds: sceneDuration, layers }));

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (busy) return;
    setError(''); setResult(null);
    if (!connected) { setError('Start the local Sonumio API on port 8000 first.'); return; }
    if (mode === 'scene' && sceneError) { setError(sceneError); return; }
    const next = new AbortController();
    controller.current = next;
    setBusy(true);
    setJobStatus('queued');
    try {
      const task = mode === 'sound'
        ? await submitGenerationJob({ prompt, duration_seconds: duration, seed, model_id: 'procedural-sfx-v0' }, next.signal)
        : await submitSceneJob({ duration_seconds: sceneDuration, layers }, next.signal);
      jobId.current = task.id;
      setJobStatus(task.status);
      const response = await awaitJob(task.id, next.signal, setJobStatus);
      if (!next.signal.aborted) setResult(response);
    } catch (problem) {
      if (!next.signal.aborted) setError(problem instanceof Error ? problem.message : 'Audio generation failed.');
    } finally {
      if (controller.current === next) {
        controller.current = null;
        jobId.current = null;
        setBusy(false);
      }
    }
  }
  function switchMode(value: 'sound' | 'scene') {
    if (!busy) { setMode(value); setError(''); setResult(null); }
  }
  function changeLayer(index: number, key: keyof SceneLayer, value: number | string | undefined) {
    setSceneNotice('');
    setLayers(previous => previous.map((layer, i) => i === index ? { ...layer, [key]: value } : layer));
  }
  async function importScene(event: ChangeEvent<HTMLInputElement>) {
    const file = event.currentTarget.files?.[0];
    event.currentTarget.value = '';
    if (!file || busy) return;
    if (file.size > MAX_SCENE_JSON_BYTES) {
      setError('Scene file exceeds the 16 KiB local import limit.');
      setSceneNotice('');
      return;
    }
    try {
      const scene = parseScenePlan(await file.text());
      setSceneDuration(scene.duration_seconds);
      setLayers(scene.layers);
      setResult(null);
      setError('');
      setSceneNotice('Imported ' + file.name.slice(0, 80) + ' locally. Ready to edit or render.');
    } catch (problem) {
      setSceneNotice('');
      setError(problem instanceof Error ? problem.message : 'Could not read scene JSON.');
    }
  }
  function cancel() {
    const identifier = jobId.current;
    controller.current?.abort();
    controller.current = null;
    jobId.current = null;
    if (identifier) void cancelJob(identifier).catch(() => {});
    setBusy(false);
    setJobStatus('cancelling');
    setError('Cancellation requested. Active rendering cannot stop immediately and may finish before cancellation takes effect.');
  }
  return <div className="shell">
    <header className="topbar">
      <a href="#main" className="brand" aria-label="Sonumio audio studio">◖◗ <strong>sonumio<span>.</span></strong></a>
      <div className="top-right"><small>LOCAL AUDIO STUDIO</small>
        <span role="status" className={'connection ' + (connected ? 'online' : 'offline')}>
          <i aria-hidden="true" /> {connected === null ? 'Connecting' : connected ? 'API connected' : 'API offline'}
        </span>
      </div>
    </header>
    <main id="main">
      <section className="hero" aria-labelledby="headline">
        <p className="overline purple">✧ THE OPEN AUDIO LAB</p>
        <h1 id="headline">Imagine it.<br/><em>Hear it.</em></h1>
        <p className="hero-description">Turn a soundscape idea into an audio file. Create individual sounds or build a scene, one layer at a time.</p>
        <div className="disclosure">◈ Development preview: the connected engine uses <strong>procedural synthesis, not a trained generative AI model.</strong></div>
      </section>
      <div className="layout">
        <section className="panel workspace" aria-labelledby="workspace-heading">
          <div className="section-heading"><div><span className="overline">WORKSPACE / 01</span><h2 id="workspace-heading">Audio studio</h2></div><span className="engine-pill">DSP ENGINE</span></div>
          <div className="tabs" aria-label="Audio creation mode">
            <button type="button" className={mode === 'sound' ? 'active' : ''} aria-pressed={mode === 'sound'} onClick={() => switchMode('sound')}>✳ Generate sound</button>
            <button type="button" className={mode === 'scene' ? 'active' : ''} aria-pressed={mode === 'scene'} onClick={() => switchMode('scene')}>≋ Build a scene</button>
          </div>
          <form onSubmit={submit}>
            {mode === 'sound' ? <>
              <label htmlFor="prompt" className="field-heading">Describe your sound</label>
              <textarea id="prompt" rows={4} maxLength={800} required value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="Soft rain in a quiet forest..." />
              <p className="hint">Supported: rain, thunder, wind, birds, water and ambient tones in English or Hindi. {prompt.length}/800 characters</p>
              <div className="presets"><span className="field-heading">TRY A PROMPT</span><div className="chips">{presets.map(item => <button key={item} type="button" onClick={() => setPrompt(item)}>{item}</button>)}</div></div>
              <div className="field-grid">
                <label>Duration <small>0.5–15 seconds</small><input type="number" min="0.5" max="15" step="0.5" required value={duration} onChange={e => setDuration(numberValue(e.target.value))}/></label>
                <label>Random seed <small>Reproducible audio</small><input type="number" min="0" max="4294967295" step="1" required value={seed} onChange={e => setSeed(numberValue(e.target.value))}/></label>
              </div>
            </> : <>
              <p className="hint">Arrange up to eight procedural layers on a shared timeline. Every layer must end before the scene finishes. Set pan from -1 (left) to +1 (right) for stereo; add an optional end pan to move the sound across the stereo field. Set an end gain to swell or fade a layer smoothly in dB. All-center scenes remain mono.</p>
              <div className="field-grid"><label>Scene duration <small>0.5–30 seconds</small><input type="number" min="0.5" max="30" step="0.5" required value={sceneDuration} onChange={e => { setSceneNotice(''); setSceneDuration(numberValue(e.target.value)); }}/></label></div>
              <div className="scene-file-actions">
                <button type="button" className="secondary-button" disabled={busy} onClick={() => sceneFileInput.current?.click()}>Import scene JSON</button>
                <input ref={sceneFileInput} className="scene-file-input" type="file" accept=".json,application/json" aria-label="Select local scene JSON file" disabled={busy} onChange={importScene}/>
                {!sceneError && <a className="secondary-button" download="sonumio-scene.json" href={sceneExportHref}>Export scene JSON</a>}
              </div>
              {sceneNotice && <p className="hint" role="status">{sceneNotice}</p>}
              <div className="layer-heading"><span className="field-heading">SOUND LAYERS ({layers.length}/8)</span><button type="button" className="secondary-button" disabled={layers.length >= 8} onClick={() => setLayers(existing => [...existing, makeLayer(existing.length)])}>+ Add layer</button></div>
              <div className="layers">{layers.map((layer, index) => <fieldset key={index} className="layer">
                <legend>Layer {index + 1}</legend>
                <button type="button" className="remove" aria-label={'Remove layer ' + (index + 1)} disabled={layers.length === 1} onClick={() => setLayers(existing => existing.filter((_, i) => i !== index))}>Remove</button>
                <label>Sound prompt<input maxLength={800} required value={layer.prompt} onChange={e => changeLayer(index, 'prompt', e.target.value)} /></label>
                <div className="field-grid">
                  <label>Starts (s)<input type="number" min="0" max="30" step=".1" required value={layer.start_seconds} onChange={e => changeLayer(index, 'start_seconds', numberValue(e.target.value))}/></label>
                  <label>Length (s)<input type="number" min=".5" max="15" step=".1" required value={layer.duration_seconds} onChange={e => changeLayer(index, 'duration_seconds', numberValue(e.target.value))}/></label>
                  <label>Gain start (dB)<input type="number" min="-60" max="12" step="0.5" required value={layer.gain_db} onChange={e => changeLayer(index, 'gain_db', numberValue(e.target.value))}/></label>
                  <label>Gain end (dB) <small>Optional · blank keeps level fixed</small><input type="number" min="-60" max="12" step="0.5" value={layer.gain_end_db ?? ''} placeholder="Fixed" onChange={e => changeLayer(index, 'gain_end_db', e.target.value === '' ? undefined : numberValue(e.target.value))}/></label>
                  <label>Pan start <small>-1 left · 0 center · +1 right</small><input type="number" min="-1" max="1" step="0.1" required value={layer.pan ?? 0} onChange={e => changeLayer(index, 'pan', numberValue(e.target.value))}/></label>
                  <label>Pan end <small>Optional · blank keeps pan fixed</small><input type="number" min="-1" max="1" step="0.1" value={layer.pan_end ?? ''} placeholder="Fixed" onChange={e => changeLayer(index, 'pan_end', e.target.value === '' ? undefined : numberValue(e.target.value))}/></label>
                  <label>Seed<input type="number" min="0" max="4294967295" required value={layer.seed} onChange={e => changeLayer(index, 'seed', numberValue(e.target.value))}/></label>
                </div>
              </fieldset>)}</div>
              {sceneError && <p role="status" className="warning">{sceneError}</p>}
            </>}
            <button className="primary-button" type="submit" disabled={busy || connected !== true || (mode === 'scene' && Boolean(sceneError))}>
              {busy ? (jobStatus === 'queued' ? 'Waiting in local queue…' : 'Generating audio…') : mode === 'sound' ? '✦ Generate sound' : '✦ Render sound scene'} <span>↗</span>
            </button>
            {busy && <button type="button" className="cancel" onClick={cancel}>Cancel request</button>}
            <p className="footnote">Local Sonumio API. Audio is not uploaded to an external provider.</p>
          </form>
        </section>
        <aside className="right-column" aria-label="Audio result and model information">
          <section className="panel preview" aria-labelledby="preview-heading">
            <span className="overline">OUTPUT / 02</span><h2 id="preview-heading">Listen & export</h2>
            {error && <div role="alert" className="warning">{error}</div>}
            {result ? <div className="ready" aria-live="polite">
              {waveform ? <div className="wave actual-wave" role="img"
                aria-label={'Actual PCM waveform with ' + waveform.peaks.length + ' peak-amplitude bars'}>
                {waveform.peaks.map((amplitude, i) => <i key={i}
                  style={{ height: Math.max(2, amplitude * 100) + '%' }} />)}
              </div> : <div className="wave wave-placeholder" role="status">
                {waveformStatus === 'loading'
                  ? 'Reading the generated WAV waveform…'
                  : 'Waveform unavailable; the audio player and download are still available.'}
              </div>}
              <span className="overline mint">AUDIO READY</span><h3>Your sound is ready.</h3>
              <audio controls preload="none" src={result.audio_url} aria-label="Generated WAV audio preview">Your browser cannot play WAV files.</audio>
              <a className="download" download="sonumio.wav" href={result.audio_url}>↓ Download WAV</a>
              <p className="hint">{result.duration_seconds.toFixed(2)}s · {(result.sample_rate / 1000).toFixed(0)} kHz {result.channels === 2 ? 'stereo' : 'mono'} WAV</p>
              {waveform && <p className="hint">Waveform: measured PCM sample peaks, not a loudness or quality rating.</p>}
              <p className="hint">{result.disclaimer}</p>
            </div> : <div className="empty" aria-live="polite">
              <div className="wave" aria-hidden="true">{Array.from({length: 19}, (_, i) => <i key={i} style={{height: (16 + Math.abs(Math.sin(i * 1.3)) * 74) + '%'}}/>)}</div>
              <h3>{busy ? (jobStatus === 'queued' ? 'Waiting for the renderer…' : 'Creating your sound…') : 'Your next sound starts here.'}</h3>
              <p>{busy ? 'Queued jobs are processed one at a time on this device.' : 'Describe a sound, generate, listen and download your WAV.'}</p>
            </div>}
          </section>
          <section className="panel provenance" aria-labelledby="provenance-heading">
            <span className="overline">TRANSPARENCY / 03</span><h2 id="provenance-heading">Know your engine</h2>
            <div className="model"><span aria-hidden="true">◈</span><div><strong>{activeModel?.id ?? 'procedural-sfx-v0'}</strong><small>{activeModel?.model_kind ?? 'Procedural DSP — NOT AI'}</small></div></div>
            <dl><div><dt>Technology</dt><dd>Digital signal processing</dd></div><div><dt>Source license</dt><dd>Apache-2.0</dd></div><div><dt>Sonumio-1</dt><dd>Research stage</dd></div></dl>
            {models.some(entry => entry.id === 'meta-musicgen-small') && <p className="hint">Third-party MusicGen is in the registry but not available in this preview. Its weights have separate non-commercial licensing.</p>}
          </section>
        </aside>
      </div>
    </main>
    <footer><span>SONUMIO / IMAGINE IT. HEAR IT.</span><span>Open source · Local-first · Development preview</span></footer>
  </div>;
}
