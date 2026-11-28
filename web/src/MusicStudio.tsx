import { useEffect, useRef, useState, type FormEvent } from 'react';
import { parseStemResult, type StemPlan, type StemMode } from './music-stem-contract';
import { submitStemJob, planStem, downloadStem } from './api';
import { ScoreNoteEditor } from './ScoreNoteEditor';
import { MusicAutomationEditor } from './MusicAutomationEditor';
import { parseAutomationFile, MAX_AUTOMATION_BYTES, type AutomationLane, type AutomationReceipt } from './music-automation-contract';
import type { NoteEdits } from './music-note-editor';
import { awaitJob, cancelJob, exportMusic, getJob, isClientJobId, isRejectedJobSubmission,
  planMusic, submitMusicJob, planMusicAutomation, exportMusicAutomation, type JobAudio, type JobStatus } from './api';
import { TRACKS, STYLES, MODES, defaultComposition, defaultMusicMix, compositionKey,
  previewFrames, validateComposition, validateMusicMix, SAMPLE_RATE,
  type Composition, type MusicMix, type MusicPlan, type Track } from './music-contract';

const STORAGE_KEY = 'sonumio.music.last-job-id';
const TRACK_LABELS: Record<Track, string> = {
  keys: 'Keys', pad: 'Pad', bass: 'Bass', lead: 'Lead', pluck: 'Pluck',
  kick: 'Kick', snare: 'Snare', hat: 'Hi-hat',
};
const ROOT_NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
type Activity = 'planning' | 'rendering' | 'recovering' | 'exporting' | 'cancelling';

function lastMusicJob(): string | null {
  try {
    const id = sessionStorage.getItem(STORAGE_KEY);
    return id && isClientJobId(id) ? id : null;
  } catch { return null; }
}
function numberInput(raw: string): number {
  return raw.trim() === '' ? NaN : Number(raw);
}
function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Music studio request failed.';
}

export function MusicStudio({ connected }: { connected: boolean | null }) {
  const [composition, setComposition] = useState(defaultComposition);
  const [mix, setMix] = useState(defaultMusicMix);
  const [lanes,setLanes] = useState<AutomationLane[]>([]);
  const [automationReceipt,setAutomationReceipt] = useState<AutomationReceipt|null>(null);
  const [plan, setPlan] = useState<MusicPlan | null>(null);
  const [basePlan, setBasePlan] = useState<MusicPlan | null>(null);
  const [noteEdits, setNoteEdits] = useState<NoteEdits | null>(null);
  const [result, setResult] = useState<JobAudio | null>(null);
  const [stemMode, setStemMode] = useState<StemMode>('dry');
  const [stemResult, setStemResult] = useState<{audio:JobAudio,plan:StemPlan}|null>(null);
  const [activity, setActivity] = useState<Activity | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [lastJob, setLastJob] = useState(lastMusicJob);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const pending = useRef<AbortController | null>(null);
  const activeJob = useRef<string | null>(null);
  const mounted = useRef(true);
  const downloads = useRef(new Set<string>());
  const cancelPending = useRef<AbortController | null>(null);
  const cancelBusy = useRef(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      pending.current?.abort();
      cancelPending.current?.abort();
      for (const url of downloads.current) URL.revokeObjectURL(url);
      downloads.current.clear();
    };
  }, []);

  const busy = activity !== null;
  const compositionError = validateComposition(composition);
  const mixError = validateMusicMix(mix);
  const currentPlan = plan && compositionKey(plan.composition) === compositionKey(composition) ? plan : null;

  function rememberJob(id: string | null) {
    activeJob.current = id;
    setLastJob(id);
    try {
      if (id) sessionStorage.setItem(STORAGE_KEY, id);
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch { /* Recovery still works for the current page. */ }
  }

  function editComposition<K extends keyof Composition>(key: K, value: Composition[K]) {
    setComposition(previous => ({ ...previous, [key]: value }));
    setPlan(null);
    setBasePlan(null);
    setNoteEdits(null);
    setLanes([]);
    setAutomationReceipt(null);
    setNotice('');
  }

  function editMix(next: MusicMix) {
    setMix(next);
    setAutomationReceipt(null);
    setNotice('');
  }

  function toggleTrack(track: Track, field: 'mute' | 'solo') {
    const opposite = field === 'mute' ? 'solo' : 'mute';
    const current = mix[field];
    const next = current.includes(track) ? current.filter(name => name !== track) : [...current, track];
    editMix({ ...mix, [field]: next, [opposite]: mix[opposite].filter(name => name !== track) });
  }

  async function run(action: Activity, task: (signal: AbortSignal) => Promise<void>) {
    if (pending.current || cancelBusy.current || connected !== true) return;
    const controller = new AbortController();
    activeJob.current = null;
    pending.current = controller;
    setActivity(action);
    setError('');
    setNotice('');
    try {
      await task(controller.signal);
    } catch (problem) {
      if (!controller.signal.aborted && mounted.current) setError(errorMessage(problem));
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        if (mounted.current && !cancelBusy.current) setActivity(null);
      }
    }
  }

  function scoreInput() { return { composition, ...(noteEdits ? { note_edits: noteEdits } : {}) }; }

  function changeNoteEdits(next: NoteEdits | null) {
    setNoteEdits(next);
    setAutomationReceipt(null);
    setPlan(next ? null : basePlan);
    setNotice('');
  }

  async function obtainPlan(signal: AbortSignal) {
    const value = await planMusic(scoreInput(), signal);
    signal.throwIfAborted();
    setPlan(value);
    setAutomationReceipt(null);
    if (!noteEdits) setBasePlan(value);
    return value;
  }

  function createPlan(event: FormEvent) {
    event.preventDefault();
    if (!compositionError) void run('planning', async signal => { await obtainPlan(signal); });
  }

  async function watch(id: string, signal: AbortSignal) {
    const audio = await awaitJob(id, signal, next => { if (!signal.aborted) setStatus(next); });
    signal.throwIfAborted();
    setStemResult(null);
    setResult(audio);
    setNotice('Render complete. This preview reflects the controls at submission time.');
  }

  function renderInput(score:MusicPlan) {
    return {...scoreInput(),mix,expected_score_sha256:score.score_sha256,automation:{lanes}};
  }

  async function verifiedAutomation(score:MusicPlan,signal:AbortSignal) {
    const input=renderInput(score);
    const receipt=await planMusicAutomation(input,signal);
    signal.throwIfAborted();
    setAutomationReceipt(receipt);
    return {...input,automation:{lanes,expected_automation_sha256:receipt.automation_sha256}};
  }

  function validateMixAutomation() {
    if(compositionError||mixError)return;
    void run('planning',async signal=>{
      const score=currentPlan??await obtainPlan(signal);
      await verifiedAutomation(score,signal);
      setNotice('Mixer automation validated against the current score.');
    });
  }

  function downloadAutomation() {
    if(compositionError||mixError)return;
    void run('exporting',async signal=>{
      const score=currentPlan??await obtainPlan(signal);
      const input=await verifiedAutomation(score,signal);
      const blob=await exportMusicAutomation(input,signal);
      signal.throwIfAborted();
      saveBlob(blob,'sonumio-automation.json');
      setNotice('Verified automation export requested.');
    });
  }

  async function loadAutomation(file:File|null) {
    if(!file||busy||compositionError)return;
    if(file.size<1||file.size>MAX_AUTOMATION_BYTES){setError('Automation file must be 1–128 KiB.');return;}
    void run('planning',async signal=>{
      const data=parseAutomationFile(await file.text(),composition);
      signal.throwIfAborted();
      const score=currentPlan??await obtainPlan(signal);
      // Validate the imported automation against the current score BEFORE
      // replacing any local controls. A failed import leaves the mix intact.
      const receipt=await planMusicAutomation({...scoreInput(),mix:data.mix,
        expected_score_sha256:score.score_sha256,automation:{lanes:data.lanes}},signal);
      signal.throwIfAborted();
      setMix(data.mix);setLanes(data.lanes);setAutomationReceipt(receipt);
      setNotice('Imported and validated mixer automation.');
    });
  }

  function render() {
    if (compositionError || mixError) return;
    void run('rendering', async signal => {
      const score = currentPlan ?? await obtainPlan(signal);
      signal.throwIfAborted();
      const input=lanes.length?await verifiedAutomation(score,signal):{...scoreInput(),mix,expected_score_sha256:score.score_sha256};
      signal.throwIfAborted();
      if (typeof crypto.randomUUID !== 'function') throw new Error('This browser cannot create local job identifiers.');
      const id = crypto.randomUUID();
      rememberJob(id);
      setResult(null);
      setStemResult(null);
      setStatus('queued');
      try {
        const job = await submitMusicJob(input, { id, signal });
        signal.throwIfAborted();
        if (job.kind !== 'music') throw new Error('Unexpected render job kind.');
      } catch (problem) {
        if (isRejectedJobSubmission(problem) && !signal.aborted) rememberJob(null);
        throw problem;
      }
      await watch(id, signal);
    });
  }

  function audition(track: Track) {
    if (compositionError || mixError) return;
    void run('rendering', async signal => {
      const score = currentPlan ?? await obtainPlan(signal);
      const verified = await verifiedAutomation(score,signal);
      const input = {...verified, track, audition:stemMode};
      const planned = await planStem(input,signal);
      signal.throwIfAborted();
      if (typeof crypto.randomUUID !== 'function') throw new Error('This browser cannot create local job identifiers.');
      const id = crypto.randomUUID();
      rememberJob(id);
      setResult(null);
      setStemResult(null);
      setStatus('queued');
      try {
        const job = await submitStemJob(input,{id,signal});
        if (job.kind !== 'music') throw new Error('Unexpected stem job kind.');
      } catch (problem) {
        if (isRejectedJobSubmission(problem) && !signal.aborted) rememberJob(null);
        throw problem;
      }
      const audio = await awaitJob(id,signal,next=>{if(!signal.aborted)setStatus(next);});
      signal.throwIfAborted();
      parseStemResult(audio,planned);
      setStemResult({audio,plan:planned});
      setNotice('Stem audition ready. It reflects the submitted score and settings.');
    });
  }

  function saveStem() {
    if (!stemResult) return;
    const {audio,plan}=stemResult;
    void run('exporting',async signal=>{
      const blob=await downloadStem(audio,plan,signal);
      signal.throwIfAborted();
      saveBlob(blob,'sonumio-'+plan.track+'-'+plan.audition+'.wav');
      setNotice('Verified stem WAV download requested.');
    });
  }

  function recover() {
    if (!lastJob) return;
    void run('recovering', async signal => {
      activeJob.current = lastJob;
      const job = await getJob(lastJob, signal);
      signal.throwIfAborted();
      if (job.kind !== 'music') throw new Error('This identifier does not refer to a music job.');
      activeJob.current = lastJob;
      await watch(lastJob, signal);
    });
  }

  async function cancel() {
    if (cancelBusy.current) return;
    const id = activeJob.current;
    if (!id) {
      pending.current?.abort();
      setNotice('Request cancelled.');
      return;
    }
    const controller = new AbortController();
    cancelBusy.current = true;
    cancelPending.current = controller;
    pending.current?.abort();
    setActivity('cancelling');
    setError('');
    try {
      const job = await cancelJob(id, controller.signal);
      if (controller.signal.aborted || !mounted.current) return;
      if (job.kind !== 'music') throw new Error('Unexpected cancellation response.');
      setStatus(job.status);
      setNotice(job.status === 'cancelling'
        ? 'Cancellation requested. Active synthesis must finish before cleanup.'
        : 'Cancellation confirmed by the local server.');
    } catch (problem) {
      if (!controller.signal.aborted && mounted.current)
        setError(errorMessage(problem) + ' Use Check last render to recover the job.');
    } finally {
      cancelBusy.current = false;
      cancelPending.current = null;
      if (mounted.current && !pending.current) setActivity(null);
    }
  }

  function saveBlob(blob:Blob,filename:string) {
    const url=URL.createObjectURL(blob);
    downloads.current.add(url);
    const anchor=document.createElement('a');
    anchor.href=url;anchor.download=filename;anchor.click();
    window.setTimeout(()=>{if(downloads.current.delete(url))URL.revokeObjectURL(url);},1000);
  }

  function download(format: 'json' | 'midi') {
    if (!currentPlan) return;
    void run('exporting', async signal => {
      const blob = await exportMusic({ ...scoreInput(), expected_score_sha256: currentPlan.score_sha256 }, format, signal);
      signal.throwIfAborted();
      saveBlob(blob,format==='json'?'sonumio-score.json':'sonumio-score.mid');
      setNotice('Verified score export requested.');
    });
  }

  const duration = compositionError ? null : previewFrames(composition) / SAMPLE_RATE;

  return <section className="panel music-studio" aria-labelledby="music-studio-heading">
    <div className="section-heading">
      <div><span className="overline">WORKSPACE / MUSIC</span><h2 id="music-studio-heading">Original music studio</h2></div>
      <span className="engine-pill">PROCEDURAL</span>
    </div>
    <p className="hint">Compose an original instrumental score, adjust the eight-track mix, and render a local stereo preview. No trained model or external recording is used.</p>
    <form onSubmit={createPlan}>
      <fieldset disabled={busy} className="music-controls">
        <legend>Composition</legend>
        <div className="field-grid">
          <label>Style<select value={composition.style} onChange={e => editComposition('style', e.target.value as Composition['style'])}>
            {STYLES.map(style => <option key={style}>{style}</option>)}</select></label>
          <label>Mode<select value={composition.mode} onChange={e => editComposition('mode', e.target.value as Composition['mode'])}>
            {MODES.map(mode => <option key={mode}>{mode}</option>)}</select></label>
          <label>Root note<select value={composition.root} onChange={e => editComposition('root', Number(e.target.value))}>
            {Array.from({ length: 24 }, (_, i) => i + 48).map(note =>
              <option key={note} value={note}>{ROOT_NAMES[note % 12]}{Math.floor(note / 12) - 1} (MIDI {note})</option>)}</select></label>
          <label>Tempo (BPM)<input type="number" min="60" max="160" step="1" required value={Number.isNaN(composition.bpm) ? '' : composition.bpm} onChange={e => editComposition('bpm', numberInput(e.target.value))}/></label>
          <label>Bars<input type="number" min="4" max="16" step="1" required value={Number.isNaN(composition.bars) ? '' : composition.bars} onChange={e => editComposition('bars', numberInput(e.target.value))}/></label>
          <label>Seed<input type="number" min="0" max="4294967295" step="1" required value={Number.isNaN(composition.seed) ? '' : composition.seed} onChange={e => editComposition('seed', numberInput(e.target.value))}/></label>
          <label>Swing: {composition.swing.toFixed(2)}<input type="range" min="0" max="0.6" step="0.01" value={composition.swing} onChange={e => editComposition('swing', Number(e.target.value))}/></label>
          <div className="music-duration"><span className="field-heading">PREVIEW LENGTH</span>{duration === null ? 'Adjust controls' : duration.toFixed(2) + ' seconds'}<p className="hint">Maximum 30 seconds</p></div>
        </div>
        {compositionError && <p className="warning" role="status">{compositionError}</p>}
        <button className="secondary-button" type="submit" disabled={connected !== true || Boolean(compositionError)}>{noteEdits ? 'Validate edited score' : 'Compose score'}</button>
      </fieldset>
    </form>
    {currentPlan && <div className="music-score-summary">
      <h3>Score structure</h3>
      <p>{currentPlan.events} events · {currentPlan.bars} bars · {currentPlan.duration_seconds.toFixed(2)}s</p>
      <div className="music-sections" aria-label="Arrangement sections">
        {currentPlan.sections.map(section => <div key={section.first_bar} style={{ flexGrow: section.bars }}>
          <strong>{section.kind}</strong><small>{section.bars} {section.bars === 1 ? 'bar' : 'bars'}</small>
        </div>)}
      </div>
      <p className="hint">Harmony (scale degrees): {currentPlan.harmony.map(degree => degree + 1).join(' · ')}</p>
      <details><summary>Score SHA-256</summary><code className="music-digest">{currentPlan.score_sha256}</code></details>
      <div className="music-actions">
        <button type="button" className="secondary-button" disabled={busy || connected !== true} onClick={() => download('json')}>Export score JSON</button>
        <button type="button" className="secondary-button" disabled={busy || connected !== true} onClick={() => download('midi')}>Export MIDI</button>
      </div>
    </div>}
    {basePlan?.note_events && <ScoreNoteEditor key={basePlan.score_sha256}
      base={{...basePlan, note_events:basePlan.note_events}} disabled={busy} onChange={changeNoteEdits}/>}
    <fieldset disabled={busy} className="music-controls">
      <legend>Track mixer</legend>
      <label>Instrument audition<select value={stemMode} onChange={e=>setStemMode(e.target.value as StemMode)}>
        <option value="dry">Dry stem (common attenuation, no mixer)</option>
        <option value="solo">Solo preview (current mixer and automation)</option>
      </select></label>
      <p className="hint">Solo auditions temporarily ignore mute and solo selections without changing your mix.</p>
      <div className="music-track-list">{TRACKS.map((track, i) => <div key={track} className="music-track">
        <div><strong>{TRACK_LABELS[track]}</strong><small>{currentPlan ? currentPlan.tracks[track].events + ' events' : 'Track ' + (i + 1)}</small></div>
        <label>Gain {mix.gains[i].toFixed(2)}<input type="range" min="0" max="2" step="0.01" value={mix.gains[i]}
          onChange={e => editMix({ ...mix, gains: mix.gains.map((gain, index) => index === i ? Number(e.target.value) : gain) })}/></label>
        <label className="music-check"><input type="checkbox" checked={mix.mute.includes(track)} onChange={() => toggleTrack(track, 'mute')}/>Mute</label>
        <label className="music-check"><input type="checkbox" checked={mix.solo.includes(track)} onChange={() => toggleTrack(track, 'solo')}/>Solo</label>
        <button type="button" className="secondary-button" disabled={connected!==true||Boolean(compositionError||mixError)} onClick={()=>audition(track)}>Audition</button>
      </div>)}</div>
      <div className="field-grid">
        <label>Room: {mix.room.toFixed(2)}<input type="range" min="0" max="1" step="0.01" value={mix.room} onChange={e => editMix({ ...mix, room: Number(e.target.value) })}/></label>
        <label>Peak ceiling: {mix.peak_ceiling.toFixed(2)}<input type="range" min="0.05" max="0.99" step="0.01" value={mix.peak_ceiling} onChange={e => editMix({ ...mix, peak_ceiling: Number(e.target.value) })}/></label>
      </div>
      <button className="secondary-button" type="button" onClick={() => editMix(defaultMusicMix())}>Reset mix</button>
    </fieldset>
    {!compositionError&&<MusicAutomationEditor key={compositionKey(composition)} composition={composition}
      lanes={lanes} receipt={automationReceipt} disabled={busy} onChange={next=>{setLanes(next);setAutomationReceipt(null);setNotice('');}}/>}
    <div className="music-actions">
      <button type="button" className="secondary-button" disabled={busy||connected!==true||Boolean(compositionError||mixError)}
        onClick={validateMixAutomation}>Validate automation</button>
      <button type="button" className="secondary-button" disabled={busy||connected!==true||Boolean(compositionError||mixError)}
        onClick={downloadAutomation}>Export automation</button>
      <label className="automation-import">Import automation JSON<input type="file" accept=".json,application/json"
        disabled={busy||connected!==true||Boolean(compositionError)} onChange={e=>{
          const file=e.target.files?.[0]??null;e.target.value='';void loadAutomation(file);
        }}/></label>
    </div>
    {mixError && <p className="warning">{mixError}</p>}
    <button type="button" className="primary-button" disabled={busy || connected !== true || Boolean(compositionError || mixError)} onClick={render}>
      {busy ? activity + '…' : 'Render stereo music'} <span>↗</span>
    </button>
    <div className="music-actions">
      {busy && activity !== 'cancelling' && <button type="button" className="secondary-button" onClick={() => {
        if (activity === 'rendering' || activity === 'recovering') void cancel();
        else pending.current?.abort();
      }}>Cancel</button>}
      {!busy && lastJob && <button type="button" className="secondary-button" disabled={connected !== true} onClick={recover}>Check last render</button>}
      {!busy && lastJob && <button type="button" className="secondary-button" onClick={() => { rememberJob(null); setStatus(null); }}>Forget job reference</button>}
    </div>
    {status && <p className="hint" role="status">Last render: {status}</p>}
    {notice && <p className="hint" role="status">{notice}</p>}
    {error && <p className="warning" role="alert">{error}</p>}
    {stemResult && <div className="music-result">
      <h3>{TRACK_LABELS[stemResult.plan.track]} · {stemResult.plan.audition === 'dry'?'dry stem':'solo preview'}</h3>
      <audio controls preload="metadata" src={stemResult.audio.audio_url} onError={()=>setError('Audition expired; render a new copy.')}/>
      <button type="button" className="secondary-button" disabled={busy||connected!==true} onClick={saveStem}>Download verified stem WAV</button>
      <details><summary>Audition score SHA-256</summary><code className="music-digest">{stemResult.plan.score_sha256}</code></details>
      <p className="hint">This audition is separate from the full mix. Its settings are fixed at submission.</p>
    </div>}
    {result && <div className="music-result">
      <h3>Music preview</h3>
      <audio controls preload="metadata" src={result.audio_url} onError={() => setError('Preview unavailable or expired; render a new copy.')}/>
      <a className="download" href={result.audio_url} download="sonumio-music.wav">Download WAV</a>
      <p className="hint">A local preview may expire. Save a copy before closing the studio.</p>
      {result.disclaimer && <p className="hint">{result.disclaimer}</p>}
    </div>}
  </section>;
}
