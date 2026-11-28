import './native-project.css';
import { useEffect,useRef,useState,type FormEvent } from 'react';
import { ScoreNoteEditor } from './ScoreNoteEditor';
import { TRACKS,defaultMusicMix,validateMusicMix,type MusicMix,type Track } from './music-contract';
import type { NoteEdits } from './music-note-editor';
import { awaitJob,cancelJob,getJob,isClientJobId,isRejectedJobSubmission,type JobAudio,type JobStatus } from './api';
import { createNativeProject,inspectNativeProject,inspectProjectScore,inspectProjectAutomation,
  reviseNativeProject,exportNativeProjectScore,submitNativeProjectJob } from './music-project-api';
import { createInput,revisionInput,readDocument,validTitle,MAX_PROJECT_BYTES,MAX_SCORE_BYTES,
  MAX_AUTOMATION_BYTES,type OpenProject,type SourceScore,type SourceAutomation } from './music-project-contract';

type Activity='opening'|'inspecting'|'creating'|'revising'|'exporting'|'rendering'|'recovering'|'cancelling';
const JOB_KEY='sonumio.project.last-render';
type RenderReference={id:string;project:string};
type Rendered={audio:JobAudio;project:string;verified:boolean};

function getReference():RenderReference|null{
  try{
    const raw=sessionStorage.getItem(JOB_KEY);
    if(!raw)return null;
    const value:unknown=JSON.parse(raw);
    if(value&&typeof value==='object'&&'id'in value&&'project'in value&&
      typeof value.id==='string'&&isClientJobId(value.id)&&typeof value.project==='string'&&/^[0-9a-f]{64}$/.test(value.project))
      return {id:value.id,project:value.project};
  }catch{/* session storage can be unavailable */}
  return null;
}
const problemText=(problem:unknown)=>problem instanceof Error?problem.message:'Native project operation failed.';

export function NativeProjectWorkspace({connected}:{connected:boolean|null}){
  const [project,setProject]=useState<OpenProject|null>(null);
  const [title,setTitle]=useState('Original music');
  const [mix,setMix]=useState<MusicMix>(defaultMusicMix);
  const [edits,setEdits]=useState<NoteEdits|null>(null);
  const [score,setScore]=useState<SourceScore|null>(null);
  const [automation,setAutomation]=useState<SourceAutomation|null>(null);
  const [activity,setActivity]=useState<Activity|null>(null);
  const [error,setError]=useState('');
  const [notice,setNotice]=useState('');
  const [lastRender,setLastRender]=useState<RenderReference|null>(getReference);
  const [status,setStatus]=useState<JobStatus|null>(null);
  const [rendered,setRendered]=useState<Rendered|null>(null);
  const [confirmClose,setConfirmClose]=useState(false);
  const pending=useRef<AbortController|null>(null);
  const cancellation=useRef<AbortController|null>(null);
  const active=useRef<RenderReference|null>(null);
  const mounted=useRef(true);
  const urls=useRef(new Set<string>());

  useEffect(()=>{
    mounted.current=true;
    return ()=>{
      mounted.current=false;pending.current?.abort();cancellation.current?.abort();
      for(const url of urls.current)URL.revokeObjectURL(url);
      urls.current.clear();
    };
  },[]);

  const busy=activity!==null;
  const available=connected===true&&!busy;
  let draftProblem:string|null=null;
  let dirty=false;
  if(project){
    try{revisionInput(project,title,edits,mix);dirty=true;}
    catch(problem){
      const message=problemText(problem);
      if(message!=='No project changes to save.'){dirty=true;draftProblem=message;}
    }
  }
  const snapshot=project?.snapshot;

  function remember(reference:RenderReference|null){
    active.current=reference;setLastRender(reference);
    try{
      if(reference)sessionStorage.setItem(JOB_KEY,JSON.stringify(reference));
      else sessionStorage.removeItem(JOB_KEY);
    }catch{/* only the current page can recover */}
  }
  function download(blob:Blob,name:string){
    const url=URL.createObjectURL(blob);
    urls.current.add(url);
    const anchor=document.createElement('a');anchor.href=url;anchor.download=name;anchor.click();
    window.setTimeout(()=>{if(urls.current.delete(url))URL.revokeObjectURL(url);},1000);
  }
  function activate(next:OpenProject){
    setProject(next);setTitle(next.snapshot.title);
    setMix({...next.snapshot.mix,gains:[...next.snapshot.mix.gains],
      mute:[...next.snapshot.mix.mute],solo:[...next.snapshot.mix.solo]});
    setEdits(null);setScore(null);setAutomation(null);setConfirmClose(false);
  }
  async function run(action:Activity,task:(signal:AbortSignal)=>Promise<void>){
    if(pending.current||cancellation.current||connected!==true)return;
    const controller=new AbortController();
    pending.current=controller;active.current=null;
    setActivity(action);setError('');setNotice('');
    try{await task(controller.signal);}
    catch(problem){if(mounted.current&&!controller.signal.aborted)setError(problemText(problem));}
    finally{
      if(pending.current===controller)pending.current=null;
      if(mounted.current&&!cancellation.current)setActivity(null);
    }
  }
  function open(file:File|null){
    if(!file||!available||dirty)return;
    void run('opening',async signal=>{
      const document=await readDocument(file,MAX_PROJECT_BYTES);
      signal.throwIfAborted();
      const next=await inspectNativeProject(document,signal);
      signal.throwIfAborted();activate(next);
      setNotice('Verified native project opened. Original file remains unchanged.');
    });
  }
  function chooseScore(file:File|null){
    if(!file||!available||project)return;
    void run('inspecting',async signal=>{
      const document=await readDocument(file,MAX_SCORE_BYTES);
      signal.throwIfAborted();
      const source=await inspectProjectScore(document,signal);
      signal.throwIfAborted();setScore(source);
      setNotice('Verified source score selected.');
    });
  }
  function chooseAutomation(file:File|null){
    if(!file||!available||project)return;
    void run('inspecting',async signal=>{
      const document=await readDocument(file,MAX_AUTOMATION_BYTES);
      signal.throwIfAborted();
      const source=await inspectProjectAutomation(document,signal);
      signal.throwIfAborted();setAutomation(source);
      setNotice('Verified automation document selected.');
    });
  }
  function create(event:FormEvent){
    event.preventDefault();
    if(!score||project||!available||!validTitle(title))return;
    void run('creating',async signal=>{
      const input=createInput(title,score,automation);
      const next=await createNativeProject(input,signal);
      signal.throwIfAborted();activate(next);
      download(new Blob([next.document],{type:'application/json'}),'music-r0.sonumio');
      setNotice('Project created. Save the downloaded .sonumio file to reopen it later.');
    });
  }
  function close(){
    setProject(null);setTitle('Original music');setMix(defaultMusicMix());setEdits(null);
    setScore(null);setAutomation(null);setConfirmClose(false);setError('');
    setNotice('Project closed. Downloaded files have not been modified.');
  }
  function resetDraft(){
    if(!project)return;
    activate(project);setNotice('Unsaved draft discarded; the inspected project is unchanged.');
  }
  function saveRevision(){
    if(!project||!available||!dirty||draftProblem)return;
    void run('revising',async signal=>{
      const input=revisionInput(project,title,edits,mix);
      const child=await reviseNativeProject(project,input,signal);
      signal.throwIfAborted();
      activate(child);
      download(new Blob([child.document],{type:'application/json'}),'music-r'+child.snapshot.revision+'.sonumio');
      setNotice('Revision '+child.snapshot.revision+' verified and downloaded. The parent file remains unchanged.');
    });
  }
  function exportScore(format:'json'|'midi'){
    if(!project||!available||dirty)return;
    void run('exporting',async signal=>{
      const blob=await exportNativeProjectScore(project,format,signal);
      signal.throwIfAborted();
      download(blob,format==='json'?'project-score.json':'project-score.mid');
      setNotice('Verified project score exported.');
    });
  }
  function toggleTrack(track:Track,field:'mute'|'solo'){
    const opposite=field==='mute'?'solo':'mute';
    setMix(current=>({...current,[field]:current[field].includes(track)?
      current[field].filter(t=>t!==track):[...current[field],track],
      [opposite]:current[opposite].filter(t=>t!==track)}));
  }
  async function watch(reference:RenderReference,signal:AbortSignal,source?:OpenProject){
    const audio=await awaitJob(reference.id,signal,next=>{if(!signal.aborted)setStatus(next);});
    signal.throwIfAborted();
    const receipt=audio as JobAudio&{project_sha256?:string;score_sha256?:string;automation_sha256?:string;frames?:number};
    const verified=receipt.project_sha256===reference.project;
    if(receipt.project_sha256!==undefined&&!verified)throw new Error('Render returned a different project identity.');
    if(source){
      if(!verified||receipt.score_sha256!==source.snapshot.score_sha256||
        receipt.automation_sha256!==source.snapshot.automation_sha256||receipt.frames!==source.snapshot.frames)
        throw new Error('Project render receipt differs from the submitted revision.');
    }
    setRendered({audio,project:reference.project,verified});
    setNotice(verified?'Verified project render is ready.':'Recovered audio reference only; project digest was not retained by the journal.');
  }
  function render(){
    if(!project||!available||dirty||!project.snapshot.preview_supported)return;
    void run('rendering',async signal=>{
      if(typeof crypto.randomUUID!=='function')throw new Error('Browser cannot create local job identifiers.');
      const reference={id:crypto.randomUUID(),project:project.snapshot.project_sha256};
      remember(reference);setRendered(null);setStatus('queued');
      try{
        await submitNativeProjectJob(project,reference.id,signal);
      }catch(problem){
        if(!signal.aborted&&isRejectedJobSubmission(problem))remember(null);
        throw problem;
      }
      signal.throwIfAborted();
      await watch(reference,signal,project);
    });
  }
  function recover(){
    if(!lastRender||!available)return;
    void run('recovering',async signal=>{
      active.current=lastRender;
      const job=await getJob(lastRender.id,signal);
      if(job.kind!=='music')throw new Error('Saved identifier is not a music job.');
      signal.throwIfAborted();
      await watch(lastRender,signal);
    });
  }
  async function cancel(){
    if(cancellation.current)return;
    const ref=active.current;
    pending.current?.abort();
    if(!ref){setNotice('Request cancelled.');return;}
    const controller=new AbortController();cancellation.current=controller;
    setActivity('cancelling');
    try{
      const job=await cancelJob(ref.id,controller.signal);
      if(controller.signal.aborted||!mounted.current)return;
      setStatus(job.status);
      setNotice(job.status==='cancelling'?'Cancellation requested; active synthesis finishes before cleanup.':'Cancellation confirmed.');
    }catch(problem){
      if(mounted.current&&!controller.signal.aborted)setError(problemText(problem)+' Check the last project render to recover.');
    }finally{
      cancellation.current=null;
      if(mounted.current&&!pending.current)setActivity(null);
    }
  }

  return <section className="panel native-project-workspace" aria-labelledby="native-project-heading">
    <div className="section-heading"><div><span className="overline">WORKSPACE / PROJECT FILES</span>
      <h2 id="native-project-heading">Native music projects</h2></div>
      <span className="engine-pill">LOCAL FILE</span></div>
    <p className="hint">Save a self-contained score and mixer document as a .sonumio file. Files are verified locally by the API; no external paths, audio recordings or weights are embedded.</p>
    <fieldset disabled={!available||dirty}>
      <legend>Open a saved project</legend>
      <label>Project file<input type="file" accept=".sonumio,application/json" onChange={e=>{
        const file=e.target.files?.[0]??null;e.target.value='';open(file);
      }}/></label>
    </fieldset>
    {!project&&<form onSubmit={create}><fieldset disabled={!available}>
      <legend>Create from exported documents</legend>
      <label>Title<input required maxLength={80} value={title} onChange={e=>setTitle(e.target.value)}/></label>
      <div className="field-grid">
        <label>Score JSON<input type="file" accept=".json,application/json" onChange={e=>{
          const file=e.target.files?.[0]??null;e.target.value='';chooseScore(file);
        }}/></label>
        <label>Automation JSON (optional)<input type="file" accept=".json,application/json" onChange={e=>{
          const file=e.target.files?.[0]??null;e.target.value='';chooseAutomation(file);
        }}/></label>
      </div>
      {score&&<p className="hint">{score.bars} bars · {score.events} events · verified score <code>{score.score_sha256.slice(0,12)}</code></p>}
      {automation&&<p className="hint">{automation.lanes} lanes · {automation.keys} keys · verified automation <code>{automation.automation_sha256.slice(0,12)}</code></p>}
      <div className="music-actions">
        {automation&&<button type="button" className="secondary-button" onClick={()=>setAutomation(null)}>Use default mix</button>}
        <button type="submit" className="secondary-button" disabled={!score||!validTitle(title)}>Create and download project</button>
      </div>
    </fieldset></form>}
    {project&&snapshot&&<>
      <div className="project-summary">
        <h3>{snapshot.title}</h3><p>Revision {snapshot.revision} · {snapshot.bars} bars · {snapshot.events} events · {snapshot.duration_seconds.toFixed(2)}s</p>
        <p>{snapshot.automation_lane_count} automation lanes · {snapshot.automation_keys} keys</p>
        <details><summary>Verified identities</summary><dl>
          <dt>Project SHA-256</dt><dd><code className="music-digest">{snapshot.project_sha256}</code></dd>
          <dt>Score SHA-256</dt><dd><code className="music-digest">{snapshot.score_sha256}</code></dd>
          <dt>Automation SHA-256</dt><dd><code className="music-digest">{snapshot.automation_sha256}</code></dd>
          {snapshot.parent_project_sha256&&<><dt>Declared parent (not independently verified on open)</dt><dd><code className="music-digest">{snapshot.parent_project_sha256}</code></dd></>}
        </dl></details>
      </div>
      {!snapshot.preview_supported&&<p className="hint">This project exceeds the browser's 30-second / 16-bar editing and rendering limit. Score export and title/mix revisions remain available; render the full project with the local CLI.</p>}
      <fieldset disabled={!available}><legend>Revision title</legend>
        <label>Title<input required maxLength={80} value={title} onChange={e=>setTitle(e.target.value)}/></label>
      </fieldset>
      {snapshot.score_plan?.note_events&&<ScoreNoteEditor
        key={snapshot.project_sha256}
        base={{...snapshot.score_plan,note_events:snapshot.score_plan.note_events}}
        disabled={!available} onChange={setEdits}/>}
      <fieldset disabled={!available} className="music-controls">
        <legend>Static mixer (existing automation retained)</legend>
        <div className="music-track-list">{TRACKS.map((track,i)=><div className="music-track" key={track}>
          <strong>{track}</strong>
          <label>Gain {mix.gains[i].toFixed(2)}<input type="range" min={0} max={2} step=".01" value={mix.gains[i]}
            onChange={e=>{const value=Number(e.target.value);setMix(current=>({...current,gains:current.gains.map((g,j)=>i===j?value:g)}));}}/></label>
          <label className="music-check"><input type="checkbox" checked={mix.mute.includes(track)} onChange={()=>toggleTrack(track,'mute')}/>Mute</label>
          <label className="music-check"><input type="checkbox" checked={mix.solo.includes(track)} onChange={()=>toggleTrack(track,'solo')}/>Solo</label>
        </div>)}</div>
        <div className="field-grid">
          <label>Room {mix.room.toFixed(2)}<input type="range" min={0} max={1} step=".01" value={mix.room}
            onChange={e=>setMix({...mix,room:Number(e.target.value)})}/></label>
          <label>Peak ceiling {mix.peak_ceiling.toFixed(2)}<input type="range" min=".05" max=".99" step=".01" value={mix.peak_ceiling}
            onChange={e=>setMix({...mix,peak_ceiling:Number(e.target.value)})}/></label>
        </div>
      </fieldset>
      {snapshot.automation_lanes.length>0&&<details><summary>Embedded automation lanes</summary>
        <div className="note-table-wrap"><table className="note-table">
          <thead><tr><th>Track</th><th>Parameter</th><th>Keys</th><th>Tick range</th></tr></thead>
          <tbody>{snapshot.automation_lanes.map(lane=><tr key={lane.track+':'+lane.parameter}>
            <td>{lane.track}</td><td>{lane.parameter}</td><td>{lane.keys.length}</td>
            <td>{lane.keys[0].tick}–{lane.keys.at(-1)?.tick}</td></tr>)}</tbody>
        </table></div></details>}
      {dirty&&<p role="status" className="hint">Unsaved draft. Save a revision before rendering or exporting the changed score.</p>}
      {(draftProblem||validateMusicMix(mix))&&<p role="alert" className="warning">{draftProblem||validateMusicMix(mix)}</p>}
      <div className="music-actions">
        <button type="button" className="secondary-button" disabled={!available||!dirty||Boolean(draftProblem)}
          onClick={saveRevision}>Save new revision</button>
        <button type="button" className="secondary-button" disabled={busy||!dirty} onClick={resetDraft}>Discard draft</button>
        <button type="button" className="secondary-button" disabled={busy||dirty}
          onClick={()=>download(new Blob([project.document],{type:'application/json'}),'music-r'+snapshot.revision+'.sonumio')}>Download project file</button>
        <button type="button" className="secondary-button" disabled={!available||dirty} onClick={()=>exportScore('json')}>Export score</button>
        <button type="button" className="secondary-button" disabled={!available||dirty} onClick={()=>exportScore('midi')}>Export MIDI</button>
        <button type="button" className="secondary-button" disabled={busy} onClick={()=>dirty?setConfirmClose(true):close()}>Close project</button>
      </div>
      {confirmClose&&dirty&&<div className="music-actions" role="group" aria-label="Confirm discard and close">
        <span>Discard unsaved changes and close?</span>
        <button type="button" className="secondary-button" disabled={busy} onClick={close}>Discard and close</button>
        <button type="button" className="secondary-button" onClick={()=>setConfirmClose(false)}>Keep editing</button>
      </div>}
      <button type="button" className="primary-button" disabled={!available||dirty||!snapshot.preview_supported} onClick={render}>Render saved project <span>↗</span></button>
    </>}
    <div className="music-actions">
      {busy&&activity!=='cancelling'&&<button type="button" className="secondary-button" onClick={()=>void cancel()}>Cancel request</button>}
      {!busy&&lastRender&&<button type="button" className="secondary-button" disabled={!available} onClick={recover}>Check last project render</button>}
      {!busy&&lastRender&&<button type="button" className="secondary-button" onClick={()=>{remember(null);setStatus(null);}}>Forget render reference</button>}
    </div>
    {activity&&<p role="status" className="hint">{activity}…</p>}
    {status&&<p role="status" className="hint">Last project render: {status}</p>}
    {notice&&<p role="status" className="hint">{notice}</p>}
    {error&&<p role="alert" className="warning">{error}</p>}
    {rendered&&<div className="music-result"><h3>Project audio preview</h3>
      <p className="hint">Project {rendered.project.slice(0,12)}…{rendered.verified?' · verified render identity':' · recovered reference only'}</p>
      <audio controls preload="metadata" src={rendered.audio.audio_url} onError={()=>setError('Project audio unavailable or expired; render again.')}/>
      <a className="download" href={rendered.audio.audio_url} download="sonumio-project.wav">Download WAV</a>
    </div>}
  </section>;
}
