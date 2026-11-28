import './render-recipe.css';
import { useEffect,useRef,useState,type FormEvent } from 'react';
import { awaitJob,cancelJob,getJob,isClientJobId,isRejectedJobSubmission,type JobAudio,type JobStatus } from './api';
import { readDocument,MAX_PROJECT_BYTES,declaredProjectId } from './music-project-contract';
import { isDigest } from './music-contract';
import { createRecipeInput,parseRecipeResult,MAX_RECIPE_BYTES,MAX_PERFORMANCE_BYTES,MAX_RACK_BYTES,
  type OpenRecipe,type RecipeReference } from './music-render-recipe-contract';
import { inspectRenderRecipe,createRenderRecipe,submitRecipeJob,downloadRecipeWav,canonicalRecipeBlob } from './music-render-recipe-api';

const STORAGE='sonumio.recipe.last-render';
const engineLabels={project:'Project mix',performance:'Expressive performance',
  rack:'Effects rack','performance-rack':'Performance + effects rack'};
type SourceKind='project'|'performance'|'rack';
type Source={document:string;name:string};
type Result={audio:JobAudio;recipe:OpenRecipe};
type Work='opening'|'reading'|'creating'|'saving'|'rendering'|'recovering'|'downloading'|'cancelling';
const limits:Record<SourceKind,number>={project:MAX_PROJECT_BYTES,performance:MAX_PERFORMANCE_BYTES,rack:MAX_RACK_BYTES};
function storedReference():RecipeReference|null{
  try{
    const value:unknown=JSON.parse(sessionStorage.getItem(STORAGE)??'null');
    if(value&&typeof value==='object'&&'id'in value&&'recipe'in value&&
      typeof value.id==='string'&&isClientJobId(value.id)&&isDigest(value.recipe))
      return {id:value.id,recipe:value.recipe};
  }catch{/* storage optional */}
  return null;
}

export function RenderRecipeWorkspace({connected}:{connected:boolean|null}){
  const [recipe,setRecipe]=useState<OpenRecipe|null>(null);
  const [sources,setSources]=useState<Partial<Record<SourceKind,Source>>>({});
  const [work,setWork]=useState<Work|null>(null);
  const [error,setError]=useState('');
  const [notice,setNotice]=useState('');
  const [reference,setReference]=useState<RecipeReference|null>(storedReference);
  const [status,setStatus]=useState<JobStatus|null>(null);
  const [result,setResult]=useState<Result|null>(null);
  const operation=useRef<AbortController|null>(null);
  const cancellation=useRef<AbortController|null>(null);
  const job=useRef<string|null>(null);
  const mounted=useRef(true);
  const downloads=useRef(new Set<string>());
  useEffect(()=>{
    mounted.current=true;
    return ()=>{
      mounted.current=false;operation.current?.abort();cancellation.current?.abort();
      for(const url of downloads.current)URL.revokeObjectURL(url);
      downloads.current.clear();
    };
  },[]);
  const available=connected===true&&work===null;
  const snapshot=recipe?.snapshot;
  const busy=work!==null;

  function remember(value:RecipeReference|null){
    setReference(value);
    try{
      if(value)sessionStorage.setItem(STORAGE,JSON.stringify(value));
      else sessionStorage.removeItem(STORAGE);
    }catch{/* optional storage */}
  }
  function save(blob:Blob,name:string){
    const url=URL.createObjectURL(blob);
    downloads.current.add(url);
    const a=document.createElement('a');
    a.href=url;a.download=name;a.click();
    window.setTimeout(()=>{if(downloads.current.delete(url))URL.revokeObjectURL(url);},1000);
  }
  async function run(action:Work,task:(signal:AbortSignal)=>Promise<void>){
    if(operation.current||cancellation.current||connected!==true)return;
    const controller=new AbortController();
    operation.current=controller;job.current=null;
    setWork(action);setError('');setNotice('');
    try{await task(controller.signal);}
    catch(problem){
      if(mounted.current&&!controller.signal.aborted)
        setError(problem instanceof Error?problem.message:'Render recipe operation failed.');
    }finally{
      if(operation.current===controller)operation.current=null;
      if(mounted.current&&!cancellation.current)setWork(null);
    }
  }
  function activate(next:OpenRecipe){
    setRecipe(next);setSources({});setResult(null);setStatus(null);
  }
  function open(file:File|null){
    if(!file||!available)return;
    void run('opening',async signal=>{
      const document=await readDocument(file,MAX_RECIPE_BYTES);
      signal.throwIfAborted();
      const next=await inspectRenderRecipe(document,signal);
      signal.throwIfAborted();activate(next);
      setNotice('Recipe verified. Its component documents remain unchanged.');
    });
  }
  function selectSource(kind:SourceKind,file:File|null){
    if(!file||!available||recipe)return;
    void run('reading',async signal=>{
      const document=await readDocument(file,limits[kind]);
      signal.throwIfAborted();
      if(kind==='project')declaredProjectId(document);
      else{
        const raw:unknown=JSON.parse(document);
        const format=kind==='performance'?'sonumio-music-performance-v1':'sonumio-music-processing-rack-v1';
        if(!raw||typeof raw!=='object'||!('format'in raw)||raw.format!==format)
          throw new Error('Selected document has the wrong format.');
      }
      setSources(previous=>({...previous,[kind]:{name:file.name,document}}));
      setNotice('Selected local '+kind+' document. Full validation happens during recipe creation.');
    });
  }
  function create(event:FormEvent){
    event.preventDefault();
    if(!sources.project||!available||recipe)return;
    void run('creating',async signal=>{
      const input=createRecipeInput(sources.project!.document,sources.performance?.document,sources.rack?.document);
      const next=await createRenderRecipe(input,signal);
      const blob=await canonicalRecipeBlob(next);
      signal.throwIfAborted();
      activate(next);save(blob,'sonumio-render-recipe.json');
      setNotice('Verified self-contained recipe created and downloaded.');
    });
  }
  function downloadRecipe(){
    if(!recipe||!available)return;
    void run('saving',async signal=>{
      const blob=await canonicalRecipeBlob(recipe);
      signal.throwIfAborted();save(blob,'sonumio-render-recipe.json');
      setNotice('Canonical render recipe downloaded.');
    });
  }
  async function watch(source:OpenRecipe,id:string,signal:AbortSignal){
    const audio=await awaitJob(id,signal,next=>{if(!signal.aborted)setStatus(next);});
    signal.throwIfAborted();
    parseRecipeResult(audio,source.snapshot);
    setResult({audio,recipe:source});setNotice('Render complete. Recipe identity and geometry verified.');
  }
  function render(){
    if(!recipe||!snapshot?.preview_supported||!available)return;
    const source=recipe;
    void run('rendering',async signal=>{
      if(typeof crypto.randomUUID!=='function')throw new Error('Browser cannot create local render identifiers.');
      const id=crypto.randomUUID();
      job.current=id;remember({id,recipe:source.snapshot.recipe_sha256});
      setResult(null);setStatus('queued');
      try{await submitRecipeJob(source,id,signal);}
      catch(problem){
        if(!signal.aborted&&isRejectedJobSubmission(problem))remember(null);
        throw problem;
      }
      signal.throwIfAborted();await watch(source,id,signal);
    });
  }
  function recover(){
    if(!recipe||!reference||!available||reference.recipe!==recipe.snapshot.recipe_sha256)return;
    const source=recipe,id=reference.id;
    void run('recovering',async signal=>{
      job.current=id;
      const pending=await getJob(id,signal);
      if(pending.kind!=='music')throw new Error('Saved job reference is not a music render.');
      signal.throwIfAborted();
      await watch(source,id,signal);
    });
  }
  function downloadAudio(){
    if(!result||!available)return;
    const source=result;
    void run('downloading',async signal=>{
      const blob=await downloadRecipeWav(source.audio,source.recipe,signal);
      signal.throwIfAborted();save(blob,'sonumio-recipe.wav');
      setNotice('WAV geometry and SHA-256 verified before download.');
    });
  }
  async function cancel(){
    if(cancellation.current)return;
    const id=job.current;
    operation.current?.abort();
    if(!id){setNotice('Request cancelled.');return;}
    const controller=new AbortController();
    cancellation.current=controller;setWork('cancelling');
    try{
      const pending=await cancelJob(id,controller.signal);
      if(!mounted.current||controller.signal.aborted)return;
      setStatus(pending.status);
      setNotice('Cancellation status: '+pending.status+'.');
    }catch(problem){
      if(mounted.current&&!controller.signal.aborted)
        setError((problem instanceof Error?problem.message:'Cancellation failed.')+' Check the saved render reference.');
    }finally{
      cancellation.current=null;
      if(mounted.current&&!operation.current)setWork(null);
    }
  }
  function fileControl(kind:SourceKind){
    return <div key={kind} className="recipe-source">
      <label>{kind==='project'?'Native project (.sonumio)':kind==='performance'?'Performance JSON (optional)':'Processing rack JSON (optional)'}
        <input type="file" accept={kind==='project'?'.sonumio,application/json':'.json,application/json'}
          onChange={e=>{const file=e.target.files?.[0]??null;e.target.value='';selectSource(kind,file);}}/>
      </label>
      {sources[kind]&&<div className="music-actions">
        <span className="hint">{sources[kind]?.name}</span>
        <button type="button" className="secondary-button" onClick={()=>setSources(previous=>{
          const next={...previous};delete next[kind];return next;
        })}>Remove</button>
      </div>}
    </div>;
  }

  return <section className="panel render-recipe-workspace" aria-labelledby="recipe-heading">
    <div className="section-heading"><div><span className="overline">WORKSPACE / RENDER PIPELINE</span>
      <h2 id="recipe-heading">Integrated render recipes</h2></div>
      <span className="engine-pill">LOCAL RENDER</span></div>
    <p className="hint">Combine a saved project with optional expressive timing and effects. The rack must use the same automation as the project. The source score is never overwritten.</p>
    <fieldset disabled={!available}><legend>Open a saved render recipe</legend>
      <label>Recipe JSON<input type="file" accept=".json,application/json" onChange={e=>{
        const file=e.target.files?.[0]??null;e.target.value='';open(file);
      }}/></label>
    </fieldset>
    {!recipe&&<form onSubmit={create}><fieldset disabled={!available}>
      <legend>Build a self-contained recipe</legend>
      {(['project','performance','rack'] as const).map(fileControl)}
      <button type="submit" className="secondary-button" disabled={!sources.project}>Create and download recipe</button>
    </fieldset></form>}
    {snapshot&&recipe&&<>
      <div className="project-summary">
        <h3>{snapshot.title}</h3>
        <p>{engineLabels[snapshot.engine]} · revision {snapshot.revision} · {snapshot.bars} bars · {(snapshot.frames/24000).toFixed(2)}s</p>
        <p className="hint">Procedural synthesis, stereo PCM16 · no external recordings or trained model.</p>
        <details><summary>Verified component digests</summary><dl>
          {([['Recipe',snapshot.recipe_sha256],['Project',snapshot.project_sha256],['Score',snapshot.score_sha256],
            ['Automation',snapshot.automation_sha256],['Performance',snapshot.performance_sha256],['Rack',snapshot.rack_sha256]] as const)
            .map(([label,value])=><div key={label}><dt>{label}</dt><dd><code className="music-digest">{value??'Not included'}</code></dd></div>)}
        </dl></details>
      </div>
      {!snapshot.preview_supported&&<p className="hint">This recipe exceeds the 30-second / 16-bar browser preview limit. Use the offline recipe CLI for full rendering and verified archives.</p>}
      <div className="music-actions">
        <button type="button" className="secondary-button" disabled={!available} onClick={downloadRecipe}>Download recipe</button>
        <button type="button" className="secondary-button" disabled={busy} onClick={()=>{setRecipe(null);setSources({});setResult(null);setNotice('Recipe closed. Saved files remain unchanged.');}}>Close recipe</button>
        <button type="button" className="primary-button" disabled={!available||!snapshot.preview_supported} onClick={render}>Render recipe</button>
      </div>
    </>}
    <div className="music-actions">
      {busy&&work!=='cancelling'&&<button type="button" className="secondary-button" onClick={()=>void cancel()}>Cancel request</button>}
      {!busy&&reference&&<button type="button" className="secondary-button"
        disabled={!available||snapshot?.recipe_sha256!==reference.recipe} onClick={recover}>Check saved render</button>}
      {!busy&&reference&&<button type="button" className="secondary-button" onClick={()=>{remember(null);setStatus(null);}}>Forget render reference</button>}
    </div>
    {reference&&!busy&&snapshot?.recipe_sha256!==reference.recipe&&<p className="hint">To verify the saved render, reopen its original recipe ({reference.recipe.slice(0,12)}…).</p>}
    {work&&<p role="status" className="hint">{work}…</p>}
    {status&&<p role="status" className="hint">Render status: {status}</p>}
    {notice&&<p role="status" className="hint">{notice}</p>}
    {error&&<p role="alert" className="warning">{error}</p>}
    {result&&<div className="music-result">
      <h3>Integrated render</h3>
      <audio controls preload="metadata" src={result.audio.audio_url} onError={()=>setError('Render audio expired or unavailable.')}/>
      <button type="button" className="secondary-button" disabled={!available} onClick={downloadAudio}>Download verified WAV</button>
      <p className="hint">Recipe {result.recipe.snapshot.recipe_sha256.slice(0,12)}… · full replay receipts are available through the offline bundle CLI.</p>
    </div>}
  </section>;
}
