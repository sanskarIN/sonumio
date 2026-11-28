import { useState, type FormEvent } from 'react';
import { TRACKS, previewFrames, type Composition, type Track } from './music-contract';
import { scoreFrame } from './music-note-editor';
import { bounds, changeKey, curvePoints, findLane, removeKey, removeLane, LAWS,
  MAX_KEYS, PARAMETERS, type AutomationLane, type AutomationKey, type Parameter,
  type Law, type Selection, type AutomationReceipt } from './music-automation-contract';

type Props = {
  composition: Composition;
  lanes: AutomationLane[];
  disabled: boolean;
  receipt: AutomationReceipt | null;
  onChange: (lanes: AutomationLane[]) => void;
};
const blank = (parameter:Parameter):AutomationKey => ({tick:0,value:parameter==='gain'?1:0,law:'linear'});
const number = (value:string) => value.trim()===''?NaN:Number(value);

function Curve({lane,composition,selection,onSelect,disabled}:{
  lane:AutomationLane|undefined;composition:Composition;selection:Selection;
  onSelect:(key:AutomationKey)=>void;disabled:boolean;
}) {
  const width=640,height=160,left=40,right=16,top=12,bottom=26;
  const [min,max]=bounds(selection.parameter);
  const total=previewFrames(composition);
  const x=(frame:number)=>left+(width-left-right)*frame/total;
  const y=(value:number)=>top+(height-top-bottom)*(max-value)/(max-min);
  const path=lane?curvePoints(lane,composition).map(([frame,value],i)=>
    (i===0?'M':'L')+x(frame).toFixed(3)+' '+y(value).toFixed(3)).join(' '):
    'M'+x(0)+' '+y(selection.parameter==='gain'?1:0)+' L'+x(total)+' '+y(selection.parameter==='gain'?1:0);
  return <div className="automation-curve"><svg viewBox={'0 0 '+width+' '+height} role="img"
    aria-label={selection.track+' '+selection.parameter+' curve, actual sample time'}>
    <rect x={left} y={top} width={width-left-right} height={height-top-bottom} className="automation-grid-bg"/>
    {[min,(min+max)/2,max].map(value=><g key={value}>
      <line x1={left} x2={width-right} y1={y(value)} y2={y(value)} className="automation-grid-line"/>
      <text x={left-6} y={y(value)+4} textAnchor="end" className="automation-axis-label">{value}</text>
    </g>)}
    {Array.from({length:composition.bars+1},(_,i)=>i).map(bar=><g key={bar}>
      <line x1={x(scoreFrame(composition,bar*32))} x2={x(scoreFrame(composition,bar*32))}
        y1={top} y2={height-bottom} className="automation-grid-line"/>
      <text x={x(scoreFrame(composition,bar*32))} y={height-8} textAnchor="middle"
        className="automation-axis-label">{bar+1}</text>
    </g>)}
    <path d={path} className="automation-line" fill="none"/>
    {lane?.keys.map(key=><circle key={key.tick} cx={x(scoreFrame(composition,key.tick))}
      cy={y(key.value)} r={5} className="automation-point" onClick={()=>{if(!disabled)onSelect(key);}}>
      <title>Tick {key.tick}: {key.value}, {key.law}</title>
    </circle>)}
  </svg></div>;
}

export function MusicAutomationEditor({composition,lanes,disabled,receipt,onChange}:Props) {
  const [selection,setSelection]=useState<Selection>({track:'lead',parameter:'gain'});
  const [draft,setDraft]=useState<AutomationKey|null>(null);
  const [previousTick,setPreviousTick]=useState<number|null>(null);
  const [error,setError]=useState('');
  const [confirmClear,setConfirmClear]=useState(false);
  const lane=findLane(lanes,selection);
  const [min,max]=bounds(selection.parameter);
  const totalKeys=lanes.reduce((count,row)=>count+row.keys.length,0);
  function switchLane(next:Selection) {
    setSelection(next);setDraft(null);setPreviousTick(null);setError('');setConfirmClear(false);
  }
  function choose(key:AutomationKey) {
    setDraft({...key});setPreviousTick(key.tick);setError('');
  }
  function add() {
    const used=new Set(lane?.keys.map(key=>key.tick)??[]);
    const free=Array.from({length:composition.bars*32+1},(_,i)=>i).find(tick=>!used.has(tick));
    if(free===undefined){setError('No free tick remains in this lane.');return;}
    setDraft({...blank(selection.parameter),tick:free});setPreviousTick(null);setError('');
  }
  function update<K extends keyof AutomationKey>(key:K,value:AutomationKey[K]) {
    setDraft(current=>current?{...current,[key]:value}:null);setError('');
  }
  function save(event:FormEvent) {
    event.preventDefault();
    if(disabled||!draft)return;
    try {
      onChange(changeKey(lanes,selection,previousTick,draft,composition));
      setDraft(null);setPreviousTick(null);setError('');
    } catch(problem) {setError(problem instanceof Error?problem.message:'Invalid automation key.');}
  }
  function deleteKey(tick:number) {
    onChange(removeKey(lanes,selection,tick));setDraft(null);setPreviousTick(null);setError('');
  }
  return <section className="music-automation-editor" aria-label="Score mixer automation">
    <div className="section-heading"><div><span className="overline">MIX / AUTOMATION</span><h3>Track automation</h3></div>
      <span className="engine-pill">{lanes.length}/16 lanes · {totalKeys} keys</span></div>
    <p className="hint">Control gain multipliers and stereo balance over the score. Interpolation starts at each key and follows the actual swung sample clock. Before the first key and after the last key, the endpoint value is held.</p>
    <fieldset disabled={disabled} className="note-toolbar">
      <legend>Automation lane</legend>
      <label>Track<select value={selection.track} onChange={e=>switchLane({...selection,track:e.target.value as Track})}>
        {TRACKS.map(track=><option key={track} value={track}>{track}</option>)}
      </select></label>
      <label>Parameter<select value={selection.parameter} onChange={e=>switchLane({...selection,parameter:e.target.value as Parameter})}>
        {PARAMETERS.map(parameter=><option key={parameter} value={parameter}>{parameter==='gain'?'Gain multiplier':'Stereo balance'}</option>)}
      </select></label>
      <div className="music-actions">
        <button className="secondary-button" type="button" onClick={add} disabled={(lane?.keys.length??0)>=MAX_KEYS}>Add key</button>
        <button className="secondary-button" type="button" disabled={!lane}
          onClick={()=>{onChange(removeLane(lanes,selection));setDraft(null);setPreviousTick(null);}}>Remove lane</button>
        <button className="secondary-button" type="button" disabled={!lanes.length}
          onClick={()=>setConfirmClear(!confirmClear)}>Clear all lanes</button>
      </div>
    </fieldset>
    {confirmClear&&<div className="music-actions" role="group" aria-label="Confirm automation reset">
      <span>Remove all {lanes.length} automation lanes?</span>
      <button type="button" disabled={disabled} className="secondary-button" onClick={()=>{onChange([]);setDraft(null);setPreviousTick(null);setConfirmClear(false);}}>Confirm clear</button>
      <button type="button" className="secondary-button" onClick={()=>setConfirmClear(false)}>Keep lanes</button>
    </div>}
    <Curve lane={lane} composition={composition} selection={selection} onSelect={choose} disabled={disabled}/>
    <p className="hint">{selection.parameter==='gain'?'Gain 1 preserves the static track gain; 0 silences it; 2 doubles it before mastering.':
      'Balance −1 attenuates the right channel, +1 attenuates the left, and 0 preserves both.'}</p>
    {lane?<div className="note-table-wrap"><table className="note-table">
      <caption>{selection.track} · {selection.parameter} · {lane.keys.length} keys</caption>
      <thead><tr><th>Tick</th><th>Time (s)</th><th>Value</th><th>Outgoing curve</th><th>Action</th></tr></thead>
      <tbody>{lane.keys.map(key=><tr key={key.tick}>
        <td>{key.tick}</td><td>{(scoreFrame(composition,key.tick)/24000).toFixed(3)}</td>
        <td>{key.value.toFixed(3)}</td><td>{key.law}</td><td>
          <button type="button" className="secondary-button" disabled={disabled} onClick={()=>choose(key)}>Edit</button>
          <button type="button" className="secondary-button" disabled={disabled} onClick={()=>deleteKey(key.tick)}>Remove</button>
        </td></tr>)}</tbody></table></div>:<p className="hint">No keys: the static mix is unchanged.</p>}
    {draft&&<form onSubmit={save} className="note-edit-form"><fieldset disabled={disabled}>
      <legend>{previousTick===null?'New automation key':'Edit automation key'}</legend>
      <div className="field-grid">
        <label>Tick<input type="number" min={0} max={composition.bars*32} step={1} required
          value={Number.isNaN(draft.tick)?'':draft.tick} onChange={e=>update('tick',number(e.target.value))}/></label>
        <label>Value<input type="number" min={min} max={max} step="any" required
          value={Number.isNaN(draft.value)?'':draft.value} onChange={e=>update('value',number(e.target.value))}/></label>
        <label>Interpolation<select value={draft.law} onChange={e=>update('law',e.target.value as Law)}>
          {LAWS.map(law=><option key={law} value={law}>{law}</option>)}
        </select></label>
      </div>
      <div className="music-actions"><button type="submit" className="secondary-button">Save key</button>
        <button type="button" className="secondary-button" onClick={()=>{setDraft(null);setPreviousTick(null);setError('');}}>Cancel</button>
      </div></fieldset></form>}
    {error&&<p role="alert" className="warning">{error}</p>}
    {receipt?<details><summary>Verified automation SHA-256</summary><code className="music-digest">{receipt.automation_sha256}</code></details>:
      <p className="hint">Validate automation to obtain a server-verified digest before export.</p>}
  </section>;
}
