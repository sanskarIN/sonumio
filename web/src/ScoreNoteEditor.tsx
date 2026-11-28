import { useMemo, useState, type FormEvent } from 'react';
import { TRACKS, type MusicPlan, type Track } from './music-contract';
import {
  addNote, changeNote, commitEdit, editCount, makeHistory, redo, removeNote,
  rowsFor, undo, validateDraft, type EditorHistory, type EditorRow, type NoteEdits,
  type NoteEvent, type NoteSelection, MAX_NOTE_EDITS,
} from './music-note-editor';

const NAMES=['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B'];
const LABELS:Record<Track,string>={keys:'Keys',pad:'Pad',bass:'Bass',lead:'Lead',
  pluck:'Pluck',kick:'Kick',snare:'Snare',hat:'Hi-hat'};
const pitchName=(pitch:number)=>NAMES[pitch%12]+(Math.floor(pitch/12)-1);
const selectionKey=(selection:NoteSelection)=>selection.kind+':'+selection.index;
const numberValue=(raw:string)=>raw.trim()===''?NaN:Number(raw);

type Props={base:MusicPlan & {note_events:NoteEvent[]};disabled:boolean;onChange:(edits:NoteEdits|null)=>void};

function NoteRoll({rows,bar,track,selected,onSelect}:{rows:EditorRow[];bar:number;track:Track;
  selected:NoteSelection|null;onSelect:(row:EditorRow)=>void}) {
  const start=bar*32,end=start+32;
  const visible=rows.filter(r=>r.note.track===track&&r.note.tick<end&&r.note.tick+r.note.length>start);
  const min=Math.max(24,Math.min(visible.length?Math.min(...visible.map(r=>r.note.pitch))-2:48,84));
  const max=Math.min(96,Math.max(visible.length?Math.max(...visible.map(r=>r.note.pitch))+2:72,min+12));
  const width=640,rowHeight=17,margin=44,height=(max-min+1)*rowHeight;
  const x=(tick:number)=>margin+(tick-start)*(width-margin)/32;
  return <div className="note-roll" aria-label={'Piano roll, bar '+(bar+1)+', '+LABELS[track]}>
    <svg viewBox={'0 0 '+width+' '+height} role="img" aria-label="Symbolic pitch and timing grid">
      {Array.from({length:max-min+1},(_,i)=>max-i).map(pitch=>{
        const y=(max-pitch)*rowHeight;
        return <g key={pitch}>
          <rect x={margin} y={y} width={width-margin} height={rowHeight}
            className={([1,3,6,8,10].includes(pitch%12)?'roll-sharp':'roll-natural')}/>
          <text x={margin-4} y={y+12} textAnchor="end" className="roll-label">{pitchName(pitch)}</text>
        </g>;
      })}
      {Array.from({length:33},(_,i)=>i).map(tick=>
        <line key={tick} x1={x(start+tick)} x2={x(start+tick)} y1={0} y2={height}
          className={tick%8===0?'roll-beat':'roll-tick'}/>)}
      {visible.map(row=>{
        const n=row.note,key=selectionKey(row.selection);
        const left=x(Math.max(start,n.tick)),right=x(Math.min(end,n.tick+n.length));
        return <rect key={key} x={left+1} y={(max-n.pitch)*rowHeight+2}
          width={Math.max(1,right-left-2)} height={rowHeight-4} rx={2}
          className={'roll-note '+(row.changed?'roll-changed ':'')+(selected&&selectionKey(selected)===key?'roll-selected':'')}
          onClick={()=>onSelect(row)}>
          <title>{LABELS[n.track]} {pitchName(n.pitch)} · tick {n.tick} · length {n.length}</title>
        </rect>;
      })}
    </svg>
  </div>;
}

export function ScoreNoteEditor({base,disabled,onChange}:Props) {
  const [history,setHistory]=useState(()=>makeHistory(base.score_sha256));
  const [track,setTrack]=useState<Track>('lead');
  const [bar,setBar]=useState(0);
  const [selected,setSelected]=useState<NoteSelection|null>(null);
  const [draft,setDraft]=useState<NoteEvent|null>(null);
  const [error,setError]=useState('');
  const [notice,setNotice]=useState('');
  const notes=base.note_events;
  const rows=useMemo(()=>rowsFor(notes,history.present),[notes,history.present]);
  const visible=rows.filter(r=>r.note.track===track&&r.note.tick<(bar+1)*32&&r.note.tick+r.note.length>bar*32);
  const count=editCount(history.present);

  function publish(next:EditorHistory) {
    setHistory(next);
    onChange(editCount(next.present)?next.present:null);
    setSelected(null);
    setDraft(null);
    setError('');
    setNotice('');
  }
  function tryCommit(next:NoteEdits,success:string) {
    const problem=validateDraft(notes,next,base.composition);
    if(problem){setError(problem);return;}
    publish(commitEdit(history,next));
    setNotice(success);
  }
  function choose(row:EditorRow) {
    if(disabled)return;
    setSelected(row.selection);
    setDraft({...row.note});
    setError('');
    setNotice('');
  }
  function beginAdd() {
    setSelected(null);
    setDraft({track,pitch:track==='kick'?36:track==='snare'?38:track==='hat'?42:base.composition.root,
      tick:bar*32,length:1,velocity:.5,pan:0});
    setError('');
    setNotice('Adding a new note. Choose a free interval before saving.');
  }
  function change<K extends keyof NoteEvent>(key:K,value:NoteEvent[K]) {
    setDraft(previous=>previous?{...previous,[key]:value}:null);
    setError('');
  }
  function save(event:FormEvent) {
    event.preventDefault();
    if(!draft||disabled)return;
    try {
      const next=selected?changeNote(notes,history.present,selected,draft):addNote(history.present,draft);
      tryCommit(next,selected?'Note updated. Validate the edited score before export.':'Note added. Validate the edited score before export.');
    } catch(problem) {setError(problem instanceof Error?problem.message:'Could not edit note.');}
  }
  function remove() {
    if(!selected||disabled)return;
    try {tryCommit(removeNote(history.present,selected),'Note removed.');}
    catch(problem){setError(problem instanceof Error?problem.message:'Could not remove note.');}
  }
  function switchTrack(value:Track) {
    setTrack(value);setSelected(null);setDraft(null);setError('');
  }
  function switchBar(value:number) {
    setBar(value);setSelected(null);setDraft(null);setError('');
  }

  return <section className="score-note-editor" aria-label="Score note editor">
    <div className="section-heading"><div><span className="overline">EDIT / NOTES</span><h3>Note editor</h3></div>
      <span className="engine-pill">{count}/{MAX_NOTE_EDITS} edits</span></div>
    <p className="hint">Edit the original symbolic score. Timing uses eight ticks per beat and 32 per bar. Changes are validated against the scale, duration and instrument polyphony.</p>
    <fieldset disabled={disabled} className="note-toolbar">
      <legend>View and history</legend>
      <label>Track<select value={track} onChange={e=>switchTrack(e.target.value as Track)}>
        {TRACKS.map(name=><option key={name} value={name}>{LABELS[name]}</option>)}
      </select></label>
      <label>Bar<select value={bar} onChange={e=>switchBar(Number(e.target.value))}>
        {Array.from({length:base.bars},(_,i)=><option key={i} value={i}>{i+1}</option>)}
      </select></label>
      <div className="music-actions">
        <button type="button" className="secondary-button" disabled={!history.past.length} onClick={()=>publish(undo(history))}>Undo</button>
        <button type="button" className="secondary-button" disabled={!history.future.length} onClick={()=>publish(redo(history))}>Redo</button>
        <button type="button" className="secondary-button" disabled={!count} onClick={()=>publish(commitEdit(history,makeHistory(base.score_sha256).present))}>Reset edits</button>
        <button type="button" className="secondary-button" disabled={count>=MAX_NOTE_EDITS} onClick={beginAdd}>Add note</button>
      </div>
    </fieldset>
    <NoteRoll rows={rows} bar={bar} track={track} selected={selected} onSelect={choose}/>
    <div className="note-table-wrap">
      <table className="note-table">
        <caption>{LABELS[track]}, bar {bar+1}: {visible.length} visible notes</caption>
        <thead><tr><th>Pitch</th><th>Start tick</th><th>Length</th><th>Velocity</th><th>Pan</th><th>Action</th></tr></thead>
        <tbody>{visible.map(row=><tr key={selectionKey(row.selection)} className={row.changed?'note-edited':''}>
          <td>{pitchName(row.note.pitch)} ({row.note.pitch})</td><td>{row.note.tick}</td><td>{row.note.length}</td>
          <td>{row.note.velocity.toFixed(3)}</td><td>{row.note.pan.toFixed(2)}</td>
          <td><button type="button" className="secondary-button" disabled={disabled} onClick={()=>choose(row)}
            aria-pressed={selected!==null&&selectionKey(selected)===selectionKey(row.selection)}>Edit</button></td>
        </tr>)}</tbody>
      </table>
      {!visible.length&&<p className="hint">No notes in this interval. Add a note or select another track/bar.</p>}
    </div>
    {draft&&<form onSubmit={save} className="note-edit-form">
      <fieldset disabled={disabled}>
        <legend>{selected?'Edit selected note':'New note'}</legend>
        <div className="field-grid">
          <label>Track<select value={draft.track} onChange={e=>change('track',e.target.value as Track)}>
            {TRACKS.map(name=><option key={name} value={name}>{LABELS[name]}</option>)}
          </select></label>
          <label>Pitch (MIDI)<input type="number" required min={24} max={96} step={1} value={Number.isNaN(draft.pitch)?'':draft.pitch}
            onChange={e=>change('pitch',numberValue(e.target.value))}/></label>
          <label>Start tick<input type="number" required min={0} max={base.bars*32-1} step={1} value={Number.isNaN(draft.tick)?'':draft.tick}
            onChange={e=>change('tick',numberValue(e.target.value))}/></label>
          <label>Length (ticks)<input type="number" required min={1} max={base.bars*32} step={1} value={Number.isNaN(draft.length)?'':draft.length}
            onChange={e=>change('length',numberValue(e.target.value))}/></label>
          <label>Velocity<input type="number" required min={.001} max={1} step="any" value={Number.isNaN(draft.velocity)?'':draft.velocity}
            onChange={e=>change('velocity',numberValue(e.target.value))}/></label>
          <label>Pan<input type="number" required min={-1} max={1} step="any" value={Number.isNaN(draft.pan)?'':draft.pan}
            onChange={e=>change('pan',numberValue(e.target.value))}/></label>
        </div>
        <div className="music-actions">
          <button type="submit" className="secondary-button">Save note</button>
          {selected&&<button type="button" className="secondary-button" onClick={remove}>Remove note</button>}
          <button type="button" className="secondary-button" onClick={()=>{setDraft(null);setSelected(null);setError('');}}>Cancel edit</button>
        </div>
      </fieldset>
    </form>}
    {error&&<p role="alert" className="warning">{error}</p>}
    {notice&&<p role="status" className="hint">{notice}</p>}
    {count>0&&<p className="hint">Draft: {rows.length} notes. Render or validate the edited score to obtain its server-verified digest.</p>}
  </section>;
}
