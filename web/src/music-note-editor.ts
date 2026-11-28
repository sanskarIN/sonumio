/** Pure score note editor logic, independent of network and React state. */
import { TRACKS, isDigest, type Composition, type Track } from './music-contract';

export const MAX_NOTE_EDITS = 128;
export type NoteEvent = {
  track: Track; pitch: number; tick: number; length: number; velocity: number; pan: number;
};
export type NoteReplacement = { index: number; note: NoteEvent };
export type NoteEdits = {
  base_score_sha256: string;
  remove: number[];
  replace: NoteReplacement[];
  add: NoteEvent[];
};
export type NoteSelection = { kind: 'base' | 'added'; index: number };
export type EditorRow = { selection: NoteSelection; note: NoteEvent; changed: boolean };
export type EditorHistory = { past: NoteEdits[]; present: NoteEdits; future: NoteEdits[] };

const integer = (n: unknown, lo: number, hi: number): n is number =>
  typeof n === 'number' && Number.isInteger(n) && n >= lo && n <= hi;
const finite = (n: unknown, lo: number, hi: number): n is number =>
  typeof n === 'number' && Number.isFinite(n) && n >= lo && n <= hi;
const object = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);
const sameFields = (v: Record<string, unknown>, keys: string[]) =>
  Object.keys(v).length === keys.length && keys.every(k => Object.hasOwn(v,k));
const NOTE_KEYS = ['track','pitch','tick','length','velocity','pan'];
const MAJOR = [0,2,4,5,7,9,11];
const MINOR = [0,2,3,5,7,8,10];
const HISTORY_LIMIT = 64;

export function scoreFrame(c: Composition, tick: number): number {
  const beat = Math.floor(tick/8), sub = tick%8;
  const rawSwing=c.swing*1000;
  const floor=Math.floor(rawSwing);
  const swing=rawSwing-floor===.5?(floor%2===0?floor:floor+1):Math.round(rawSwing);
  const scaled = sub <= 4 ? sub*(1000+swing) : 4*(1000+swing)+(sub-4)*(1000-swing);
  const numerator = (beat*8000+scaled)*60*24000;
  const denominator = c.bpm*8000;
  return Math.floor((2*numerator+denominator)/(2*denominator));
}

export function parseNote(value: unknown, c: Composition): NoteEvent {
  if (!object(value) || !sameFields(value,NOTE_KEYS) || !TRACKS.includes(value.track as Track) ||
      !integer(value.pitch,24,96) || !integer(value.tick,0,c.bars*32-1) ||
      !integer(value.length,1,c.bars*32) || !finite(value.velocity,.001,1) ||
      !finite(value.pan,-1,1))
    throw new Error('Invalid score note fields.');
  const note = value as NoteEvent;
  if (note.tick+note.length > c.bars*32 ||
      scoreFrame(c,note.tick+note.length)-scoreFrame(c,note.tick)>96000)
    throw new Error('Note exceeds the score or four-second duration limit.');
  if (TRACKS.indexOf(note.track)<5) {
    const degree = ((note.pitch-c.root)%12+12)%12;
    if (!(c.mode==='major'?MAJOR:MINOR).includes(degree)) throw new Error('Pitched note is outside the selected scale.');
  }
  return note;
}

export function noteKey(n: NoteEvent): string {
  return JSON.stringify([n.track,n.pitch,n.tick,n.length,n.velocity,n.pan]);
}
export function compareNotes(a: NoteEvent,b: NoteEvent): number {
  return a.tick-b.tick || TRACKS.indexOf(a.track)-TRACKS.indexOf(b.track) ||
    a.pitch-b.pitch || a.length-b.length || a.velocity-b.velocity || a.pan-b.pan;
}
export function validateNotes(notes: NoteEvent[],c: Composition): void {
  if (!Array.isArray(notes) || notes.length<1 || notes.length>4096)
    throw new Error('A score needs 1–4096 notes.');
  const seen=new Set<string>();
  for (const n of notes) {
    parseNote(n,c);
    const key=noteKey(n);
    if (seen.has(key)) throw new Error('Duplicate score note.');
    seen.add(key);
  }
  for (const track of TRACKS) {
    const boundaries: [number,number,number][]=[];
    for (const n of notes) if(n.track===track) boundaries.push([n.tick,1,n.pitch],[n.tick+n.length,-1,n.pitch]);
    boundaries.sort((a,b)=>a[0]-b[0]||a[1]-b[1]||a[2]-b[2]);
    const pitches=new Map<number,number>();
    let count=0;
    for(const [,delta,pitch] of boundaries) {
      count+=delta;
      const active=(pitches.get(pitch)??0)+delta;
      pitches.set(pitch,active);
      if(active>1 || count>(track==='keys'||track==='pad'?4:1))
        throw new Error('Overlapping notes exceed '+track+' polyphony.');
    }
  }
}

export function parseNoteEvents(value:unknown,c:Composition,expected:number):NoteEvent[] {
  if(!Array.isArray(value)||value.length!==expected) throw new Error('Score note count differs from the plan.');
  const notes=value.map(row=>parseNote(row,c));
  validateNotes(notes,c);
  for(let i=1;i<notes.length;i++) if(compareNotes(notes[i-1],notes[i])>0) throw new Error('Score notes are not canonical.');
  return notes;
}

export function emptyEdits(digest:string):NoteEdits {
  if(!isDigest(digest)) throw new Error('Invalid base score digest.');
  return {base_score_sha256:digest,remove:[],replace:[],add:[]};
}
export function editCount(e:NoteEdits):number {return e.remove.length+e.replace.length+e.add.length;}
export function copyEdits(e:NoteEdits):NoteEdits {
  return {...e,remove:[...e.remove],replace:e.replace.map(r=>({...r,note:{...r.note}})),add:e.add.map(n=>({...n}))};
}
export function validateEditShape(e:unknown,c:Composition):asserts e is NoteEdits {
  if(!object(e)||!sameFields(e,['base_score_sha256','remove','replace','add'])||
      !isDigest(e.base_score_sha256)||!Array.isArray(e.remove)||!Array.isArray(e.replace)||!Array.isArray(e.add))
    throw new Error('Invalid note edit transaction.');
  const count=e.remove.length+e.replace.length+e.add.length;
  if(count<1||count>MAX_NOTE_EDITS) throw new Error('Choose 1–128 note edits.');
  const removed=e.remove;
  for(let i=0;i<removed.length;i++) if(!integer(removed[i],0,4095)||(i&&removed[i]<=removed[i-1])) throw new Error('Remove indices must be unique and sorted.');
  let previous=-1;
  for(const row of e.replace) {
    if(!object(row)||!sameFields(row,['index','note'])||!integer(row.index,0,4095)||row.index<=previous||removed.includes(row.index))
      throw new Error('Replacement indices must be unique, sorted and not removed.');
    parseNote(row.note,c);
    previous=row.index;
  }
  e.add.forEach(row=>parseNote(row,c));
}
export function rowsFor(base:NoteEvent[],e:NoteEdits):EditorRow[] {
  const removed=new Set(e.remove);
  const replacements=new Map(e.replace.map(r=>[r.index,r.note]));
  if([...removed,...replacements.keys()].some(i=>i>=base.length)) throw new Error('Edit references an unavailable base note.');
  const rows:EditorRow[]=[];
  base.forEach((note,index)=>{
    if(!removed.has(index)) rows.push({selection:{kind:'base',index},note:replacements.get(index)??note,changed:replacements.has(index)});
  });
  e.add.forEach((note,index)=>rows.push({selection:{kind:'added',index},note,changed:true}));
  return rows.sort((a,b)=>compareNotes(a.note,b.note));
}
export function validateDraft(base:NoteEvent[],e:NoteEdits,c:Composition):string|null {
  try {
    if(editCount(e)>0) validateEditShape(e,c);
    validateNotes(rowsFor(base,e).map(r=>r.note),c);
    return null;
  } catch(problem) {return problem instanceof Error?problem.message:'Invalid note edits.';}
}
export function changeNote(base:NoteEvent[],e:NoteEdits,selection:NoteSelection,note:NoteEvent):NoteEdits {
  const next=copyEdits(e);
  if(selection.kind==='added') {
    if(!integer(selection.index,0,next.add.length-1)) throw new Error('Added note no longer exists.');
    next.add[selection.index]={...note};
  } else {
    if(!integer(selection.index,0,base.length-1)||next.remove.includes(selection.index)) throw new Error('Base note no longer exists.');
    next.replace=next.replace.filter(r=>r.index!==selection.index);
    if(noteKey(note)!==noteKey(base[selection.index])) next.replace.push({index:selection.index,note:{...note}});
    next.replace.sort((a,b)=>a.index-b.index);
  }
  return next;
}
export function removeNote(e:NoteEdits,selection:NoteSelection):NoteEdits {
  const next=copyEdits(e);
  if(selection.kind==='added') {
    if(!integer(selection.index,0,next.add.length-1)) throw new Error('Added note no longer exists.');
    next.add.splice(selection.index,1);
  } else {
    if(next.remove.includes(selection.index)) return next;
    next.replace=next.replace.filter(r=>r.index!==selection.index);
    next.remove.push(selection.index);
    next.remove.sort((a,b)=>a-b);
  }
  return next;
}
export function addNote(e:NoteEdits,note:NoteEvent):NoteEdits {
  return {...copyEdits(e),add:[...e.add.map(n=>({...n})),{...note}]};
}
export function makeHistory(digest:string):EditorHistory {
  return {past:[],present:emptyEdits(digest),future:[]};
}
export function commitEdit(h:EditorHistory,next:NoteEdits):EditorHistory {
  if(h.present.base_score_sha256!==next.base_score_sha256) throw new Error('Cannot combine different base scores.');
  if(JSON.stringify(h.present)===JSON.stringify(next)) return h;
  return {past:[...h.past,copyEdits(h.present)].slice(-HISTORY_LIMIT),present:copyEdits(next),future:[]};
}
export function undo(h:EditorHistory):EditorHistory {
  if(!h.past.length)return h;
  return {past:h.past.slice(0,-1),present:copyEdits(h.past[h.past.length-1]),future:[copyEdits(h.present),...h.future]};
}
export function redo(h:EditorHistory):EditorHistory {
  if(!h.future.length)return h;
  return {past:[...h.past,copyEdits(h.present)].slice(-HISTORY_LIMIT),present:copyEdits(h.future[0]),future:h.future.slice(1)};
}
