import { describe, expect, it } from 'vitest';
import { defaultComposition, validateScoreInput } from './music-contract';
import {
  addNote, changeNote, commitEdit, editCount, emptyEdits, makeHistory, parseNoteEvents,
  redo, removeNote, rowsFor, scoreFrame, undo, validateDraft, validateEditShape, validateNotes,
  type NoteEvent,
} from './music-note-editor';

const digest='a'.repeat(64);
const composition=defaultComposition();
const note=(override:Partial<NoteEvent>={}):NoteEvent=>({
  track:'lead',pitch:60,tick:0,length:4,velocity:.5,pan:0,...override
});
const base=[note(),note({pitch:63,tick:8}),note({track:'kick',pitch:36,tick:16})];

describe('note geometry and validation',()=>{
  it('matches the swung absolute score clock',()=>{
    const c={...composition,swing:.2};
    expect(scoreFrame(c,0)).toBe(0);
    expect(scoreFrame(c,4)).toBe(7200);
    expect(scoreFrame(c,8)).toBe(12000);
    expect(scoreFrame(c,32)).toBe(48000);
    expect(scoreFrame(c,128)).toBe(192000);
  });
  it('matches half-even swing rounding',()=>{
    const c={...composition,swing:.0005};
    expect(scoreFrame(c,4)).toBe(6000);
    expect(scoreFrame({...c,swing:.0015},4)).toBe(6012);
  });
  it.each<Partial<NoteEvent>>([
    {pitch:61},{pitch:97},{tick:-1},{length:0},{tick:127,length:2},
    {velocity:0},{pan:2},{velocity:NaN},{length:80},
  ])('rejects invalid note %j',override=>{
    expect(()=>validateNotes([note(override)],composition)).toThrow();
  });
  it('accepts drum pitches without enforcing the pitched scale',()=>{
    expect(()=>validateNotes([note({track:'hat',pitch:61})],composition)).not.toThrow();
  });
  it('allows adjacent but rejects overlapping monophonic notes',()=>{
    expect(()=>validateNotes([note(),note({tick:4,pitch:63})],composition)).not.toThrow();
    expect(()=>validateNotes([note(),note({tick:3,pitch:63})],composition)).toThrow('polyphony');
  });
  it('enforces same-pitch and four-voice polyphony independently',()=>{
    const chord=[48,51,55,60].map(pitch=>note({track:'keys',pitch}));
    expect(()=>validateNotes(chord,composition)).not.toThrow();
    expect(()=>validateNotes([...chord,note({track:'keys',pitch:63})],composition)).toThrow('polyphony');
    expect(()=>validateNotes([...chord.slice(0,2),note({track:'keys',pitch:48,velocity:.6})],composition)).toThrow('polyphony');
  });
  it('validates server note count, canonical order and exact fields',()=>{
    expect(parseNoteEvents(base,composition,3)).toEqual(base);
    expect(()=>parseNoteEvents(base,composition,4)).toThrow('count');
    expect(()=>parseNoteEvents([...base].reverse(),composition,3)).toThrow('canonical');
    expect(()=>parseNoteEvents([{...base[0],unknown:1}],composition,1)).toThrow('fields');
  });
});
describe('atomic note edit draft',()=>{
  it('replaces, removes and adds without shifting original indices',()=>{
    let draft=emptyEdits(digest);
    draft=changeNote(base,draft,{kind:'base',index:1},note({tick:8,pitch:67}));
    draft=removeNote(draft,{kind:'base',index:0});
    draft=addNote(draft,note({tick:0,pitch:63}));
    const rows=rowsFor(base,draft);
    expect(rows.map(r=>r.selection)).toEqual([{kind:'added',index:0},{kind:'base',index:1},{kind:'base',index:2}]);
    expect(rows.map(r=>r.note.pitch)).toEqual([63,67,36]);
    expect(editCount(draft)).toBe(3);
    expect(validateDraft(base,draft,composition)).toBeNull();
    expect(base[1].pitch).toBe(63);
  });
  it('restoring an original note removes its replacement operation',()=>{
    let draft=changeNote(base,emptyEdits(digest),{kind:'base',index:1},note({tick:8,pitch:67}));
    draft=changeNote(base,draft,{kind:'base',index:1},base[1]);
    expect(editCount(draft)).toBe(0);
  });
  it('editing/removing an added note changes only the addition list',()=>{
    let draft=addNote(emptyEdits(digest),note({tick:24}));
    draft=changeNote(base,draft,{kind:'added',index:0},note({tick:25}));
    expect(draft.add[0].tick).toBe(25);
    expect(removeNote(draft,{kind:'added',index:0}).add).toEqual([]);
    expect(draft.add.length).toBe(1);
  });
  it('rejects overlapping draft and stale source index',()=>{
    expect(validateDraft(base,addNote(emptyEdits(digest),note({tick:2,pitch:63})),composition)).toContain('polyphony');
    expect(()=>rowsFor(base,{...emptyEdits(digest),remove:[99]})).toThrow('unavailable');
  });
  it('rejects invalid transaction shape before request',()=>{
    expect(()=>validateEditShape(emptyEdits(digest),composition)).toThrow('1–128');
    expect(()=>validateEditShape({...emptyEdits(digest),remove:[2,1]},composition)).toThrow('sorted');
    expect(()=>validateEditShape({...emptyEdits(digest),remove:[1],replace:[{index:1,note:base[1]}]},composition)).toThrow('not removed');
    expect(()=>validateEditShape({...emptyEdits(digest),remove:Array.from({length:129},(_,i)=>i)},composition)).toThrow('1–128');
    expect(()=>validateScoreInput({composition,note_edits:{...emptyEdits(digest),add:[note({pitch:61})]}})).toThrow('scale');
  });
});
describe('editor history',()=>{
  it('undoes, redoes and clears redo on a new edit',()=>{
    let history=makeHistory(digest);
    history=commitEdit(history,addNote(history.present,note({tick:24})));
    history=commitEdit(history,removeNote(history.present,{kind:'base',index:0}));
    expect(editCount(history.present)).toBe(2);
    history=undo(history);
    expect(editCount(history.present)).toBe(1);
    history=redo(history);
    expect(editCount(history.present)).toBe(2);
    history=undo(history);
    history=commitEdit(history,removeNote(history.present,{kind:'base',index:1}));
    expect(history.future).toEqual([]);
    expect(history.present.remove).toEqual([1]);
  });
  it('keeps bounded immutable history',()=>{
    let history=makeHistory(digest);
    for(let i=0;i<80;i++) {
      const next={...emptyEdits(digest),add:[note({tick:i})]};
      history=commitEdit(history,next);
      next.add[0].tick=127;
    }
    expect(history.past.length).toBe(64);
    expect(history.present.add[0].tick).toBe(79);
    expect(()=>commitEdit(history,emptyEdits('b'.repeat(64)))).toThrow('different base');
  });
});
