/** Project snapshots preserve the original file bytes; never stringify parsed
 * score data to reconstruct its digest-bearing document. */
import { isDigest, parseMusicPlan, validateMusicMix, type MusicMix,
  type MusicPlan, type Composition } from './music-contract';
import { validateEditShape, type NoteEdits } from './music-note-editor';

export const MAX_PROJECT_BYTES=2*1024*1024;
export const MAX_SCORE_BYTES=1024*1024;
export const MAX_AUTOMATION_BYTES=128*1024;
export const PROJECT_FORMAT='sonumio-music-project-v1';
export type ProjectSnapshot={
  format:typeof PROJECT_FORMAT;
  project_sha256:string; score_sha256:string; automation_sha256:string;
  score_bytes_sha256:string; content_sha256:string;
  title:string; revision:number; parent_project_sha256:string|null;
  events:number; bars:number; frames:number; sample_rate:number; duration_seconds:number;
  automation_lane_count:number; automation_keys:number;
  automation_lanes:Array<{track:string;parameter:string;keys:Array<{tick:number;value:number;law:string}>}>;
  mix:MusicMix; preview_supported:boolean; score_plan:MusicPlan|null;
  embedded_audio:false; embedded_weights:false; external_paths:false; authenticated:false; parent_verified:false;
};
export type SourceScore={document:string;score_sha256:string;frames:number;bars:number;events:number;preview_supported:boolean};
export type SourceAutomation={document:string;automation_sha256:string;lanes:number;keys:number};
export type OpenProject={document:string;snapshot:ProjectSnapshot};
export type ProjectCreate={
  title:string;score_document:string;expected_score_sha256:string;
  automation_document?:string;expected_automation_sha256?:string;
};
export type ProjectRevision={
  project_document:string;expected_project_sha256:string;title?:string;
  note_edits?:NoteEdits;mix?:MusicMix;
  score_document?:string;expected_score_sha256?:string;
  automation_document?:string;expected_automation_sha256?:string;
};
const record=(value:unknown):value is Record<string,unknown>=>
  value!==null&&typeof value==='object'&&!Array.isArray(value);
const integer=(value:unknown,lo:number,hi:number):value is number=>
  typeof value==='number'&&Number.isInteger(value)&&value>=lo&&value<=hi;
const TRACKS=['keys','pad','bass','lead','pluck','kick','snare','hat'];
const PARAMS=['gain','balance'];
const LAWS=['linear','smoothstep','step'];
const finite=(n:unknown,lo:number,hi:number):n is number=>
  typeof n==='number'&&Number.isFinite(n)&&n>=lo&&n<=hi;
export function validTitle(title:string):boolean {
  return typeof title==='string'&&Array.from(title).length>=1&&Array.from(title).length<=80&&
    title===title.trim()&&title===title.normalize('NFC')&&!/\p{C}/u.test(title);
}
export function checkedDocument(text:string,max:number):string {
  if(typeof text!=='string'||text.length===0||text.length>max||
    new TextEncoder().encode(text).byteLength>max)
    throw new Error('Document exceeds its local byte limit or is empty.');
  return text;
}
export async function readDocument(file:File,max:number):Promise<string>{
  if(file.size<1||file.size>max)throw new Error('Selected file exceeds the document byte limit or is empty.');
  const bytes=await file.arrayBuffer();
  if(bytes.byteLength!==file.size)throw new Error('Selected file changed while reading.');
  return checkedDocument(new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes),max);
}
export function declaredProjectId(document:string):string {
  checkedDocument(document,MAX_PROJECT_BYTES);
  let raw:unknown;
  try{raw=JSON.parse(document);}catch{throw new Error('Invalid native project JSON.');}
  if(!record(raw)||raw.format!==PROJECT_FORMAT||!isDigest(raw.project_sha256))
    throw new Error('Unsupported project format or missing identity.');
  return raw.project_sha256;
}
export function parseScoreSource(document:string,raw:unknown):SourceScore{
  checkedDocument(document,MAX_SCORE_BYTES);
  if(!record(raw)||!isDigest(raw.score_sha256)||!integer(raw.frames,1,2880000)||
    !integer(raw.bars,4,32)||!integer(raw.events,1,4096)||
    raw.preview_supported!==(raw.bars<=16&&raw.frames<=720000))
    throw new Error('Invalid score inspection receipt.');
  return {...raw,document} as SourceScore;
}
export function parseAutomationSource(document:string,raw:unknown):SourceAutomation{
  checkedDocument(document,MAX_AUTOMATION_BYTES);
  if(!record(raw)||!isDigest(raw.automation_sha256)||!integer(raw.lanes,0,16)||
    !integer(raw.keys,raw.lanes,raw.lanes*64))
    throw new Error('Invalid automation inspection receipt.');
  return {...raw,document} as SourceAutomation;
}
export function parseProjectSnapshot(raw:unknown,expected:string):ProjectSnapshot {
  if(!isDigest(expected)||!record(raw)||raw.format!==PROJECT_FORMAT||
    raw.project_sha256!==expected||!isDigest(raw.score_sha256)||!isDigest(raw.automation_sha256)||
    !isDigest(raw.score_bytes_sha256)||!isDigest(raw.content_sha256)||
    typeof raw.title!=='string'||!validTitle(raw.title)||!integer(raw.revision,0,2147483647)||
    (raw.revision===0?raw.parent_project_sha256!==null:!isDigest(raw.parent_project_sha256))||
    !integer(raw.events,1,4096)||!integer(raw.bars,4,32)||!integer(raw.frames,1,2880000)||
    raw.sample_rate!==24000||raw.duration_seconds!==raw.frames/24000||
    raw.preview_supported!==(raw.bars<=16&&raw.frames<=720000)||
    raw.embedded_audio!==false||raw.embedded_weights!==false||raw.external_paths!==false||
    raw.authenticated!==false||raw.parent_verified!==false)
    throw new Error('Invalid native project identity or geometry.');
  if(!record(raw.mix)||validateMusicMix(raw.mix as MusicMix))
    throw new Error('Invalid project mixer.');
  if(!integer(raw.automation_lane_count,0,16)||!Array.isArray(raw.automation_lanes)||
    raw.automation_lanes.length!==raw.automation_lane_count||
    !integer(raw.automation_keys,0,1024))throw new Error('Invalid project automation count.');
  let previous=-1,keys=0;
  for(const lane of raw.automation_lanes){
    if(!record(lane)||typeof lane.track!=='string'||typeof lane.parameter!=='string'||
      !TRACKS.includes(lane.track)||!PARAMS.includes(lane.parameter)||
      !Array.isArray(lane.keys)||lane.keys.length<1||lane.keys.length>64)
      throw new Error('Invalid project automation lane.');
    const order=TRACKS.indexOf(lane.track)*2+PARAMS.indexOf(lane.parameter);
    if(order<=previous)throw new Error('Project automation lane order differs.');
    previous=order;
    let tick=-1;
    for(const key of lane.keys){
      if(!record(key)||!integer(key.tick,0,raw.bars*32)||key.tick<=tick||
        !finite(key.value,lane.parameter==='gain'?0:-1,lane.parameter==='gain'?2:1)||
        typeof key.law!=='string'||!LAWS.includes(key.law))
        throw new Error('Invalid project automation key.');
      tick=key.tick;keys++;
    }
  }
  if(keys!==raw.automation_keys)throw new Error('Project automation key count differs.');
  if(raw.preview_supported){
    if(!record(raw.score_plan)||!record(raw.score_plan.composition))
      throw new Error('Project preview plan is missing.');
    const plan=parseMusicPlan(raw.score_plan,{
      composition:raw.score_plan.composition as Composition,expected_score_sha256:raw.score_sha256,
    });
    if(!plan.note_events||plan.frames!==raw.frames||plan.events!==raw.events||plan.bars!==raw.bars)
      throw new Error('Project score plan differs from project geometry.');
  }else if(raw.score_plan!==null)throw new Error('Unsupported project exposes a preview plan.');
  return raw as ProjectSnapshot;
}
export function createInput(title:string,score:SourceScore,automation?:SourceAutomation|null):ProjectCreate {
  if(!validTitle(title))throw new Error('Project title must contain 1–80 printable NFC characters.');
  if(!isDigest(score.score_sha256))throw new Error('Inspect the score first.');
  checkedDocument(score.document,MAX_SCORE_BYTES);
  const input:ProjectCreate={title,score_document:score.document,expected_score_sha256:score.score_sha256};
  if(automation){
    if(!isDigest(automation.automation_sha256))throw new Error('Inspect the automation first.');
    input.automation_document=checkedDocument(automation.document,MAX_AUTOMATION_BYTES);
    input.expected_automation_sha256=automation.automation_sha256;
  }
  return input;
}
export function revisionInput(project:OpenProject,title:string,edits:NoteEdits|null,mix:MusicMix):ProjectRevision {
  if(!validTitle(title))throw new Error('Invalid project title.');
  const snapshot=project.snapshot;
  if(declaredProjectId(project.document)!==snapshot.project_sha256)throw new Error('Project file and inspected identity differ.');
  const input:ProjectRevision={project_document:project.document,expected_project_sha256:snapshot.project_sha256};
  if(title!==snapshot.title)input.title=title;
  if(edits){
    if(!snapshot.score_plan||edits.base_score_sha256!==snapshot.score_sha256)
      throw new Error('Note edits require the inspected project score.');
    validateEditShape(edits,snapshot.score_plan.composition);
    input.note_edits=edits;
  }
  const problem=validateMusicMix(mix);
  if(problem)throw new Error(problem);
  const normalized=(value:MusicMix)=>JSON.stringify({
    gains:value.gains,mute:TRACKS.filter(t=>value.mute.includes(t as MusicMix['mute'][number])),
    solo:TRACKS.filter(t=>value.solo.includes(t as MusicMix['solo'][number])),
    room:value.room,peak_ceiling:value.peak_ceiling,
  });
  if(normalized(mix)!==normalized(snapshot.mix))input.mix=mix;
  if(!input.title&&!input.note_edits&&!input.mix)throw new Error('No project changes to save.');
  return input;
}
