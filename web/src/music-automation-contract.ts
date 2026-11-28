/** Pure score-clock automation editing and API receipt checks. */
import { TRACKS, SAMPLE_RATE, isDigest, validateMusicMix, previewFrames,
  type Composition, type Track, type MusicMix, type MusicRenderInput } from './music-contract';
import { scoreFrame } from './music-note-editor';

export const PARAMETERS = ['gain','balance'] as const;
export const LAWS = ['linear','smoothstep','step'] as const;
export const MAX_KEYS = 64;
export const MAX_AUTOMATION_BYTES = 128 * 1024;
export const AUTOMATION_FORMAT = 'sonumio-mix-automation-v1';
export type Parameter = typeof PARAMETERS[number];
export type Law = typeof LAWS[number];
export type AutomationKey = { tick:number; value:number; law:Law };
export type AutomationLane = { track:Track; parameter:Parameter; keys:AutomationKey[] };
export type AutomationInput = { lanes:AutomationLane[]; expected_automation_sha256?:string };
export type AutomationReceipt = {
  score_sha256:string; automation_sha256:string;
  frames:number; sample_rate:number; keys:number; engine:string; audio_rendered:false;
  lanes:Array<Omit<AutomationLane,'keys'> & {keys:Array<AutomationKey & {frame:number}>}>;
  mix:MusicMix;
};
type AutomationFile = {format:typeof AUTOMATION_FORMAT; lanes:AutomationLane[]; mix:MusicMix};
export type Selection = {track:Track; parameter:Parameter};

const object = (value:unknown):value is Record<string,unknown> =>
  value!==null && typeof value==='object' && !Array.isArray(value);
const exact = (value:Record<string,unknown>,keys:string[]) =>
  Object.keys(value).length===keys.length && keys.every(key=>Object.hasOwn(value,key));
const finite=(value:unknown,lo:number,hi:number):value is number =>
  typeof value==='number' && Number.isFinite(value) && value>=lo && value<=hi;
const integer=(value:unknown,lo:number,hi:number):value is number =>
  finite(value,lo,hi) && Number.isInteger(value);

export function laneOrder(a:Selection,b:Selection):number {
  return TRACKS.indexOf(a.track)-TRACKS.indexOf(b.track) ||
    PARAMETERS.indexOf(a.parameter)-PARAMETERS.indexOf(b.parameter);
}
const sameLane=(a:Selection,b:Selection)=>a.track===b.track&&a.parameter===b.parameter;
export function bounds(parameter:Parameter):[number,number] {
  return parameter==='gain'?[0,2]:[-1,1];
}
export function parseKey(raw:unknown,parameter:Parameter,totalTicks:number):AutomationKey {
  const [min,max]=bounds(parameter);
  if(!object(raw)||!exact(raw,['tick','value','law'])||
    !integer(raw.tick,0,totalTicks)||!finite(raw.value,min,max)||!LAWS.includes(raw.law as Law))
    throw new Error('Invalid automation key: check tick, value and interpolation.');
  return {tick:raw.tick,value:raw.value,law:raw.law as Law};
}
export function parseLanes(raw:unknown,composition:Composition):AutomationLane[] {
  if(!Array.isArray(raw)||raw.length>TRACKS.length*PARAMETERS.length) throw new Error('Invalid automation lane count.');
  const result:AutomationLane[]=[];
  for(const value of raw) {
    if(!object(value)||!exact(value,['track','parameter','keys'])||
      !TRACKS.includes(value.track as Track)||!PARAMETERS.includes(value.parameter as Parameter)||
      !Array.isArray(value.keys)||value.keys.length<1||value.keys.length>MAX_KEYS)
      throw new Error('Invalid automation lane.');
    const track=value.track as Track,parameter=value.parameter as Parameter;
    const keys=value.keys.map(key=>parseKey(key,parameter,composition.bars*32));
    if(keys.some((key,i)=>i>0&&key.tick<=keys[i-1].tick))
      throw new Error('Automation key ticks must be unique and increasing.');
    const lane={track,parameter,keys};
    if(result.length&&laneOrder(result[result.length-1],lane)>=0)
      throw new Error('Automation lanes must be unique and canonically ordered.');
    result.push(lane);
  }
  return result;
}
export function validateAutomation(input:MusicRenderInput):void {
  if(input.automation===undefined)return;
  const a=input.automation;
  if(!object(a)||!Object.hasOwn(a,'lanes')||
    Object.keys(a).some(k=>!['lanes','expected_automation_sha256'].includes(k))||
    (a.expected_automation_sha256!==undefined&&!isDigest(a.expected_automation_sha256)))
    throw new Error('Invalid automation request.');
  parseLanes(a.lanes,input.composition);
}
export function copyLanes(lanes:AutomationLane[]):AutomationLane[] {
  return lanes.map(lane=>({...lane,keys:lane.keys.map(key=>({...key}))}));
}
export function findLane(lanes:AutomationLane[],selection:Selection):AutomationLane|undefined {
  return lanes.find(lane=>sameLane(lane,selection));
}
export function changeKey(lanes:AutomationLane[],selection:Selection,previousTick:number|null,
                          key:AutomationKey,c:Composition):AutomationLane[] {
  const checked=parseKey(key,selection.parameter,c.bars*32);
  const result=copyLanes(lanes);
  const index=result.findIndex(lane=>sameLane(lane,selection));
  const keys=index===-1?[]:result[index].keys;
  if(previousTick!==null&&!keys.some(row=>row.tick===previousTick))
    throw new Error('The original automation key no longer exists.');
  if(keys.some(row=>row.tick===checked.tick&&row.tick!==previousTick))
    throw new Error('Another automation key already uses this tick.');
  const next=[...keys.filter(row=>row.tick!==previousTick),checked].sort((a,b)=>a.tick-b.tick);
  if(next.length>MAX_KEYS)throw new Error('A lane supports at most 64 keys.');
  if(index===-1)result.push({...selection,keys:next});
  else result[index]={...selection,keys:next};
  result.sort(laneOrder);
  return parseLanes(result,c);
}
export function removeKey(lanes:AutomationLane[],selection:Selection,tick:number):AutomationLane[] {
  return copyLanes(lanes).flatMap(lane=>{
    if(!sameLane(lane,selection))return [lane];
    const keys=lane.keys.filter(key=>key.tick!==tick);
    return keys.length?[{...lane,keys}]:[];
  });
}
export function removeLane(lanes:AutomationLane[],selection:Selection):AutomationLane[] {
  return copyLanes(lanes).filter(lane=>!sameLane(lane,selection));
}

export function evaluateLane(lane:AutomationLane,c:Composition,frame:number):number {
  if(!Number.isFinite(frame)||frame<0||frame>previewFrames(c))throw new Error('Invalid automation preview frame.');
  const points=lane.keys.map(key=>({...key,frame:scoreFrame(c,key.tick)}));
  if(frame<=points[0].frame)return points[0].value;
  const last=points[points.length-1];
  if(frame>=last.frame)return last.value;
  const i=points.findIndex(key=>key.frame>frame);
  const a=points[i-1],b=points[i];
  let phase=(frame-a.frame)/(b.frame-a.frame);
  if(a.law==='step')phase=0;
  else if(a.law==='smoothstep')phase=phase*phase*(3-2*phase);
  return a.value+(b.value-a.value)*phase;
}

/** Preview vertices include both sides of every step discontinuity. */
export function curvePoints(lane:AutomationLane,c:Composition):Array<[number,number]> {
  const frames=previewFrames(c), points:Array<[number,number]>=[];
  const keys=lane.keys.map(key=>({...key,frame:scoreFrame(c,key.tick)}));
  points.push([0,keys[0].value]);
  keys.forEach((key,index)=>{
    points.push([key.frame, key.value]);
    if(index===keys.length-1)return;
    const next=keys[index+1];
    if(key.law==='step')points.push([next.frame,key.value]);
    else if(key.law==='smoothstep')for(let j=1;j<16;j++) {
      const f=key.frame+(next.frame-key.frame)*j/16;
      points.push([f,evaluateLane(lane,c,f)]);
    }
  });
  points.push([frames,keys[keys.length-1].value]);
  return points;
}
function parseMix(raw:unknown):MusicMix {
  if(!object(raw)||!exact(raw,['gains','mute','solo','room','peak_ceiling']))
    throw new Error('Invalid automation mix fields.');
  const mix=raw as MusicMix;
  const problem=validateMusicMix(mix);
  if(problem)throw new Error(problem);
  return {gains:[...mix.gains],mute:TRACKS.filter(t=>mix.mute.includes(t)),
    solo:TRACKS.filter(t=>mix.solo.includes(t)),room:mix.room,peak_ceiling:mix.peak_ceiling};
}
const mixKey=(mix:MusicMix)=>JSON.stringify(parseMix(mix));
export function parseAutomationFile(raw:string,c:Composition):AutomationFile {
  if(new TextEncoder().encode(raw).length>MAX_AUTOMATION_BYTES)throw new Error('Automation file exceeds 128 KiB.');
  let value:unknown;
  try{value=JSON.parse(raw);}catch{throw new Error('Invalid automation JSON.');}
  if(!object(value)||!exact(value,['format','lanes','mix'])||value.format!==AUTOMATION_FORMAT)
    throw new Error('Unsupported automation file.');
  return {format:AUTOMATION_FORMAT,lanes:parseLanes(value.lanes,c),mix:parseMix(value.mix)};
}
export function parseAutomationReceipt(raw:unknown,input:MusicRenderInput):AutomationReceipt {
  validateAutomation(input);
  if(!object(raw)||raw.engine!==AUTOMATION_FORMAT||raw.audio_rendered!==false||
    !isDigest(raw.score_sha256)||!isDigest(raw.automation_sha256)||
    raw.sample_rate!==SAMPLE_RATE||raw.frames!==previewFrames(input.composition)||
    !Array.isArray(raw.lanes)||!integer(raw.keys,0,16*MAX_KEYS)||
    (input.expected_score_sha256!==undefined&&raw.score_sha256!==input.expected_score_sha256)||
    (input.automation?.expected_automation_sha256!==undefined&&raw.automation_sha256!==input.automation.expected_automation_sha256))
    throw new Error('Invalid automation receipt identity or geometry.');
  if(mixKey(parseMix(raw.mix))!==mixKey(input.mix))throw new Error('Automation receipt mix differs.');
  const stripped=raw.lanes.map(lane=>{
    if(!object(lane)||!exact(lane,['track','parameter','keys'])||!Array.isArray(lane.keys))
      throw new Error('Invalid automation receipt lane.');
    return {track:lane.track,parameter:lane.parameter,keys:lane.keys.map(key=>{
      if(!object(key)||!exact(key,['tick','value','law','frame'])||
        !integer(key.tick,0,input.composition.bars*32)||key.frame!==scoreFrame(input.composition,key.tick))
        throw new Error('Automation receipt sample-clock mismatch.');
      return {tick:key.tick,value:key.value,law:key.law};
    })};
  });
  const lanes=parseLanes(stripped,input.composition);
  const expected=parseLanes(input.automation?.lanes??[],input.composition);
  if(JSON.stringify(lanes)!==JSON.stringify(expected)||raw.keys!==lanes.reduce((n,l)=>n+l.keys.length,0))
    throw new Error('Automation receipt differs from requested lanes.');
  return raw as AutomationReceipt;
}
