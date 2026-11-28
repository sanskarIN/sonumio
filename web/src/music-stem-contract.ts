/** Strict browser-side audition receipts and canonical PCM16 geometry. */
import { TRACKS, MUSIC_ENGINE, SAMPLE_RATE, previewFrames, isDigest,
  validateScoreInput, validateMusicMix, type Track, type MusicRenderInput } from './music-contract';
import { validateAutomation } from './music-automation-contract';

export type StemMode='dry'|'solo';
export type StemInput=MusicRenderInput & { track:Track; audition:StemMode; expected_score_sha256:string };
export type StemPlan={
  protocol:'sonumio-studio-stem-audition-v1'; engine_id:string; model_id:string;
  score_sha256:string; track:Track; audition:StemMode; frames:number;
  sample_rate:number; channels:2; duration_seconds:number;
  automation_sha256:string|null; effective_automation_sha256:string|null;
  static_mix_applied:boolean; room_applied:boolean; mute_solo_overridden:boolean;
  common_dry_attenuation:boolean; learned_model:false; external_recordings_used:false;
  disclaimer:string;
};
export type StemResult=StemPlan & {wav_sha256:string; master_attenuation:number; peak_ceiling:number};

function record(value:unknown):value is Record<string,unknown>{
  return typeof value==='object'&&value!==null&&!Array.isArray(value);
}
function requirePlan(value:unknown):StemPlan {
  if(!record(value)||value.protocol!=='sonumio-studio-stem-audition-v1'||value.engine_id!==MUSIC_ENGINE||
      value.model_id!==MUSIC_ENGINE||!isDigest(value.score_sha256)||!TRACKS.includes(value.track as Track)||
      (value.audition!=='dry'&&value.audition!=='solo')||value.sample_rate!==SAMPLE_RATE||value.channels!==2||
      typeof value.frames!=='number'||!Number.isInteger(value.frames)||value.frames<1||value.frames>720000||
      value.duration_seconds!==value.frames/SAMPLE_RATE||value.learned_model!==false||
      value.external_recordings_used!==false||typeof value.disclaimer!=='string')
    throw new Error('Invalid stem audition receipt.');
  const dry=value.audition==='dry';
  if(value.static_mix_applied===dry||value.mute_solo_overridden===dry||
      value.common_dry_attenuation!==dry||value.static_mix_applied!==!dry||
      value.mute_solo_overridden!==!dry||typeof value.room_applied!=='boolean'||
      (dry?(value.automation_sha256!==null||value.effective_automation_sha256!==null||value.room_applied):
        (!isDigest(value.automation_sha256)||!isDigest(value.effective_automation_sha256))))
    throw new Error('Invalid stem processing provenance.');
  return value as StemPlan;
}
export function validateStemInput(input:StemInput):void {
  validateScoreInput(input);
  const error=validateMusicMix(input.mix);
  if(error)throw new Error(error);
  validateAutomation(input);
  if(!isDigest(input.expected_score_sha256)||!TRACKS.includes(input.track)||
    !['dry','solo'].includes(input.audition))
    throw new Error('Choose a validated score and instrument audition mode.');
  if(input.audition==='solo'&&!isDigest(input.automation?.expected_automation_sha256))
    throw new Error('Validate the current automation before solo audition.');
}
export function parseStemPlan(value:unknown,input:StemInput):StemPlan {
  validateStemInput(input);
  const plan=requirePlan(value);
  if(plan.track!==input.track||plan.audition!==input.audition||
      plan.score_sha256!==input.expected_score_sha256||plan.frames!==previewFrames(input.composition)||
      (plan.audition==='solo'&&(plan.automation_sha256!==input.automation?.expected_automation_sha256||
        plan.room_applied!==(input.mix.room!==0))))
    throw new Error('Stem plan does not match the requested score or mix.');
  return plan;
}
export function parseStemResult(value:unknown,plan:StemPlan):StemResult {
  const actual=requirePlan(value);
  const keys: (keyof StemPlan)[]=['protocol','engine_id','model_id','score_sha256','track','audition',
    'frames','sample_rate','channels','duration_seconds','automation_sha256','effective_automation_sha256',
    'static_mix_applied','room_applied','mute_solo_overridden','common_dry_attenuation',
    'learned_model','external_recordings_used','disclaimer'];
  if(keys.some(key=>actual[key]!==plan[key]))throw new Error('Stem result differs from the planned audition.');
  if(!record(value)||!isDigest(value.wav_sha256)||
      typeof value.master_attenuation!=='number'||!Number.isFinite(value.master_attenuation)||
      value.master_attenuation<=0||value.master_attenuation>1||
      typeof value.peak_ceiling!=='number'||!Number.isFinite(value.peak_ceiling)||
      value.peak_ceiling<.05||value.peak_ceiling>.99||
      (plan.audition==='dry'&&value.peak_ceiling!==.9))
    throw new Error('Invalid stem WAV receipt.');
  return value as StemResult;
}
export async function verifyStemWav(bytes:Uint8Array<ArrayBuffer>,plan:StemPlan,expected:string):Promise<void>{
  if(bytes.length!==44+4*plan.frames||!isDigest(expected))
    throw new Error('Invalid stem WAV geometry.');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  const tag=(offset:number)=>String.fromCharCode(...bytes.subarray(offset,offset+4));
  if(tag(0)!=='RIFF'||tag(8)!=='WAVE'||tag(12)!=='fmt '||tag(36)!=='data'||
    view.getUint32(4,true)!==bytes.length-8||view.getUint32(16,true)!==16||
    view.getUint16(20,true)!==1||view.getUint16(22,true)!==2||
    view.getUint32(24,true)!==SAMPLE_RATE||view.getUint32(28,true)!==SAMPLE_RATE*4||
    view.getUint16(32,true)!==4||view.getUint16(34,true)!==16||
    view.getUint32(40,true)!==plan.frames*4)
    throw new Error('Stem audio is not canonical stereo PCM16 WAV.');
  const hash=await crypto.subtle.digest('SHA-256',bytes);
  const digest=Array.from(new Uint8Array(hash),x=>x.toString(16).padStart(2,'0')).join('');
  if(digest!==expected)throw new Error('Stem WAV checksum mismatch.');
}
