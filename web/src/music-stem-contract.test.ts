import { describe, expect, it } from 'vitest';
import { defaultComposition, defaultMusicMix, previewFrames, MUSIC_ENGINE } from './music-contract';
import { parseStemPlan,parseStemResult,validateStemInput,verifyStemWav,type StemInput,type StemPlan } from './music-stem-contract';

const digest='a'.repeat(64);
const auto='b'.repeat(64);
const effective='c'.repeat(64);
const input:StemInput={composition:defaultComposition(),mix:defaultMusicMix(),
  track:'lead',audition:'dry',expected_score_sha256:digest};
function plan():StemPlan{
  const frames=previewFrames(input.composition);
  return {protocol:'sonumio-studio-stem-audition-v1',engine_id:MUSIC_ENGINE,model_id:MUSIC_ENGINE,
    score_sha256:digest,track:'lead',audition:'dry',frames,sample_rate:24000,channels:2,
    duration_seconds:frames/24000,automation_sha256:null,effective_automation_sha256:null,
    static_mix_applied:false,room_applied:false,mute_solo_overridden:false,common_dry_attenuation:true,
    learned_model:false,external_recordings_used:false,disclaimer:'procedural'};
}
function wav(frames:number){
  const bytes=new Uint8Array(44+4*frames);
  const view=new DataView(bytes.buffer);
  const tag=(offset:number,text:string)=>bytes.set(new TextEncoder().encode(text),offset);
  tag(0,'RIFF');tag(8,'WAVE');tag(12,'fmt ');tag(36,'data');
  view.setUint32(4,bytes.length-8,true);view.setUint32(16,16,true);
  view.setUint16(20,1,true);view.setUint16(22,2,true);
  view.setUint32(24,24000,true);view.setUint32(28,96000,true);
  view.setUint16(32,4,true);view.setUint16(34,16,true);view.setUint32(40,frames*4,true);
  return bytes;
}
async function hash(bytes:Uint8Array<ArrayBuffer>){
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),
    x=>x.toString(16).padStart(2,'0')).join('');
}

describe('stem receipts',()=>{
  it('accepts pinned dry and solo requests',()=>{
    expect(parseStemPlan(plan(),input)).toEqual(plan());
    const solo={...input,audition:'solo' as const,automation:{lanes:[],expected_automation_sha256:auto}};
    const receipt={...plan(),audition:'solo',automation_sha256:auto,effective_automation_sha256:effective,
      static_mix_applied:true,room_applied:true,mute_solo_overridden:true,common_dry_attenuation:false};
    expect(parseStemPlan(receipt,solo)).toEqual(receipt);
    expect(()=>validateStemInput({...input,audition:'solo'})).toThrow(/automation/);
  });
  it.each([{track:'bass'},{score_sha256:'0'.repeat(64)},{sample_rate:48000},
    {channels:1},{frames:3},{room_applied:true},{static_mix_applied:true},
    {effective_automation_sha256:effective},{learned_model:true}])('rejects conflicting plan %j',patch=>{
      expect(()=>parseStemPlan({...plan(),...patch},input)).toThrow();
  });
  it('binds result to plan and checksum',()=>{
    const p=plan();
    const result={...p,wav_sha256:auto,master_attenuation:1,peak_ceiling:.9};
    expect(parseStemResult(result,p)).toEqual(result);
    expect(()=>parseStemResult({...result,track:'bass'},p)).toThrow();
    expect(()=>parseStemResult({...result,wav_sha256:'bad'},p)).toThrow();
    expect(()=>parseStemResult({...result,master_attenuation:2},p)).toThrow();
  });
  it('validates canonical WAV bytes and detects tampering',async()=>{
    const p=plan();
    const bytes=wav(p.frames), digest=await hash(bytes);
    await expect(verifyStemWav(bytes,p,digest)).resolves.toBeUndefined();
    bytes[44]=1;
    await expect(verifyStemWav(bytes,p,digest)).rejects.toThrow(/checksum/);
    bytes[22]=1;
    await expect(verifyStemWav(bytes,p,await hash(bytes))).rejects.toThrow(/canonical/);
    await expect(verifyStemWav(bytes.slice(0,-1),p,digest)).rejects.toThrow(/geometry/);
  });
});
