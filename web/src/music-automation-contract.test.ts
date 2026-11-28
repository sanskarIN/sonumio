import { describe,expect,it } from 'vitest';
import { defaultComposition,defaultMusicMix,type MusicRenderInput } from './music-contract';
import { scoreFrame } from './music-note-editor';
import { AUTOMATION_FORMAT, changeKey, copyLanes, curvePoints, evaluateLane, findLane,
  parseAutomationFile, parseAutomationReceipt, parseLanes, removeKey, removeLane,
  validateAutomation, type AutomationLane, type AutomationReceipt, type Selection } from './music-automation-contract';

const c={...defaultComposition(),swing:.2};
const selection:Selection={track:'lead',parameter:'gain'};
const lane:AutomationLane={...selection,keys:[
  {tick:0,value:.2,law:'linear'},{tick:4,value:1.2,law:'smoothstep'},
  {tick:8,value:.8,law:'step'},{tick:16,value:1.4,law:'linear'},
]};
const digest='a'.repeat(64),autoDigest='b'.repeat(64);
const input:MusicRenderInput={composition:c,mix:defaultMusicMix(),
  expected_score_sha256:digest,automation:{lanes:[lane],expected_automation_sha256:autoDigest}};
function receipt():AutomationReceipt {
  return {score_sha256:digest,automation_sha256:autoDigest,frames:192000,sample_rate:24000,
    keys:4,engine:AUTOMATION_FORMAT,audio_rendered:false,mix:defaultMusicMix(),
    lanes:[{...lane,keys:lane.keys.map(key=>({...key,frame:scoreFrame(c,key.tick)}))}]};
}
describe('automation key validation and edits',()=>{
  it('adds and moves keys without modifying the input',()=>{
    let result=changeKey([],selection,null,{tick:4,value:1,law:'step'},c);
    const prior=copyLanes(result);
    result=changeKey(result,selection,4,{tick:8,value:.5,law:'linear'},c);
    expect(findLane(result,selection)?.keys[0].tick).toBe(8);
    expect(prior[0].keys[0].tick).toBe(4);
    result=changeKey(result,{track:'keys',parameter:'balance'},null,{tick:0,value:-1,law:'linear'},c);
    expect(result.map(row=>row.track)).toEqual(['keys','lead']);
    expect(removeKey(result,selection,8).map(row=>row.track)).toEqual(['keys']);
    expect(removeLane(result,{track:'keys',parameter:'balance'})).toEqual([result[1]]);
  });
  it('rejects a missing source key and duplicate destination ticks',()=>{
    expect(()=>changeKey([lane],selection,99,{tick:2,value:1,law:'linear'},c)).toThrow('no longer exists');
    expect(()=>changeKey([lane],selection,4,{tick:8,value:1,law:'linear'},c)).toThrow('already uses');
  });
  it.each([
    {tick:-1,value:1,law:'linear'},{tick:129,value:1,law:'linear'},
    {tick:1,value:-1,law:'linear'},{tick:1,value:2.1,law:'linear'},
    {tick:1,value:NaN,law:'linear'},{tick:1,value:1,law:'cubic'},
  ])('rejects invalid key %j',key=>{
    expect(()=>parseLanes([{...lane,keys:[key]}],c)).toThrow();
  });
  it('rejects wrong ordering and duplicate identities',()=>{
    expect(()=>parseLanes([lane,lane],c)).toThrow('unique');
    expect(()=>parseLanes([{...lane,keys:[...lane.keys].reverse()}],c)).toThrow('increasing');
    expect(()=>parseLanes([{...lane,parameter:'balance',keys:[{tick:0,value:1.1,law:'linear'}]}],c)).toThrow();
    expect(()=>parseLanes([{...lane,keys:Array.from({length:65},(_,tick)=>({tick,value:1,law:'linear'}))}],c)).toThrow();
  });
});
describe('exact sample-clock curve preview',()=>{
  it('holds endpoints and interpolates on actual frames',()=>{
    expect(evaluateLane(lane,c,0)).toBe(.2);
    expect(evaluateLane(lane,c,3600)).toBeCloseTo(.7);
    expect(evaluateLane(lane,c,7200)).toBe(1.2);
    expect(evaluateLane(lane,c,9600)).toBeCloseTo(1);
    expect(evaluateLane(lane,c,23999)).toBe(.8);
    expect(evaluateLane(lane,c,24000)).toBe(1.4);
    expect(evaluateLane(lane,c,192000)).toBe(1.4);
  });
  it('retains vertical step discontinuities',()=>{
    const points=curvePoints(lane,c);
    const same=points.filter(([frame])=>frame===24000);
    expect(same).toEqual([[24000,.8],[24000,1.4]]);
    expect(points[0]).toEqual([0,.2]);
    expect(points.at(-1)).toEqual([192000,1.4]);
  });
});
describe('file and server receipt validation',()=>{
  it('accepts canonical file and normalizes track selectors',()=>{
    const raw=JSON.stringify({format:AUTOMATION_FORMAT,lanes:[lane],
      mix:{...defaultMusicMix(),mute:['hat','keys']}});
    expect(parseAutomationFile(raw,c).mix.mute).toEqual(['keys','hat']);
    expect(()=>parseAutomationFile('not JSON',c)).toThrow();
    expect(()=>parseAutomationFile(JSON.stringify({format:AUTOMATION_FORMAT,lanes:[lane],mix:{}}),c)).toThrow();
  });
  it('binds exact key values and frames and mix to the request',()=>{
    expect(parseAutomationReceipt(receipt(),input)).toEqual(receipt());
    const changed=receipt();
    changed.lanes[0].keys[0].value=.3;
    expect(()=>parseAutomationReceipt(changed,input)).toThrow('differs');
    const wrong=receipt();wrong.lanes[0].keys[1].frame++;
    expect(()=>parseAutomationReceipt(wrong,input)).toThrow('sample-clock');
    const badMix=receipt();badMix.mix.room=.7;
    expect(()=>parseAutomationReceipt(badMix,input)).toThrow('mix differs');
  });
  it('rejects stale score and automation identity',()=>{
    expect(()=>parseAutomationReceipt({...receipt(),score_sha256:'0'.repeat(64)},input)).toThrow();
    expect(()=>parseAutomationReceipt({...receipt(),automation_sha256:'0'.repeat(64)},input)).toThrow();
    expect(()=>parseAutomationReceipt({...receipt(),keys:3},input)).toThrow();
    expect(()=>validateAutomation({...input,automation:{lanes:[lane],expected_automation_sha256:'bad'}})).toThrow();
  });
});
