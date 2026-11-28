import { describe,expect,it } from 'vitest';
import { defaultMusicMix } from './music-contract';
import { PROJECT_FORMAT,createInput,revisionInput,declaredProjectId,parseProjectSnapshot,
  parseScoreSource,parseAutomationSource,validTitle,checkedDocument,
  type OpenProject,type ProjectSnapshot } from './music-project-contract';

const id='a'.repeat(64),score='b'.repeat(64),automation='c'.repeat(64);
const document=JSON.stringify({format:PROJECT_FORMAT,project_sha256:id});
function snapshot():ProjectSnapshot{
  return {format:PROJECT_FORMAT,project_sha256:id,score_sha256:score,automation_sha256:automation,
    score_bytes_sha256:'d'.repeat(64),content_sha256:'e'.repeat(64),
    title:'Original music',revision:0,parent_project_sha256:null,
    events:20,bars:24,frames:1152000,sample_rate:24000,duration_seconds:48,
    automation_lane_count:1,automation_keys:2,
    automation_lanes:[{track:'lead',parameter:'gain',keys:[{tick:0,value:.2,law:'linear'},{tick:128,value:1,law:'step'}]}],
    mix:defaultMusicMix(),preview_supported:false,score_plan:null,embedded_audio:false,
    embedded_weights:false,external_paths:false,authenticated:false,parent_verified:false};
}
function opened():OpenProject{return {document,snapshot:snapshot()};}
describe('native project document validation',()=>{
  it('preserves raw bytes rather than serializing parsed JSON',()=>{
    const raw=' { "format": "'+PROJECT_FORMAT+'", "project_sha256": "'+id+'", "value": 0.0 } ';
    expect(declaredProjectId(raw)).toBe(id);
    expect(checkedDocument(raw,1024)).toBe(raw);
    expect(()=>checkedDocument('éé',3)).toThrow('byte limit');
  });
  it('validates printable normalized project titles',()=>{
    expect(validTitle('संगीत · Étude')).toBe(true);
    expect(validTitle('e\u0301')).toBe(false);
    expect(validTitle(' Title')).toBe(false);
    expect(validTitle('line\nbreak')).toBe(false);
    expect(validTitle('x'.repeat(81))).toBe(false);
  });
  it('accepts long projects with explicit preview restrictions',()=>{
    expect(parseProjectSnapshot(snapshot(),id)).toEqual(snapshot());
    expect(()=>parseProjectSnapshot({...snapshot(),preview_supported:true},id)).toThrow();
    expect(()=>parseProjectSnapshot({...snapshot(),revision:1},id)).toThrow();
    expect(()=>parseProjectSnapshot({...snapshot(),embedded_weights:true},id)).toThrow();
  });
  it('rejects inconsistent automation metadata',()=>{
    const s=snapshot();s.automation_lanes[0].keys[1].tick=0;
    expect(()=>parseProjectSnapshot(s,id)).toThrow('key');
    expect(()=>parseProjectSnapshot({...snapshot(),automation_keys:3},id)).toThrow('count');
    expect(()=>parseProjectSnapshot({...snapshot(),project_sha256:'0'.repeat(64)},id)).toThrow();
  });
});
describe('project creation and revision requests',()=>{
  it('pins the exact inspected source strings',()=>{
    const source=parseScoreSource('{"sha256":"'+score+'","value":0.0}',
      {score_sha256:score,frames:192000,bars:4,events:10,preview_supported:true});
    const mix=parseAutomationSource('{"value":1.0}',{automation_sha256:automation,lanes:1,keys:2});
    const input=createInput('Original music',source,mix);
    expect(input.score_document).toBe(source.document);
    expect(input.automation_document).toBe(mix.document);
    expect(input.expected_score_sha256).toBe(score);
    expect(input.expected_automation_sha256).toBe(automation);
  });
  it('does not silently create revisions with no changes',()=>{
    expect(()=>revisionInput(opened(),'Original music',null,defaultMusicMix())).toThrow('No project changes');
    const revised=revisionInput(opened(),'New name',null,defaultMusicMix());
    expect(revised).toEqual({project_document:document,expected_project_sha256:id,title:'New name'});
  });
  it('includes only changed mixer settings',()=>{
    const mix={...defaultMusicMix(),room:.2};
    const input=revisionInput(opened(),'Original music',null,mix);
    expect(input.mix).toEqual(mix);
    expect(input.title).toBeUndefined();
    expect(input.project_document).toBe(document);
  });
  it('rejects note edits for long or different source scores',()=>{
    const edits={base_score_sha256:score,remove:[0],replace:[],add:[]};
    expect(()=>revisionInput(opened(),'Original music',edits,defaultMusicMix())).toThrow('inspected project');
    expect(()=>revisionInput({...opened(),document:JSON.stringify({format:PROJECT_FORMAT,project_sha256:'0'.repeat(64)})},
      'New',null,defaultMusicMix())).toThrow('identity');
  });
});
