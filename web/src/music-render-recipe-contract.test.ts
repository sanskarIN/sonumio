import { describe,expect,it } from 'vitest';
import { MUSIC_ENGINE } from './music-contract';
import { FORMAT,declaredRecipe,createRecipeInput,parseRecipeSnapshot,parseRecipeResult,validateCreatedRecipe,
  type RecipeSnapshot,type RecipeIdentity } from './music-render-recipe-contract';

const id='a'.repeat(64),project='b'.repeat(64),score='c'.repeat(64),automation='d'.repeat(64);
const performance='e'.repeat(64),rack='f'.repeat(64);
const identity:RecipeIdentity={project_sha256:project,score_sha256:score,automation_sha256:automation,
  performance_sha256:performance,rack_sha256:rack};
const document=JSON.stringify({format:FORMAT,recipe_sha256:id,identity});
function snapshot():RecipeSnapshot{
  return {...identity,format:FORMAT,recipe_sha256:id,content_sha256:'1'.repeat(64),
    title:'Recipe',engine:'performance-rack',frames:192000,sample_rate:24000,channels:2,bars:4,revision:0,
    preview_supported:true,performance:{performance_sha256:performance,score_sha256:score,frames:192000},
    rack:{rack_sha256:rack,automation_sha256:automation,score_sha256:score,frames:192000},
    rendered:false,learned_model:false,external_recordings_used:false,authenticated:false};
}
function result(){
  return {...identity,recipe_sha256:id,receipt_sha256:'2'.repeat(64),wav_sha256:'3'.repeat(64),
    recipe_engine:'performance-rack',frames:192000,sample_rate:24000,channels:2,duration_seconds:8,
    master_attenuation:1,engine_id:MUSIC_ENGINE,model_id:MUSIC_ENGINE,
    learned_model:false,external_recordings_used:false};
}
describe('integrated recipe contracts',()=>{
  it('keeps raw source document strings and pins project identity',()=>{
    const p=' {"format":"sonumio-music-project-v1","project_sha256":"'+project+'","value":0.0} ';
    const e='{"format":"sonumio-music-performance-v1","value":0.0}';
    const r='{"format":"sonumio-music-processing-rack-v1","value":1.0}';
    const input=createRecipeInput(p,e,r);
    expect(input.project_document).toBe(p);
    expect(input.expected_project_sha256).toBe(project);
    expect(input.performance_document).toBe(e);
    expect(input.rack_document).toBe(r);
    validateCreatedRecipe(input,{document,snapshot:snapshot()});
    expect(()=>validateCreatedRecipe(createRecipeInput(p),{document,snapshot:snapshot()})).toThrow();
    expect(()=>createRecipeInput(p,'{}')).toThrow();
  });
  it('validates all declared component identities',()=>{
    expect(declaredRecipe(document)).toEqual({recipe_sha256:id,identity});
    expect(parseRecipeSnapshot(snapshot(),document)).toEqual(snapshot());
    expect(()=>parseRecipeSnapshot({...snapshot(),performance_sha256:null},document)).toThrow();
    expect(()=>parseRecipeSnapshot({...snapshot(),rack:{...snapshot().rack,automation_sha256:'0'.repeat(64)}},document)).toThrow();
    expect(()=>parseRecipeSnapshot({...snapshot(),preview_supported:false},document)).toThrow();
    expect(()=>parseRecipeSnapshot({...snapshot(),engine:'project'},document)).toThrow();
  });
  it('validates queue result against every recipe identity and frame count',()=>{
    expect(parseRecipeResult(result(),snapshot())).toEqual(result());
    for(const patch of [{recipe_sha256:'0'.repeat(64)},{score_sha256:'0'.repeat(64)},
      {performance_sha256:null},{recipe_engine:'project'},{frames:2},{channels:1},
      {receipt_sha256:'bad'},{wav_sha256:'bad'},{master_attenuation:2},{learned_model:true}])
      expect(()=>parseRecipeResult({...result(),...patch},snapshot())).toThrow();
  });
});
