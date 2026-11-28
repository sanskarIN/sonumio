/** Integrated recipe identity and preview receipts; preserve raw UTF-8 bytes. */
import { isDigest, MUSIC_ENGINE, SAMPLE_RATE } from './music-contract';
import { checkedDocument, declaredProjectId, MAX_PROJECT_BYTES } from './music-project-contract';

export const FORMAT='sonumio-project-render-recipe-v1';
export const MAX_RECIPE_BYTES=3*1024*1024;
export const MAX_PERFORMANCE_BYTES=512*1024;
export const MAX_RACK_BYTES=256*1024;
export const MAX_CREATE_BYTES=6*1024*1024;

export type RecipeEngine='project'|'performance'|'rack'|'performance-rack';
export type RecipeIdentity={
  project_sha256:string;score_sha256:string;automation_sha256:string;
  performance_sha256:string|null;rack_sha256:string|null;
};
export type RecipeSnapshot=RecipeIdentity&{
  format:typeof FORMAT;recipe_sha256:string;content_sha256:string;
  title:string;engine:RecipeEngine;frames:number;sample_rate:number;channels:2;
  bars:number;revision:number;preview_supported:boolean;
  performance:Record<string,unknown>|null;rack:Record<string,unknown>|null;
  rendered:false;learned_model:false;external_recordings_used:false;authenticated:false;
};
export type OpenRecipe={document:string;snapshot:RecipeSnapshot};
export type CreateRecipeInput={
  project_document:string;expected_project_sha256:string;
  performance_document?:string;rack_document?:string;
};
export type RecipeResult=RecipeIdentity&{
  recipe_sha256:string;receipt_sha256:string;wav_sha256:string;recipe_engine:RecipeEngine;
  frames:number;sample_rate:number;channels:2;duration_seconds:number;master_attenuation:number;
  engine_id:string;model_id:string;learned_model:false;external_recordings_used:false;
};

const record=(value:unknown):value is Record<string,unknown>=>
  typeof value==='object'&&value!==null&&!Array.isArray(value);
const integer=(value:unknown,lo:number,hi:number):value is number=>
  typeof value==='number'&&Number.isInteger(value)&&value>=lo&&value<=hi;
const IDs=['project_sha256','score_sha256','automation_sha256','performance_sha256','rack_sha256'] as const;
function identity(raw:unknown):RecipeIdentity{
  if(!record(raw)||!isDigest(raw.project_sha256)||!isDigest(raw.score_sha256)||
    !isDigest(raw.automation_sha256)||
    (raw.performance_sha256!==null&&!isDigest(raw.performance_sha256))||
    (raw.rack_sha256!==null&&!isDigest(raw.rack_sha256)))
    throw new Error('Invalid render recipe component identities.');
  return Object.fromEntries(IDs.map(key=>[key,raw[key]])) as RecipeIdentity;
}
export function engineFor(value:RecipeIdentity):RecipeEngine{
  return value.rack_sha256!==null?(value.performance_sha256!==null?'performance-rack':'rack'):
    value.performance_sha256!==null?'performance':'project';
}
function sameIdentity(a:RecipeIdentity,b:RecipeIdentity){
  return IDs.every(key=>a[key]===b[key]);
}
function parsed(document:string,max:number):Record<string,unknown>{
  checkedDocument(document,max);
  let raw:unknown;
  try{raw=JSON.parse(document);}catch{throw new Error('Invalid JSON document.');}
  if(!record(raw))throw new Error('Expected a JSON object.');
  return raw;
}
export function declaredRecipe(document:string):{recipe_sha256:string;identity:RecipeIdentity}{
  const raw=parsed(document,MAX_RECIPE_BYTES);
  if(raw.format!==FORMAT||!isDigest(raw.recipe_sha256))
    throw new Error('Unsupported render recipe format or missing digest.');
  return {recipe_sha256:raw.recipe_sha256,identity:identity(raw.identity)};
}
export function createRecipeInput(project:string,performance?:string|null,rack?:string|null):CreateRecipeInput{
  checkedDocument(project,MAX_PROJECT_BYTES);
  const input:CreateRecipeInput={project_document:project,expected_project_sha256:declaredProjectId(project)};
  if(performance!==undefined&&performance!==null){
    const raw=parsed(performance,MAX_PERFORMANCE_BYTES);
    if(raw.format!=='sonumio-music-performance-v1')throw new Error('Unsupported performance document.');
    input.performance_document=performance;
  }
  if(rack!==undefined&&rack!==null){
    const raw=parsed(rack,MAX_RACK_BYTES);
    if(raw.format!=='sonumio-music-processing-rack-v1')throw new Error('Unsupported rack document.');
    input.rack_document=rack;
  }
  checkedDocument(JSON.stringify(input),MAX_CREATE_BYTES);
  return input;
}
export function parseRecipeSnapshot(raw:unknown,document:string):RecipeSnapshot{
  const declared=declaredRecipe(document);
  if(!record(raw)||raw.format!==FORMAT||raw.recipe_sha256!==declared.recipe_sha256||
    !sameIdentity(identity(raw),declared.identity)||raw.engine!==engineFor(declared.identity)||
    !isDigest(raw.content_sha256)||typeof raw.title!=='string'||raw.title.length<1||
    Array.from(raw.title).length>80||!integer(raw.frames,1,2880000)||
    !integer(raw.bars,4,32)||!integer(raw.revision,0,2147483647)||
    raw.sample_rate!==SAMPLE_RATE||raw.channels!==2||
    raw.preview_supported!==(raw.frames<=720000&&raw.bars<=16)||
    raw.rendered!==false||raw.learned_model!==false||
    raw.external_recordings_used!==false||raw.authenticated!==false)
    throw new Error('Render recipe inspection does not match the document.');
  if(declared.identity.performance_sha256===null){
    if(raw.performance!==null)throw new Error('Unexpected performance description.');
  }else if(!record(raw.performance)||raw.performance.performance_sha256!==declared.identity.performance_sha256||
    raw.performance.score_sha256!==declared.identity.score_sha256||raw.performance.frames!==raw.frames)
    throw new Error('Performance description differs from recipe.');
  if(declared.identity.rack_sha256===null){
    if(raw.rack!==null)throw new Error('Unexpected rack description.');
  }else if(!record(raw.rack)||raw.rack.rack_sha256!==declared.identity.rack_sha256||
    raw.rack.automation_sha256!==declared.identity.automation_sha256||raw.rack.frames!==raw.frames||
    raw.rack.score_sha256!==declared.identity.score_sha256)
    throw new Error('Rack description differs from recipe.');
  return raw as RecipeSnapshot;
}
export function validateCreatedRecipe(input:CreateRecipeInput,recipe:OpenRecipe):void{
  const snapshot=recipe.snapshot;
  if(snapshot.project_sha256!==input.expected_project_sha256||
    (snapshot.performance_sha256!==null)!==(input.performance_document!==undefined)||
    (snapshot.rack_sha256!==null)!==(input.rack_document!==undefined))
    throw new Error('Created render recipe differs from selected components.');
}
export function parseRecipeResult(raw:unknown,expected:RecipeSnapshot):RecipeResult{
  if(!record(raw)||raw.recipe_sha256!==expected.recipe_sha256||
    !sameIdentity(identity(raw),expected)||raw.recipe_engine!==expected.engine||
    raw.frames!==expected.frames||raw.sample_rate!==SAMPLE_RATE||raw.channels!==2||
    raw.duration_seconds!==expected.frames/SAMPLE_RATE||
    raw.engine_id!==MUSIC_ENGINE||raw.model_id!==MUSIC_ENGINE||
    !isDigest(raw.receipt_sha256)||!isDigest(raw.wav_sha256)||
    typeof raw.master_attenuation!=='number'||!Number.isFinite(raw.master_attenuation)||
    raw.master_attenuation<=0||raw.master_attenuation>1||
    raw.learned_model!==false||raw.external_recordings_used!==false)
    throw new Error('Render result differs from the inspected recipe.');
  return raw as RecipeResult;
}
export type RecipeReference={id:string;recipe:string};
