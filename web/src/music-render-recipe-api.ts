/** Same-origin bounded integrated recipe requests and verified audio downloads. */
import { ApiError, isClientJobId, validateJob, validateAudioPath, withJobDeadline,
  isRejectedJobSubmission, type AudioJob, type JobAudio } from './api';
import { isDigest } from './music-contract';
import { checkedDocument } from './music-project-contract';
import { responseBytes, jsonResponse, checksum, downloadBytes, fail } from './music-project-api';
import { verifyStemWav } from './music-stem-contract';
import { declaredRecipe, parseRecipeSnapshot, parseRecipeResult, validateCreatedRecipe,
  MAX_RECIPE_BYTES, MAX_CREATE_BYTES, type OpenRecipe, type CreateRecipeInput } from './music-render-recipe-contract';

async function post(path:string,body:string,signal:AbortSignal,pin?:string,job?:string){
  const headers:Record<string,string>={'Content-Type':'application/json'};
  if(pin!==undefined){
    if(!isDigest(pin))throw new Error('Invalid render recipe digest.');
    headers['X-Sonumio-Recipe-SHA256']=pin;
  }
  if(job!==undefined){
    if(!isClientJobId(job))throw new Error('Invalid render job identifier.');
    headers['X-Sonumio-Job-Id']=job;
  }
  const response=await fetch(path,{method:'POST',body,headers,signal,
    credentials:'omit',redirect:'error',cache:'no-store'});
  if(!response.ok)await fail(response,signal);
  return response;
}
export async function inspectRenderRecipe(document:string,signal?:AbortSignal):Promise<OpenRecipe>{
  const declared=declaredRecipe(document);
  return withJobDeadline(signal,async pending=>{
    const response=await post('/v1/music/render-recipe/inspect',document,pending,declared.recipe_sha256);
    const snapshot=parseRecipeSnapshot(await jsonResponse(response,128*1024,pending),document);
    pending.throwIfAborted();
    return {document,snapshot};
  });
}
export async function createRenderRecipe(input:CreateRecipeInput,signal?:AbortSignal):Promise<OpenRecipe>{
  const body=checkedDocument(JSON.stringify(input),MAX_CREATE_BYTES);
  return withJobDeadline(signal,async pending=>{
    const response=await post('/v1/music/render-recipe/create',body,pending);
    const id=response.headers.get('X-Sonumio-Recipe-SHA256');
    const digest=response.headers.get('X-Sonumio-Content-SHA256');
    if(!isDigest(id)||!isDigest(digest)||
      response.headers.get('X-Sonumio-Project-SHA256')!==input.expected_project_sha256)
      throw new Error('Invalid render recipe download receipt.');
    const bytes=await downloadBytes(response,'application/json',MAX_RECIPE_BYTES,pending);
    const document=new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes);
    if(declaredRecipe(document).recipe_sha256!==id)throw new Error('Recipe identity differs from download.');
    const result=await inspectRenderRecipe(document,pending);
    if(result.snapshot.content_sha256!==digest)throw new Error('Recipe content checksum differs from inspection.');
    validateCreatedRecipe(input,result);
    return result;
  });
}
export async function submitRecipeJob(recipe:OpenRecipe,id:string,signal?:AbortSignal):Promise<AudioJob>{
  if(!recipe.snapshot.preview_supported)throw new Error('Recipe exceeds browser preview limits.');
  if(declaredRecipe(recipe.document).recipe_sha256!==recipe.snapshot.recipe_sha256)
    throw new Error('Recipe identity differs from inspection.');
  try{
    return await withJobDeadline(signal,async pending=>{
      const response=await post('/v1/jobs/music-recipe',recipe.document,pending,recipe.snapshot.recipe_sha256,id);
      const job=validateJob(await jsonResponse(response,128*1024,pending) as AudioJob,id);
      if(job.kind!=='music')throw new Error('Unexpected render recipe job kind.');
      return job;
    });
  }catch(error){
    if(signal?.aborted||isRejectedJobSubmission(error))throw error;
    throw new Error('Recipe render submission could not be confirmed; check the saved job reference.',{cause:error});
  }
}
export async function downloadRecipeWav(audio:JobAudio,recipe:OpenRecipe,signal?:AbortSignal):Promise<Blob>{
  const report=parseRecipeResult(audio,recipe.snapshot);
  return withJobDeadline(signal,async pending=>{
    const response=await fetch(validateAudioPath(audio.audio_url),{signal:pending,
      credentials:'omit',redirect:'error',cache:'no-store'});
    if(!response.ok)throw new ApiError('Recipe audio unavailable (HTTP '+response.status+')',response.status);
    const size=44+4*report.frames;
    if(response.headers.get('content-length')!==String(size))
      throw new Error('Unexpected recipe audio length.');
    const bytes=await responseBytes(response,size,pending);
    await verifyStemWav(bytes,{frames:report.frames},report.wav_sha256);
    pending.throwIfAborted();
    return new Blob([bytes],{type:'audio/wav'});
  });
}
export async function canonicalRecipeBlob(recipe:OpenRecipe):Promise<Blob>{
  const bytes=new TextEncoder().encode(recipe.document);
  if(await checksum(bytes)!==recipe.snapshot.content_sha256)
    throw new Error('Opened recipe is not canonical; use the offline CLI to normalize it before saving.');
  return new Blob([bytes],{type:'application/json'});
}
