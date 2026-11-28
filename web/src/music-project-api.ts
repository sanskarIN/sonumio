/** Same-origin, bounded project transport. The raw UTF-8 source document
 * is preserved across all requests, including revision JSON envelopes. */
import { ApiError, isClientJobId, isRejectedJobSubmission, validateJob, withJobDeadline,
  type AudioJob } from './api';
import { isDigest } from './music-contract';
import { checkedDocument, declaredProjectId, parseProjectSnapshot, parseScoreSource,
  parseAutomationSource, MAX_PROJECT_BYTES, MAX_SCORE_BYTES, MAX_AUTOMATION_BYTES,
  type OpenProject, type ProjectCreate, type ProjectRevision, type SourceScore, type SourceAutomation,
} from './music-project-contract';

const MAX_MUTATION_BYTES=5*1024*1024;
const MAX_SNAPSHOT_BYTES=2*1024*1024;
type LocalOptions={signal?:AbortSignal;pin?:string;jobId?:string};
type ProjectDownload={document:string;id:string;score:string;automation:string;checksum:string};

const decode=(bytes:Uint8Array)=>new TextDecoder('utf-8',{fatal:true,ignoreBOM:true}).decode(bytes);

async function responseBytes(response:Response,limit:number,signal:AbortSignal):Promise<Uint8Array<ArrayBuffer>> {
  const declared=response.headers.get('content-length');
  const size=declared===null?null:Number(declared);
  if(size!==null&&(!/^[0-9]+$/.test(declared!)||!Number.isInteger(size)||size<1||size>limit))
    throw new Error('Invalid project response length.');
  if(!response.body)throw new Error('Project response is missing its body.');
  const reader=response.body.getReader(), chunks:Uint8Array[]=[];
  let count=0;
  try {
    while(true){
      signal.throwIfAborted();
      const {value,done}=await reader.read();
      if(done)break;
      if(count+value.byteLength>limit||(size!==null&&count+value.byteLength>size))
        throw new Error('Project response exceeds its byte budget.');
      chunks.push(value);count+=value.byteLength;
    }
  }finally{
    try{await reader.cancel();}catch{/* already closed */}
    reader.releaseLock();
  }
  if(count===0||(size!==null&&count!==size))throw new Error('Truncated project response.');
  const buffer=new Uint8Array(count);
  let offset=0;
  for(const chunk of chunks){buffer.set(chunk,offset);offset+=chunk.byteLength;}
  return buffer;
}

async function fail(response:Response,signal:AbortSignal):Promise<never>{
  let detail='Project request failed (HTTP '+response.status+').';
  try{
    const raw:unknown=JSON.parse(decode(await responseBytes(response,8192,signal)));
    if(raw&&typeof raw==='object'&&'detail'in raw&&typeof raw.detail==='string')
      detail=raw.detail;
  }catch{/* preserve status if server error body is invalid or oversized */}
  throw new ApiError(detail,response.status);
}

async function post(path:string,body:string,signal:AbortSignal,options:LocalOptions={}):Promise<Response>{
  const headers:Record<string,string>={'Content-Type':'application/json'};
  if(options.pin!==undefined){
    if(!isDigest(options.pin))throw new Error('Invalid project pin.');
    headers['X-Sonumio-Project-SHA256']=options.pin;
  }
  if(options.jobId!==undefined){
    if(!isClientJobId(options.jobId))throw new Error('Invalid project job identifier.');
    headers['X-Sonumio-Job-Id']=options.jobId;
  }
  const response=await fetch(path,{method:'POST',body,headers,signal,
    credentials:'omit',cache:'no-store',redirect:'error'});
  if(!response.ok)await fail(response,signal);
  return response;
}

async function jsonResponse(response:Response,limit:number,signal:AbortSignal):Promise<unknown>{
  if(response.headers.get('content-type')?.split(';')[0]!=='application/json')
    throw new Error('Unexpected project response type.');
  return JSON.parse(decode(await responseBytes(response,limit,signal)));
}
async function checksum(bytes:Uint8Array<ArrayBuffer>):Promise<string>{
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))]
    .map(x=>x.toString(16).padStart(2,'0')).join('');
}
async function downloadBytes(response:Response,media:string,limit:number,signal:AbortSignal):Promise<Uint8Array<ArrayBuffer>>{
  const digest=response.headers.get('X-Sonumio-Content-SHA256');
  if(response.headers.get('content-type')?.split(';')[0]!==media||
    !isDigest(digest)||response.headers.get('content-length')===null)
    throw new Error('Invalid project download receipt.');
  const bytes=await responseBytes(response,limit,signal);
  if(await checksum(bytes)!==digest)throw new Error('Project download checksum mismatch.');
  signal.throwIfAborted();
  return bytes;
}

export async function inspectProjectScore(document:string,signal?:AbortSignal):Promise<SourceScore>{
  checkedDocument(document,MAX_SCORE_BYTES);
  return withJobDeadline(signal,async pending=>{
    const response=await post('/v1/music/project/inspect-score',document,pending);
    const receipt=parseScoreSource(document,await jsonResponse(response,8192,pending));
    let raw:unknown;
    try{raw=JSON.parse(document);}catch{throw new Error('Invalid score JSON.');}
    if(!raw||typeof raw!=='object'||!('sha256'in raw)||raw.sha256!==receipt.score_sha256)
      throw new Error('Score inspection identity differs from the file.');
    return receipt;
  });
}
export async function inspectProjectAutomation(document:string,signal?:AbortSignal):Promise<SourceAutomation>{
  checkedDocument(document,MAX_AUTOMATION_BYTES);
  return withJobDeadline(signal,async pending=>{
    const response=await post('/v1/music/project/inspect-automation',document,pending);
    return parseAutomationSource(document,await jsonResponse(response,8192,pending));
  });
}

export async function inspectNativeProject(document:string,signal?:AbortSignal):Promise<OpenProject>{
  const id=declaredProjectId(document);
  return withJobDeadline(signal,async pending=>{
    const response=await post('/v1/music/project/inspect',document,pending,{pin:id});
    return {document,snapshot:parseProjectSnapshot(await jsonResponse(response,MAX_SNAPSHOT_BYTES,pending),id)};
  });
}

async function mutationDownload(path:string,input:ProjectCreate|ProjectRevision,signal:AbortSignal):Promise<ProjectDownload>{
  const body=checkedDocument(JSON.stringify(input),MAX_MUTATION_BYTES);
  const response=await post(path,body,signal);
  const id=response.headers.get('X-Sonumio-Project-SHA256');
  const score=response.headers.get('X-Sonumio-Score-SHA256');
  const automation=response.headers.get('X-Sonumio-Automation-SHA256');
  const digest=response.headers.get('X-Sonumio-Content-SHA256');
  if(!isDigest(id)||!isDigest(score)||!isDigest(automation)||!isDigest(digest))
    throw new Error('Project download omitted verified content identities.');
  const bytes=await downloadBytes(response,'application/json',MAX_PROJECT_BYTES,signal);
  const document=decode(bytes);
  if(declaredProjectId(document)!==id)throw new Error('Project document and download identity differ.');
  return {document,id,score,automation,checksum:digest};
}

async function inspectDownloaded(download:ProjectDownload,signal:AbortSignal):Promise<OpenProject>{
  const project=await inspectNativeProject(download.document,signal);
  if(project.snapshot.project_sha256!==download.id||project.snapshot.score_sha256!==download.score||
    project.snapshot.automation_sha256!==download.automation||project.snapshot.content_sha256!==download.checksum)
    throw new Error('Downloaded project and verified inspection receipt differ.');
  return project;
}

export async function createNativeProject(input:ProjectCreate,signal?:AbortSignal):Promise<OpenProject>{
  return withJobDeadline(signal,async pending=>{
    const download=await mutationDownload('/v1/music/project/create',input,pending);
    if(download.score!==input.expected_score_sha256||
      (input.expected_automation_sha256!==undefined&&download.automation!==input.expected_automation_sha256))
      throw new Error('Created project differs from pinned sources.');
    const project=await inspectDownloaded(download,pending);
    if(project.snapshot.title!==input.title||project.snapshot.revision!==0||
      project.snapshot.parent_project_sha256!==null)
      throw new Error('Created project metadata differs.');
    return project;
  });
}

export async function reviseNativeProject(parent:OpenProject,input:ProjectRevision,signal?:AbortSignal):Promise<OpenProject>{
  if(input.expected_project_sha256!==parent.snapshot.project_sha256||input.project_document!==parent.document)
    throw new Error('Revision does not address the inspected project.');
  return withJobDeadline(signal,async pending=>{
    const download=await mutationDownload('/v1/music/project/revise',input,pending);
    const project=await inspectDownloaded(download,pending);
    const snapshot=project.snapshot, previous=parent.snapshot;
    if(snapshot.parent_project_sha256!==previous.project_sha256||snapshot.revision!==previous.revision+1||
      snapshot.title!==(input.title??previous.title)||snapshot.project_sha256===previous.project_sha256||
      (!input.note_edits&&!input.score_document&&snapshot.score_sha256!==previous.score_sha256)||
      (!input.mix&&!input.automation_document&&snapshot.automation_sha256!==previous.automation_sha256)||
      (input.expected_score_sha256!==undefined&&snapshot.score_sha256!==input.expected_score_sha256)||
      (input.expected_automation_sha256!==undefined&&snapshot.automation_sha256!==input.expected_automation_sha256))
      throw new Error('Project revision receipt differs from requested changes.');
    return project;
  });
}

export async function exportNativeProjectScore(project:OpenProject,format:'json'|'midi',signal?:AbortSignal):Promise<Blob>{
  if(format!=='json'&&format!=='midi')throw new Error('Unsupported project export format.');
  const id=declaredProjectId(project.document);
  if(id!==project.snapshot.project_sha256)throw new Error('Project source identity differs.');
  return withJobDeadline(signal,async pending=>{
    const response=await post('/v1/music/project/'+(format==='json'?'score':'midi'),project.document,pending,{pin:id});
    if(response.headers.get('X-Sonumio-Project-SHA256')!==id||
      response.headers.get('X-Sonumio-Score-SHA256')!==project.snapshot.score_sha256)
      throw new Error('Project score export receipt differs.');
    const media=format==='json'?'application/json':'audio/midi';
    return new Blob([await downloadBytes(response,media,MAX_SCORE_BYTES,pending)],{type:media});
  });
}

export async function submitNativeProjectJob(project:OpenProject,id:string,signal?:AbortSignal):Promise<AudioJob>{
  if(!project.snapshot.preview_supported)throw new Error('Project exceeds studio preview duration.');
  if(!isClientJobId(id))throw new Error('Invalid client job identifier.');
  if(declaredProjectId(project.document)!==project.snapshot.project_sha256)throw new Error('Project identity differs.');
  try{
    return await withJobDeadline(signal,async pending=>{
      const response=await post('/v1/jobs/music-project',project.document,pending,
        {pin:project.snapshot.project_sha256,jobId:id});
      const job=validateJob(await jsonResponse(response,64*1024,pending) as AudioJob,id);
      if(job.kind!=='music')throw new Error('Unexpected project render job kind.');
      return job;
    });
  }catch(error){
    if(signal?.aborted||isRejectedJobSubmission(error))throw error;
    throw new Error('Project render submission could not be confirmed. Use Check last project render.',{cause:error});
  }
}
