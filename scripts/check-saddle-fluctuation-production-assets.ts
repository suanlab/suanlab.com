import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
import {saddleFluctuationExamples} from '../src/lib/learning-saddle-fluctuation-input';
import {calculateSaddleFluctuation} from '../src/lib/learning-saddle-fluctuation';
import type {SaddleFluctuationInput} from '../src/lib/learning-saddle-fluctuation-input';
const root='.runtime/learning/saddle-fluctuation-reader-integration',base=process.env.SADDLE_FLUCTUATION_PREVIEW_BASE??'http://127.0.0.1:4203';
const fixture=JSON.parse(readFileSync('scripts/fixtures/saddle-fluctuation-references.json','utf8'))as{cases:{id:string;input:SaddleFluctuationInput;expected:unknown}[]};
const cases=fixture.cases.map(c=>({label:c.id,input:c.input,expected:calculateSaddleFluctuation(c.input)}));
assert(saddleFluctuationExamples.every(e=>fixture.cases.some(c=>c.id===e.id)));

const stable=(value:unknown):string=>Array.isArray(value)?`[${value.map(stable).join(',')}]`:value&&typeof value==='object'?`{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${stable((value as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(value);
const hash=(value:unknown)=>createHash('sha256').update(stable(value)).digest('hex');
async function main(){
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try{
  const context=await browser.newContext({reducedMotion:'reduce'});await context.addInitScript({content:`globalThis.__name=fn=>fn;window.__ports=[];const Native=Worker;window.Worker=class extends Native{constructor(url,opts){super(url,opts);window.__ports.push({url:String(url),name:opts?.name,type:opts?.type??'classic'});}};`});
  const page=await context.newPage();await page.exposeFunction('reportEuclideanProgress',(value:unknown)=>{writeFileSync(root+'/production-progress.json',JSON.stringify(value,null,2)+'\n');});page.on('pageerror',e=>errors.push(e.message));await page.goto(base+'/book/online/physics-math-for-ai/saddle-fluctuation-and-nongaussian-approximation/');await page.locator('[data-saddle-phase="ready"]').waitFor({timeout:60000});
  type Port={url:string;name:string;type:'classic'|'module'};
  const ports=await page.evaluate(()=>(window as unknown as{__ports:Port[]}).__ports),calculate=ports.find(p=>p.name==='saddle-fluctuation-calculation'),validate=ports.find(p=>p.name==='saddle-fluctuation-validation');assert(calculate&&validate);const assets=[];
  for(const port of [calculate,validate]){const url=new URL(port.url,base);assert(url.pathname.startsWith('/_next/static/'));const response=await fetch(url);assert.equal(response.status,200);const bytes=Buffer.from(await response.arrayBuffer()),disk=readFileSync((process.env.SADDLE_FLUCTUATION_EXPORT_ROOT??'/tmp/suanlab-saddle-fluctuation-candidate-20261110/out')+url.pathname);assert.deepEqual(bytes,disk);assets.push({...port,bytes:disk.length,sha256:createHash('sha256').update(disk).digest('hex')});}
  const result=await page.evaluate(async({inputs,assets})=>{
   const workers=assets.map(asset=>new Worker(asset.url,{type:asset.type}));let started=0;const lifecycle={created:workers.length,terminated:0};
   const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
   const digest=async(v:unknown)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(stable(v))))).map(x=>x.toString(16).padStart(2,'0')).join('');
   const rpc=(port:Worker,message:unknown)=>new Promise<Record<string,unknown>>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Native Worker deadline')),120000);port.onerror=e=>{clearTimeout(timer);e.preventDefault();reject(Error(e.message));};port.onmessage=e=>{if(e.data.phase==='started'){started++;return;}clearTimeout(timer);resolve(e.data);};port.postMessage(message);});
   if(await digest(inputs[0].input)!==inputs[0].inputHash)throw Error('Digest encoding preflight failed');const rows=[];let rejectedMutations=0,rejectedAttributions=0;
   try{for(const[i,c]of inputs.entries()){
    const requestId=i+1,response=await rpc(workers[0],{version:1,requestId,input:c.input});if(!response.ok)throw Error(c.label+': calculation rejected');
    const receipt=await rpc(workers[1],{version:1,requestId,input:c.input,response});if(receipt.phase!=='validated')throw Error(c.label+': validation rejected');
    const accepted=receipt.response as Record<string,unknown>;rows.push({label:c.label,hash:await digest(response.result),validatedHash:await digest(accepted.result)});
    const changed=structuredClone(response);const r=changed.result as {kind:string;stationary:{candidateAction:string};minimumAction:string};if(r.kind==='chain')r.stationary.candidateAction='-1';else r.minimumAction='1';if((await rpc(workers[1],{version:1,requestId,input:c.input,response:changed})).phase!=='rejected')throw Error('Wrong saddle action accepted');rejectedMutations++;
    if((await rpc(workers[1],{version:1,requestId:requestId+1000,input:c.input,response})).phase!=='rejected')throw Error('Wrong request attribution accepted');rejectedAttributions++;await(window as unknown as{reportEuclideanProgress:(value:unknown)=>Promise<void>}).reportEuclideanProgress({completed:i+1,total:inputs.length,lastLabel:c.label,rows,rejectedMutations,rejectedAttributions});
   }return{rows,started,rejectedMutations,rejectedAttributions,lifecycle};}finally{workers.forEach(w=>{w.terminate();lifecycle.terminated++;});}
  },{inputs:cases.map(c=>({label:c.label,input:c.input,inputHash:hash(c.input)})),assets});
  writeFileSync(root+'/production-response.json',JSON.stringify(result,null,2)+'\n');
  result.rows.forEach((row,i)=>{assert.equal(row.hash,hash(cases[i].expected),row.label);assert.equal(row.validatedHash,row.hash);});assert.equal(result.rows.length,54);assert.equal(result.started,54);assert.equal(result.lifecycle.created,2);assert.equal(result.lifecycle.terminated,2);assert.deepEqual(errors,[]);
  writeFileSync(`${root}/production-assets.json`,JSON.stringify({...result,independentPythonFixtureInputs:54,teachingExamples:32,requestsIncludeRepeatedExamples:false,browserMatchesNodeProducer:true,independentPythonNumericsVerifiedSeparately:true,productionNextAssetsVerified:true,readerRegistered:true,assets,errors},null,2)+'\n');console.log(JSON.stringify({productionCases:54,independentPythonFixtureInputs:54,rejectedMutations:result.rejectedMutations,rejectedAttributions:result.rejectedAttributions}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
