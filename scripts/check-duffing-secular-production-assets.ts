import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
import {duffingSecularExamples} from '../src/lib/learning-duffing-secular-input';
import {calculateDuffingSecular} from '../src/lib/learning-duffing-secular';
import type {DuffingSecularInput} from '../src/lib/learning-duffing-secular-input';
const root='.runtime/learning/duffing-secular-reader-integration',base=process.env.DUFFING_SECULAR_PREVIEW_BASE??'http://127.0.0.1:4186';
const fixture=JSON.parse(readFileSync('scripts/fixtures/duffing-secular-reference.json','utf8'))as{cases:{id:string;input:DuffingSecularInput}[]};
const cases=[...fixture.cases,...duffingSecularExamples].map(c=>({label:c.id,input:c.input,expected:calculateDuffingSecular(c.input)}));

const stable=(value:unknown):string=>Array.isArray(value)?`[${value.map(stable).join(',')}]`:value&&typeof value==='object'?`{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${stable((value as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(value);
const hash=(value:unknown)=>createHash('sha256').update(stable(value)).digest('hex');
async function main(){
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try{
  const context=await browser.newContext();await context.addInitScript({content:`globalThis.__name=fn=>fn;window.__ports=[];const Native=Worker;window.Worker=class extends Native{constructor(url,opts){super(url,opts);window.__ports.push({url:String(url),name:opts?.name,type:opts?.type??'classic'});}};`});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(base+'/book/online/physics-math-for-ai/duffing-secular-frequency-long-time/');await page.locator('[data-duffing-phase="ready"]').waitFor({timeout:60000});
  type Port={url:string;name:string;type:'classic'|'module'};
  const ports=await page.evaluate(()=>(window as unknown as{__ports:Port[]}).__ports),calculate=ports.find(p=>p.name==='duffing-secular-calculation'),validate=ports.find(p=>p.name==='duffing-secular-validation');assert(calculate&&validate);const assets=[];
  for(const port of [calculate,validate]){const url=new URL(port.url,base);assert(url.pathname.startsWith('/_next/static/'));const response=await fetch(url);assert.equal(response.status,200);const bytes=Buffer.from(await response.arrayBuffer()),disk=readFileSync((process.env.DUFFING_SECULAR_EXPORT_ROOT??'/tmp/suanlab-duffing-secular-candidate-20261110/out')+url.pathname);assert.deepEqual(bytes,disk);assets.push({...port,bytes:disk.length,sha256:createHash('sha256').update(disk).digest('hex')});}
  const result=await page.evaluate(async({inputs,assets})=>{
   const workers=assets.map(asset=>new Worker(asset.url,{type:asset.type}));let started=0;
   const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
   const digest=async(v:unknown)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(stable(v))))).map(x=>x.toString(16).padStart(2,'0')).join('');
   const rpc=(port:Worker,message:unknown)=>new Promise<Record<string,unknown>>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('Native Worker deadline')),30000);port.onerror=e=>{clearTimeout(timer);e.preventDefault();reject(Error(e.message));};port.onmessage=e=>{if(e.data.phase==='started'){started++;return;}clearTimeout(timer);resolve(e.data);};port.postMessage(message);});
   const rows=[];let rejectedMutations=0,rejectedAttributions=0;
   try{for(const[i,c]of inputs.entries()){
    const requestId=i+1,response=await rpc(workers[0],{version:1,requestId,input:c.input});if(!response.ok)throw Error(c.label+': calculation rejected');
    const receipt=await rpc(workers[1],{version:1,requestId,input:c.input,response});if(receipt.phase!=='validated')throw Error(c.label+': validation rejected');
    const accepted=receipt.response as Record<string,unknown>;rows.push({label:c.label,hash:await digest(response.result),validatedHash:await digest(accepted.result)});
    const changed=structuredClone(response);const r=changed.result as {mode:string};r.mode=r.mode==='equilibrium'?'nonlinear-periodic':'equilibrium';if((await rpc(workers[1],{version:1,requestId,input:c.input,response:changed})).phase!=='rejected')throw Error('Wrong oscillator mode accepted');rejectedMutations++;
    if((await rpc(workers[1],{version:1,requestId:requestId+1000,input:c.input,response})).phase!=='rejected')throw Error('Wrong request attribution accepted');rejectedAttributions++;
   }return{rows,started,rejectedMutations,rejectedAttributions,workersCreated:2,workersTerminated:2};}finally{workers.forEach(w=>w.terminate());}
  },{inputs:cases.map(c=>({label:c.label,input:c.input})),assets});
  result.rows.forEach((row,i)=>{assert.equal(row.hash,hash(cases[i].expected),row.label);assert.equal(row.validatedHash,row.hash);});assert.equal(result.rows.length,50);assert.equal(result.started,50);assert.deepEqual(errors,[]);
  writeFileSync(`${root}/production-assets.json`,JSON.stringify({...result,independentPythonFixtureInputs:33,teachingExamples:17,requestsIncludeRepeatedExamples:true,productionNextAssetsVerified:true,readerRegistered:true,assets,errors},null,2)+'\n');console.log(JSON.stringify({productionCases:50,independentPythonFixtureInputs:33,rejectedMutations:result.rejectedMutations,rejectedAttributions:result.rejectedAttributions}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
