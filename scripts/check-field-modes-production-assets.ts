import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
import {fieldModesExamples} from '../src/lib/learning-field-modes-examples';
import {calculateFieldModes} from '../src/lib/learning-field-modes';
import type {FieldModesInput} from '../src/lib/learning-field-modes-input';
const root='.runtime/learning/field-modes-reader-integration',base=process.env.FIELD_MODES_PREVIEW_BASE??'http://127.0.0.1:4179';
const fixture=JSON.parse(readFileSync('scripts/fixtures/field-modes-independent.json','utf8'))as{cases:{label:string;input:FieldModesInput;expected:Record<string,unknown>}[]};
const cases=[...fixture.cases.map(c=>({label:c.label,input:c.input,expected:calculateFieldModes(c.input)})),...fieldModesExamples.map(c=>({label:c.id,input:c.input,expected:calculateFieldModes(c.input)}))];
for(const boundary of ['periodic','dirichlet-zero','neumann-zero']as const){const input={...fieldModesExamples.at(-2)!.input,boundary,length:'1/999999999999999999',speed:'999999999999999999',quadratic:'-999999999999999999',quartic:'999999999999999999',background:'999999999999999999',position:Array.from({length:32},(_,i)=>i%2?'999999999999999999':'-999999999999999999'),velocity:Array.from({length:32},()=> '1/999999999999999999')};cases.push({label:boundary+'/extreme32',input,expected:calculateFieldModes(input)});}

const stable=(value:unknown):string=>Array.isArray(value)?`[${value.map(stable).join(',')}]`:value&&typeof value==='object'?`{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${stable((value as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(value);
const hash=(value:unknown)=>createHash('sha256').update(stable(value)).digest('hex');
async function main(){
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try{
  const context=await browser.newContext();await context.addInitScript({content:`globalThis.__name=fn=>fn;window.__ports=[];const Native=Worker;window.Worker=class extends Native{constructor(url,opts){super(url,opts);window.__ports.push({url:String(url),name:opts?.name,type:opts?.type??'classic'});}};`});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));await page.goto(base+'/book/online/physics-math-for-ai/field-fourier-modes-lattice-continuum/');await page.locator('[data-modes-phase="ready"]').waitFor({timeout:60000});
  type Port={url:string;name:string;type:'classic'|'module'};
  const ports=await page.evaluate(()=>(window as unknown as{__ports:Port[]}).__ports),calculate=ports.find(p=>p.name==='field-modes-calculation'),validate=ports.find(p=>p.name==='field-modes-validation');assert(calculate&&validate);const assets=[];
  for(const port of [calculate,validate]){const url=new URL(port.url,base);assert(url.pathname.startsWith('/_next/static/'));const response=await fetch(url);assert.equal(response.status,200);const bytes=Buffer.from(await response.arrayBuffer()),disk=readFileSync('out'+url.pathname);assert.deepEqual(bytes,disk);assets.push({...port,bytes:disk.length,sha256:createHash('sha256').update(disk).digest('hex')});}
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
    const changed=structuredClone(response);((changed.result as Record<string,unknown>).exact as Record<string,unknown>).fullEnergy='999';if((await rpc(workers[1],{version:1,requestId,input:c.input,response:changed})).phase!=='rejected')throw Error('Wrong energy accepted');rejectedMutations++;
    if((await rpc(workers[1],{version:1,requestId:requestId+1000,input:c.input,response})).phase!=='rejected')throw Error('Wrong request attribution accepted');rejectedAttributions++;
   }return{rows,started,rejectedMutations,rejectedAttributions,workersCreated:2,workersTerminated:2};}finally{workers.forEach(w=>w.terminate());}
  },{inputs:cases.map(c=>({label:c.label,input:c.input})),assets});
  result.rows.forEach((row,i)=>{assert.equal(row.hash,hash(cases[i].expected),row.label);assert.equal(row.validatedHash,row.hash);});assert.equal(result.rows.length,105);assert.equal(result.started,105);assert.deepEqual(errors,[]);
  writeFileSync(`${root}/production-assets.json`,JSON.stringify({...result,independentMatrixCases:87,extremeRationalCases:3,teachingExamples:15,productionNextAssetsVerified:true,readerRegistered:true,assets,errors},null,2)+'\n');console.log(JSON.stringify({productionCases:105,independentMatrixCases:87,rejectedMutations:result.rejectedMutations,rejectedAttributions:result.rejectedAttributions}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
