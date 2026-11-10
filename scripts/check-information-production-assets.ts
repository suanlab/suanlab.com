import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {chromium} from 'playwright';
import {quantumInformationPresets} from '../src/lib/learning-quantum-information-presets';
import {calculateQuantumInformation} from '../src/lib/learning-quantum-information-calculation';
import {q,zero,one} from '../src/lib/learning-quantum-hilbert-core';
const root='.runtime/learning/quantum-information-reader-integration',base=process.env.INFORMATION_PREVIEW_BASE??'http://127.0.0.1:4176';
const scalar=(x:string)=>({real:[x,'0'],imaginary:['0','0']});
const maximum={kind:'bb84',input:[scalar('1'),scalar('1')],comparison:[scalar('1'),scalar('0')],attackFraction:'1/99999999999999999999999999999999',eveZProbability:'1/2',sampleSize:256};
const cases=[...Object.values(quantumInformationPresets).flat().map(p=>({label:p.input.kind+'/'+p.id,input:p.input})),{label:'maximum-bb84',input:maximum},{label:'maximum-source',input:{kind:'entropy',correlationWeight:'1',coherence:'1/3',probabilityOne:'1/3',copies:64,retainedRank:String(BigInt(1)<<BigInt(64))}},{label:'maximum-kl',input:{kind:'error-correction',input:[scalar('1'),scalar('1')],selectedError:'YYY',physicalXProbability:'1/3',errors:Array(16).fill('III')}}];
const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).filter(k=>!(Object.hasOwn(v,'bounds')&&Object.hasOwn(v,'resolution')&&(k==='value'||k==='logMagnitude'))).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
const digest=(v:unknown)=>createHash('sha256').update(stable(v)).digest('hex');

async function main(){
 const expected=cases.map(c=>digest(calculateQuantumInformation(c.input).result));
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try{
  const context=await browser.newContext();await context.addInitScript({content:`globalThis.__name=fn=>fn;localStorage.setItem('suanlab-language','en');window.__informationPorts=[];const Native=Worker;window.Worker=class extends Native{constructor(url,opts){super(url,opts);window.__informationPorts.push({url:String(url),name:opts?.name,type:opts?.type??'classic'});}};`});
  const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base+'/book/online/physics-math-for-ai/binary-qubit-state-discrimination/');await page.locator('[data-information-phase="ready"]').waitFor({timeout:60000});
  type Port={url:string;name:string;type:'module'|'classic'};
  const ports=await page.evaluate(()=>(window as unknown as {__informationPorts:Port[]}).__informationPorts);
  const calculate=ports.find(p=>p.name==='quantum-information-calculation'),validate=ports.find(p=>p.name==='quantum-information-validation');assert(calculate&&validate);
  const assets=[];
  for(const port of [calculate,validate]){const url=new URL(port.url,base);assert(url.pathname.startsWith('/_next/static/'));const response=await fetch(url);assert.equal(response.status,200);const served=Buffer.from(await response.arrayBuffer()),disk=readFileSync('out'+url.pathname);assert.deepEqual(served,disk);assets.push({...port,bytes:disk.length,sha256:createHash('sha256').update(disk).digest('hex')});}
  const probes=await page.evaluate(async({assets,inputs,forged})=>{
   const workers=assets.map(asset=>new Worker(asset.url,{type:asset.type}));let started=0;
   const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).filter(k=>!(Object.hasOwn(v,'bounds')&&Object.hasOwn(v,'resolution')&&(k==='value'||k==='logMagnitude'))).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
   const hash=async(v:unknown)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(stable(v))))).map(b=>b.toString(16).padStart(2,'0')).join('');
   const rpc=(worker:Worker,message:unknown)=>new Promise<Record<string,unknown>>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('RPC deadline')),30000);worker.onerror=e=>{clearTimeout(timer);e.preventDefault();reject(Error(e.message));};worker.onmessage=e=>{if(e.data.phase==='started'){started++;return;}clearTimeout(timer);resolve(e.data);};worker.postMessage(message);});
   const results=[];
   const alter=(value:unknown):boolean=>{
    if(!value||typeof value!=='object')return false;
    const r=value as Record<string,unknown>;
    for(const k of Object.keys(r)){
     const child=r[k];if(child&&typeof child==='object'&&!Array.isArray(child)){
      const c=child as Record<string,unknown>;
      if(Object.hasOwn(c,'bounds')&&Object.hasOwn(c,'resolution')){r[k]=structuredClone(forged[c.value===0?1:0]);return true;}
      if(Array.isArray(c.real)&&typeof c.real[0]==='string'){c.real[0]=c.real[0]==='1'?'0':'1';return true;}
     }
     if(alter(child))return true;
    }return false;
   };
   
   try{
    for(const[i,c]of inputs.entries()){
     const response=await rpc(workers[0],{version:1,requestId:i+1,input:c.input});if(!response.ok)throw Error(c.label+': calculation failed');
     const receipt=await rpc(workers[1],{version:1,requestId:i+1,input:c.input,response});if(receipt.phase!=='validated')throw Error(c.label+': validation failed');
     results.push({label:c.label,hash:await hash(response.result),validatedHash:await hash((receipt.response as Record<string,unknown>).result)});
     const bad=structuredClone(response);if(!alter(bad.result))throw Error('No mathematical coordinate to mutate');
     if((await rpc(workers[1],{version:1,requestId:i+1,input:c.input,response:bad})).phase!=='rejected')throw Error('False mathematics accepted');
    }
    if((await rpc(workers[0],{version:2,requestId:1000,input:inputs[0].input})).error!=='invalid-request')throw Error('Malformed admitted');
    return {results,started,workersCreated:2,workersTerminated:2};
   }finally{workers.forEach(w=>w.terminate());}
  },{assets,inputs:cases,forged:[q(zero),q(one)]});

  assert.equal(probes.results.length,cases.length);assert.equal(probes.started,cases.length);
  for(const[i,row]of probes.results.entries()){assert.equal(row.hash,expected[i],row.label);assert.equal(row.validatedHash,row.hash);}
  assert.deepEqual(errors,[]);
  writeFileSync(root+'/production-assets.json',JSON.stringify({assets,...probes,errors,productionNextAssetsVerified:true,workerFactoryOverride:false,mathematicalMutationsRejected:cases.length},null,2)+'\n');
  console.log(JSON.stringify({cases:cases.length,mutationsRejected:cases.length,assets:assets.map(a=>({name:a.name,bytes:a.bytes,type:a.type}))}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
