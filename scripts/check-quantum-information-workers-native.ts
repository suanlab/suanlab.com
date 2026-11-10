import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {quantumInformationPresets} from '../src/lib/learning-quantum-information-presets';
import {calculateQuantumInformation} from '../src/lib/learning-quantum-information-calculation';
import {q,zero,one} from '../src/lib/learning-quantum-hilbert-core';
import type {InformationCalculationController,InformationCalculationState,InformationWorkerPort} from '../src/lib/learning-quantum-information-controller';
const runtime='.runtime/learning/quantum-information-workers';
const scalar=(x:string)=>({real:[x,'0'],imaginary:['0','0']});
const maximum={kind:'bb84',input:[scalar('1'),scalar('1')],comparison:[scalar('1'),scalar('0')],attackFraction:'1/99999999999999999999999999999999',eveZProbability:'1/2',sampleSize:256};
const cases=[...Object.values(quantumInformationPresets).flat().map(p=>({label:p.input.kind+'/'+p.id,input:p.input})),{label:'maximum-bb84',input:maximum},{label:'maximum-source',input:{kind:'entropy',correlationWeight:'1',coherence:'1/3',probabilityOne:'1/3',copies:64,retainedRank:String(BigInt(1)<<BigInt(64))}},{label:'maximum-kl',input:{kind:'error-correction',input:[scalar('1'),scalar('1')],selectedError:'YYY',physicalXProbability:'1/3',errors:Array(16).fill('III')}}];
const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).filter(k=>!(Object.hasOwn(v,'bounds')&&Object.hasOwn(v,'resolution')&&(k==='value'||k==='logMagnitude'))).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
const digest=(v:unknown)=>createHash('sha256').update(stable(v)).digest('hex');
async function main(){
 mkdirSync(runtime,{recursive:true});const codes:Record<string,string>={},graphs=[];
 for(const [kind,entry]of Object.entries({calculate:'src/lib/learning-quantum-information.worker.ts',validate:'src/lib/learning-quantum-information-validation.worker.ts',controller:'src/lib/learning-quantum-information-controller.ts'})){
  const result=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'iife',globalName:kind==='controller'?'InformationRuntime':undefined,metafile:true});codes[kind]=result.outputFiles[0].text;
  const inputs=Object.keys(result.metafile!.inputs);
  if(kind==='controller')assert(!inputs.some(p=>/learning-quantum-hilbert-core|learning-quantum-information-(?:response|calculation|.*validator|evaluator)/.test(p)));
  if(kind==='validate')assert(!inputs.some(p=>/learning-quantum-information-(?:calculation|capacity|discrimination|ensemble|entropies|joint-entropy|relative-entropy|bb84|protocols|error-correction|majorization|bell-dilution)\.ts$/.test(p)));
  writeFileSync(`${runtime}/${kind}.js`,codes[kind]);graphs.push({kind,inputs,metafile:result.metafile});
 }
 writeFileSync(`${runtime}/dependency-graphs.json`,JSON.stringify(graphs,null,2)+'\n');
 const browser=await chromium.launch({headless:true});const errors:string[]=[];
 try{
  const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=(fn)=>fn;'});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/__information_worker_harness__/',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Information Worker checks</title><body><label for="typing">Typing</label><input id="typing"></body></html>'}));
  await page.goto('http://127.0.0.1:4175/__information_worker_harness__/');await page.addScriptTag({content:codes.controller});
  const probes=await page.evaluate(async({codes,inputs,forged})=>{
   const urls=[codes.calculate,codes.validate].map(code=>URL.createObjectURL(new Blob([code],{type:'text/javascript'}))),workers=urls.map(url=>new Worker(url,{type:'module'}));let started=0;
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
   }finally{workers.forEach(w=>w.terminate());urls.forEach(u=>URL.revokeObjectURL(u));}
  },{codes,inputs:cases,forged:[q(zero),q(one)]});
  assert.equal(probes.results.length,cases.length);assert.equal(probes.started,cases.length);
  for(const[i,row]of probes.results.entries()){assert.equal(row.hash,digest(calculateQuantumInformation(cases[i].input).result),row.label);assert.equal(row.validatedHash,row.hash);}
  const pending=page.evaluate(async({codes,reps,maximum,forged})=>{
   const Runtime=(window as unknown as {InformationRuntime:{InformationCalculationController:typeof InformationCalculationController}}).InformationRuntime;
   const urls=[codes.calculate,codes.validate,'self.onmessage=()=>{};','throw Error("intentional startup failure");'].map(code=>URL.createObjectURL(new Blob([code],{type:'text/javascript'})));
   let created=0,terminated=0,validationCreated=0,maximumGap=0,last=performance.now(),samples=0,pendingLast=false;
   const active=new Set<Worker>(),controllers:InformationCalculationController[]=[],states:InformationCalculationState[]=[];let current:InformationCalculationController|null=null;
   const heartbeat=setInterval(()=>{const now=performance.now(),pending=current?.getSnapshot().status==='pending';if(pending){if(pendingLast)maximumGap=Math.max(maximumGap,now-last);samples++;}pendingLast=!!pending;last=now;},10);
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
   const port=(url:string,tamper=false):InformationWorkerPort=>{
    const worker=new Worker(url,{type:'module'});created++;active.add(worker);
    const p:InformationWorkerPort={onmessage:null,onerror:null,onmessageerror:null,postMessage:v=>worker.postMessage(v),terminate:()=>{if(active.delete(worker)){terminated++;worker.terminate();}}};
    worker.onmessage=e=>{const data=e.data;if(tamper&&data.ok){if(!alter(data.result))throw Error('No coordinate to tamper');}p.onmessage?.call(worker,{data}as MessageEvent);};
    worker.onerror=e=>p.onerror?.call(worker,e);worker.onmessageerror=e=>p.onmessageerror?.call(worker,e);return p;
   };
   const until=async(check:()=>boolean)=>{const start=performance.now();while(!check()){if(performance.now()-start>30000)throw Error('Controller deadline');await new Promise(r=>setTimeout(r,2));}};
   const make=(mode:'normal'|'tamper'|'hang-calculate'|'hang-validate'|'broken'='normal',timeout=30000)=>{
    const c=new Runtime.InformationCalculationController({calculate:()=>port(urls[mode==='hang-calculate'?2:mode==='broken'?3:0],mode==='tamper'),validate:()=>{validationCreated++;return port(urls[mode==='hang-validate'?2:1]);}},state=>{states.push(state);document.body.dataset.phase=state.status==='pending'?state.phase:state.status;},{},timeout);controllers.push(c);current=c;return c;
   };
   try{
    const c=make();for(const input of [...reps,maximum]){c.run(input);await until(()=>c.getSnapshot().status!=='pending');if(c.getSnapshot().status!=='ready')throw Error('Native ready failed');}
    c.run(maximum);c.cancel();await new Promise(r=>setTimeout(r,20));if(c.getSnapshot().status!=='idle')throw Error('Cancel calculate failed');
    const n=validationCreated;c.run(maximum);await until(()=>validationCreated>n);c.cancel();await new Promise(r=>setTimeout(r,20));if(c.getSnapshot().status!=='idle')throw Error('Cancel validate failed');
    c.run(maximum);c.run(reps[0]);await until(()=>c.getSnapshot().status!=='pending');if(c.getSnapshot().input?.kind!==reps[0].kind||c.getSnapshot().status!=='ready')throw Error('Replacement failed');
    const bad=make('tamper');bad.run(reps[0]);await until(()=>bad.getSnapshot().status!=='pending');if(bad.getSnapshot().status!=='error')throw Error('Tampered response accepted');
    for(const mode of ['hang-calculate','hang-validate']as const){const h=make(mode,1000);h.run(reps[0]);await until(()=>h.getSnapshot().status!=='pending');const s=h.getSnapshot();if(s.status!=='error'||s.error!=='timeout')throw Error('Timeout failed');}
    const broken=make('broken');broken.run(reps[0]);await until(()=>broken.getSnapshot().status!=='pending');if(broken.getSnapshot().status!=='error')throw Error('Worker startup failure not handled');
    c.run(reps[1]);await until(()=>c.getSnapshot().status!=='pending');if(c.getSnapshot().status!=='ready')throw Error('Retry failed');
    return {created,terminated,active:active.size,validationCreated,maximumGap,samples,readyCount:states.filter(s=>s.status==='ready').length,tamperRejected:true,cancelAndReplacement:true,timeouts:true,startupFailureAndRetry:true};
   }finally{clearInterval(heartbeat);controllers.forEach(c=>c.dispose());active.forEach(w=>w.terminate());urls.forEach(u=>URL.revokeObjectURL(u));}
  },{codes,reps:Object.values(quantumInformationPresets).map(xs=>xs[0].input),maximum,forged:[q(zero),q(one)]});
  await page.locator('#typing').fill('worker remains responsive');const lifecycle=await pending;
  assert.equal(lifecycle.active,0);assert.equal(lifecycle.created,lifecycle.terminated);assert(lifecycle.samples>20);assert(lifecycle.maximumGap<250,`Main-thread gap ${lifecycle.maximumGap}`);assert.equal(await page.locator('#typing').inputValue(),'worker remains responsive');assert.deepEqual(errors,[]);
  writeFileSync(`${runtime}/native-verification.json`,JSON.stringify({normalCases:cases.length,mathematicalTamperRejected:cases.length+1,probes,lifecycle,pageErrors:errors,nativeModuleWorkers:true,comparison:'Exact bounds/sign/resolution and all non-display fields; value/logMagnitude validated with native formatter',productionReaderRegistered:false},null,2)+'\n');
  console.log(JSON.stringify({normalCases:cases.length,created:probes.workersCreated+lifecycle.created,terminated:probes.workersTerminated+lifecycle.terminated,maximumMainThreadGap:lifecycle.maximumGap,pageErrors:errors.length}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
