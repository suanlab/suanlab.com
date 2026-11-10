import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {calculateRealTimePhase} from '../src/lib/learning-real-time-phase';
import {realTimePhaseExamples} from '../src/lib/learning-real-time-phase-input';
const root='.runtime/learning/real-time-phase-worker';mkdirSync(root,{recursive:true});
const stable=(value:unknown):string=>Array.isArray(value)?`[${value.map(stable).join(',')}]`:value&&typeof value==='object'?`{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${stable((value as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(value);
const hash=(value:unknown)=>createHash('sha256').update(stable(value)).digest('hex');
async function main() {
 const codes:Record<string,string>={};
 for(const[kind,entry]of Object.entries({calculate:'src/lib/learning-real-time-phase.worker.ts',validate:'src/lib/learning-real-time-phase-validation.worker.ts'})) {
  const bundle=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'esm',target:'es2020',metafile:true});codes[kind]=bundle.outputFiles[0].text;
  if(kind==='validate')for(const name of ['learning-real-time-phase.ts','learning-real-time-phase-exact.ts','learning-real-time-wave-observations.ts','learning-real-time-airy.ts','learning-quantum-hilbert-core.ts'])assert(!Object.keys(bundle.metafile!.inputs).includes('src/lib/'+name));
  writeFileSync(`${root}/${kind}.js`,codes[kind]);writeFileSync(`${root}/${kind}-native-dependency-graph.json`,JSON.stringify(bundle.metafile,null,2)+'\n');
 }
 const expected:string[]=[];
 for(const row of realTimePhaseExamples){console.log('Node baseline: '+row.id);expected.push(hash(calculateRealTimePhase(row.input)));writeFileSync(root+'/node-baseline-hashes.json',JSON.stringify(expected,null,2)+'\n');}
 const browser=await chromium.launch({headless:true}),errors:string[]=[];
 try {
  const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=fn=>fn;'});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
  await page.exposeFunction('reportRealTimeWorkerProgress',(value:unknown)=>{writeFileSync(root+'/native-progress.json',JSON.stringify(value,null,2)+'\n');console.log(JSON.stringify(value));});
  await page.route('**/__real_time_phase_workers__/',r=>r.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Real-time Worker verification</title><body><label>Typing<input id="typing"></label></body></html>'}));
  for(const kind of ['calculate','validate'])await page.route(`**/__real_time_phase_workers__/${kind}.js`,r=>r.fulfill({contentType:'text/javascript',body:codes[kind]}));
  await page.goto('http://127.0.0.1:4175/__real_time_phase_workers__/');
  const work=page.evaluate(async inputs=>{
   const workers=[new Worker('calculate.js',{type:'module'}),new Worker('validate.js',{type:'module'})];
   const lifecycle={created:2,terminated:0};let calculationStarted=0,verificationStarted=0;
   const state=window as unknown as{__realTimeBatch:{started:boolean;finished:boolean};reportRealTimeWorkerProgress:(value:unknown)=>Promise<void>};state.__realTimeBatch={started:true,finished:false};
   const stable=(value:unknown):string=>Array.isArray(value)?`[${value.map(stable).join(',')}]`:value&&typeof value==='object'?`{${Object.keys(value).sort().map(k=>`${JSON.stringify(k)}:${stable((value as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(value);
   const digest=async(value:unknown)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(stable(value))))).map(n=>n.toString(16).padStart(2,'0')).join('');
   const rpc=(port:Worker,message:unknown)=>new Promise<Record<string,unknown>>((resolve,reject)=>{
    const timer=setTimeout(()=>reject(Error('Native Worker120s deadline')),120000);
    port.onerror=e=>{clearTimeout(timer);e.preventDefault();reject(Error(e.message));};
    port.onmessage=e=>{if(e.data.phase==='started'){calculationStarted++;return;}if(e.data.phase==='verification-started'){verificationStarted++;return;}clearTimeout(timer);resolve(e.data);};
    port.postMessage(message);
   });
   const changedNumber=(text:string)=>{const[n,d='1']=text.split('/');let a=BigInt(n)+BigInt(d),b=BigInt(d),x=a<BigInt(0)?-a:a,y=b;while(y)[x,y]=[y,x%y];a/=x;b/=x;return b===BigInt(1)?String(a):`${a}/${b}`;};
   const rows=[];let rejectedMutations=0,rejectedAttributions=0;
   try {
    for(const[i,c]of inputs.entries()) {
     const requestId=i+1,t=performance.now(),response=await rpc(workers[0],{version:1,requestId,input:c.input}),calculationMilliseconds=performance.now()-t;
     if(!response.ok)throw Error(c.id+': producer rejected');
     const v=performance.now(),receipt=await rpc(workers[1],{version:1,requestId,input:c.input,response}),validationMilliseconds=performance.now()-v;
     if(receipt.phase!=='validated'||receipt.fullResponseValidated!==true)throw Error(c.id+': validator rejected');
     const accepted=receipt.response as Record<string,unknown>;rows.push({id:c.id,hash:await digest(response.result),validatedHash:await digest(accepted.result),calculationMilliseconds,validationMilliseconds});
     const bad=structuredClone(response),result=bad.result as{kind:string;exact:{determinant:string};wave:{samples:{position:{lower:string}}[]};scaledIntegral:{lower:string}};
     if(result.kind==='chain')result.exact.determinant=changedNumber(result.exact.determinant);
     else if(result.kind==='packet')result.wave.samples[64].position.lower=changedNumber(result.wave.samples[64].position.lower);
     else result.scaledIntegral.lower=changedNumber(result.scaledIntegral.lower);
     if((await rpc(workers[1],{version:1,requestId,input:c.input,response:bad})).phase!=='rejected')throw Error(c.id+': corrupt mathematics accepted');rejectedMutations++;
     if((await rpc(workers[1],{version:1,requestId:requestId+1000,input:c.input,response})).phase!=='rejected')throw Error(c.id+': foreign attribution accepted');rejectedAttributions++;
     await state.reportRealTimeWorkerProgress({completed:i+1,id:c.id,calculationMilliseconds,validationMilliseconds});
    }
    return{rows,calculationStarted,verificationStarted,rejectedMutations,rejectedAttributions,lifecycle};
   }finally{state.__realTimeBatch.finished=true;workers.forEach(w=>{w.terminate();lifecycle.terminated++;});}
  },realTimePhaseExamples.map(row=>({id:row.id,input:row.input})));
  await page.waitForFunction(()=>(window as unknown as{__realTimeBatch?:{started:boolean}}).__realTimeBatch?.started);
  await page.locator('#typing').fill('Typing remains responsive during real-time Worker calculations');
  assert.equal(await page.locator('#typing').inputValue(),'Typing remains responsive during real-time Worker calculations');
  const typingBeforeBatchFinished=await page.evaluate(()=>!(window as unknown as{__realTimeBatch:{finished:boolean}}).__realTimeBatch.finished);assert(typingBeforeBatchFinished);
  const result=await work;assert.equal(result.rows.length,32);assert.equal(result.calculationStarted,32);assert.equal(result.verificationStarted,96);
  assert.equal(result.rejectedMutations,32);assert.equal(result.rejectedAttributions,32);assert.deepEqual(result.lifecycle,{created:2,terminated:2});
  result.rows.forEach((r,i)=>{assert.equal(r.hash,expected[i],r.id);assert.equal(r.validatedHash,r.hash,r.id);});assert.deepEqual(errors,[]);
  const report={...result,actualModuleWorkers:true,browserMatchesNodeProducer:true,independentNumericsVerifiedSeparately:true,typingBeforeBatchFinished,productionNextAssetsVerified:false,readerRegistered:false,errors};
  writeFileSync(root+'/native-workers.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({actualModuleWorkers:true,cases:32,rejectedMutations:32,rejectedAttributions:32,typingBeforeBatchFinished}));
 }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
