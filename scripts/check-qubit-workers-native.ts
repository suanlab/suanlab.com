import assert from 'node:assert/strict';
import { mkdirSync,writeFileSync,readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { build } from 'esbuild';
import { chromium } from 'playwright';
import { qubitWorkerCases as cases,qubitWorkerRepresentatives as representatives } from './learning/qubit-worker-cases';
import type { QubitCalculationController,QubitCalculationState } from '../src/lib/learning-qubit-controller';
import type { QubitInput } from '../src/lib/learning-qubit-input';
const runtime='.runtime/learning/quantum-qubit-workers';
const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
const digest=(v:unknown)=>createHash('sha256').update(stable(v)).digest('hex');
async function main(){
  mkdirSync(runtime,{recursive:true});const entries={calculate:'src/lib/learning-qubit.worker.ts',validate:'src/lib/learning-qubit-validation.worker.ts',controller:'src/lib/learning-qubit-controller.ts'};
  const codes:Record<string,string>={},graphs:unknown[]=[];
  for(const [kind,entry]of Object.entries(entries)){
    const result=await build({entryPoints:[entry],bundle:true,write:false,platform:'browser',format:'iife',globalName:kind==='controller'?'QubitRuntime':undefined,metafile:true,logLevel:'silent'});
    codes[kind]=result.outputFiles[0].text;const inputs=Object.keys(result.metafile!.inputs);
    if(kind==='controller')assert(!inputs.some(p=>/learning-(?:quantum-hilbert-core|qubit-(?:scalar|decode|channel-algebra|circuit-exact|circuit-report|channel|evaluator|validation-evaluator))\.ts$/.test(p)));
    if(kind==='validate')for(const excluded of ['learning-qubit-circuit-exact.ts','learning-qubit-circuit-report.ts','learning-qubit-channel.ts','learning-qubit-evaluator.ts'])assert(!inputs.includes(`src/lib/${excluded}`));
    writeFileSync(`${runtime}/${kind}.js`,codes[kind]);graphs.push({kind,entry,inputs,dependencyHashes:Object.fromEntries(inputs.map(p=>[p,createHash('sha256').update(readFileSync(p)).digest('hex')])),metafile:result.metafile,bundleSha256:createHash('sha256').update(codes[kind]).digest('hex')});
  }
  writeFileSync(`${runtime}/worker-dependency-graphs.json`,JSON.stringify({graphs,controllerNumericalEnginesExcluded:true,validationProductEnginesExcluded:true},null,2)+'\n');
  const browser=await chromium.launch({headless:true});const errors:string[]=[];
  try{
    const context=await browser.newContext();await context.addInitScript({content:'globalThis.__name=(fn)=>fn;'});
    const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/__qubit_worker_harness__/',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><html lang="en"><title>Qubit Worker checks</title><body><label for="typing">Typing during calculation</label><input id="typing"></body></html>'}));
    await page.goto('http://localhost:4175/__qubit_worker_harness__/');
    await page.addScriptTag({content:codes.controller});
    const probes=await page.evaluate(async({codes,inputs})=>{
      const urls=[codes.calculate,codes.validate].map(code=>URL.createObjectURL(new Blob([code],{type:'text/javascript'})));
      const calculate=new Worker(urls[0],{type:'module'}),validate=new Worker(urls[1],{type:'module'});let started=0;
      const stable=(v:unknown):string=>Array.isArray(v)?`[${v.map(stable).join(',')}]`:v&&typeof v==='object'?`{${Object.keys(v).sort().map(k=>`${JSON.stringify(k)}:${stable((v as Record<string,unknown>)[k])}`).join(',')}}`:JSON.stringify(v);
      const hash=async(v:unknown)=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(stable(v))))).map(b=>b.toString(16).padStart(2,'0')).join('');
      const rpc=(worker:Worker,frame:unknown)=>new Promise<Record<string,unknown>>((resolve,reject)=>{
        const timer=setTimeout(()=>reject(Error('Native Worker response timeout')),30000);
        worker.onerror=e=>{clearTimeout(timer);e.preventDefault();reject(Error(e.message));};worker.onmessageerror=()=>{clearTimeout(timer);reject(Error('Native clone error'));};
        worker.onmessage=e=>{if(e.data.phase==='started'){started++;return;}clearTimeout(timer);resolve(e.data);};worker.postMessage(frame);
      });
      const results:{label:string;hash:string;validatedHash:string}[]=[];
      try{
        for(const [index,c]of inputs.entries()){
          const id=index+1,response=await rpc(calculate,{version:1,requestId:id,input:c.input});
          if(!response.ok)throw Error(c.label+': '+response.error);
          const receipt=await rpc(validate,{version:1,requestId:id,input:c.input,response});
          if(receipt.phase!=='validated')throw Error(c.label+': validation rejected');
          results.push({label:c.label,hash:await hash(response.result),validatedHash:await hash((receipt.response as Record<string,unknown>).result)});
        }
        const malformed=await rpc(calculate,{version:2,requestId:9000,input:inputs[0].input});
        if(malformed.ok!==false||malformed.error!=='invalid-request')throw Error('Malformed request admitted');
        return {results,started,workersCreated:2,workersTerminated:2,malformedRejected:true};
      }finally{calculate.terminate();validate.terminate();urls.forEach(url=>URL.revokeObjectURL(url));}
    },{codes,inputs:cases.map(c=>({label:c.label,input:c.input}))});
    assert.equal(probes.results.length,595);assert.equal(probes.started,595);
    probes.results.forEach((row,i)=>{assert.equal(row.label,cases[i].label);assert.equal(row.hash,digest(cases[i].expected),row.label);assert.equal(row.validatedHash,row.hash,row.label);});
    const pending=page.evaluate(async({codes,reps,maximum})=>{
      const runtime=(window as unknown as {QubitRuntime:{QubitCalculationController:typeof QubitCalculationController}}).QubitRuntime;
      const calculateUrl=URL.createObjectURL(new Blob([codes.calculate],{type:'text/javascript'})),validateUrl=URL.createObjectURL(new Blob([codes.validate],{type:'text/javascript'}));
      const hangingUrl=URL.createObjectURL(new Blob(['self.onmessage=()=>{};'],{type:'text/javascript'})),brokenUrl=URL.createObjectURL(new Blob(['throw Error("intentional Worker startup failure");'],{type:'text/javascript'}));
      let created=0,terminated=0,validationCreated=0,maximumGap=0,samples=0,last=performance.now(),wasPending=false;
      const active=new Set<Worker>(),states:QubitCalculationState[]=[],controllers:QubitCalculationController[]=[];
      let current:QubitCalculationController|null=null;
      const heartbeat=setInterval(()=>{const now=performance.now(),isPending=current?.getSnapshot().status==='pending';if(isPending){if(wasPending)maximumGap=Math.max(maximumGap,now-last);samples++;}wasPending=!!isPending;last=now;},10);
      const port=(url:string,tamper=false)=>{
        const worker=new Worker(url,{type:'module'}),stop=worker.terminate.bind(worker);created++;active.add(worker);
        worker.terminate=()=>{if(active.delete(worker))terminated++;stop();};
        if(tamper){
          let handler:Worker['onmessage']=null;
          Object.defineProperty(worker,'onmessage',{get:()=>handler,set:value=>{handler=value;worker.addEventListener('message',event=>{
            if(event.data.ok){
              const data=structuredClone(event.data),r=data.result;
              const scalar=data.kind==='circuit'?r.prefixes.at(-1).singleQubits[0].density[0][0]:data.kind==='channel'?r.output[0][0]:data.kind==='environment'?r.system[0][0]:data.kind==='instrument'?r.branches[0].probability:r.record[0][0];
              scalar.real[0]=scalar.real[0]==='0'?'1':'0';handler?.call(worker,{data} as MessageEvent);
            }else handler?.call(worker,event);
          },{once:false});}});
        }return worker;
      };
      const until=async(check:()=>boolean)=>{const start=performance.now();while(!check()){if(performance.now()-start>30000)throw Error('Native controller timeout');await new Promise(r=>setTimeout(r,2));}};
      const make=(options:{tamper?:boolean;hang?:boolean;broken?:boolean;timeout?:number}={})=>{
        const controller=new runtime.QubitCalculationController({calculate:()=>port(options.hang?hangingUrl:options.broken?brokenUrl:calculateUrl,!!options.tamper),validate:()=>{validationCreated++;return port(validateUrl);}},state=>{
          states.push(state);document.body.dataset.phase=state.status==='pending'?state.phase:state.status;
        },{},options.timeout??30000);controllers.push(controller);current=controller;return controller;
      };
      const outcomes:{kind:string;status:string}[]=[];
      try{
        const controller=make();
        for(const input of [...reps,maximum]){controller.run(input);await until(()=>controller.getSnapshot().status!=='pending');const state=controller.getSnapshot();if(state.status!=='ready')throw Error('Native ready failed');outcomes.push({kind:input.kind,status:'ready'});}
        controller.run(maximum);controller.cancel();await new Promise(r=>setTimeout(r,30));if(controller.getSnapshot().status!=='idle')throw Error('Calculation cancellation failed');
        const previous=validationCreated;controller.run(maximum);await until(()=>validationCreated>previous);controller.cancel();await new Promise(r=>setTimeout(r,30));if(controller.getSnapshot().status!=='idle')throw Error('Validation cancellation failed');
        controller.run(reps[1]);controller.run(reps[0]);await until(()=>controller.getSnapshot().status!=='pending');if(controller.getSnapshot().status!=='ready'||controller.getSnapshot().input?.kind!==reps[0].kind)throw Error('Replacement failed');
        controller.dispose();
        for(const input of reps){const bad=make({tamper:true});bad.run(input);await until(()=>bad.getSnapshot().status!=='pending');const state=bad.getSnapshot();if(state.status!=='error'||state.error!=='invalid-response')throw Error('Tampered response admitted');outcomes.push({kind:input.kind,status:'tamper-rejected'});bad.dispose();}
        const timeout=make({hang:true,timeout:100});timeout.run(reps[0]);await until(()=>timeout.getSnapshot().status==='error');const timed=timeout.getSnapshot();if(timed.status!=='error'||timed.error!=='timeout')throw Error('Worker timeout failed');timeout.dispose();
        const broken=make({broken:true});broken.run(reps[0]);await until(()=>broken.getSnapshot().status==='error');const failed=broken.getSnapshot();if(failed.status!=='error'||failed.error!=='worker-failed')throw Error('Worker failure missing');broken.dispose();
        const retry=make();retry.run(reps[4]);await until(()=>retry.getSnapshot().status==='ready');retry.dispose();
        return {outcomes,workersCreated:created,workersTerminated:terminated,activeWorkers:active.size,heartbeatSamples:samples,maximumPendingHeartbeatGapMilliseconds:maximumGap,
          calculationCancelVerified:true,validationCancelVerified:true,replacementVerified:true,tamperKinds:reps.length,tamperMechanism:'valid-scalar-replacement',deadlineVerified:true,startupFailureVerified:true,retryVerified:true,readyStates:states.filter(s=>s.status==='ready').length};
      }finally{clearInterval(heartbeat);controllers.forEach(c=>c.dispose());active.forEach(w=>w.terminate());[calculateUrl,validateUrl,hangingUrl,brokenUrl].forEach(url=>URL.revokeObjectURL(url));}
    },{codes,reps:representatives.map(c=>c.input),maximum:cases.find(c=>c.label==='circuit/sixteen-gates')!.input});
    await page.waitForFunction(()=>document.body.dataset.phase==='validating');
    const typingStart=performance.now();await page.locator('#typing').fill('Input remains responsive');const typingMilliseconds=performance.now()-typingStart;
    const controllers=await pending;
    assert.equal(await page.locator('#typing').inputValue(),'Input remains responsive');assert.equal(controllers.activeWorkers,0);assert.equal(controllers.workersCreated,controllers.workersTerminated);
    assert(controllers.heartbeatSamples>0);assert(controllers.maximumPendingHeartbeatGapMilliseconds<500);assert(typingMilliseconds<1000);assert.deepEqual(errors,[]);
    writeFileSync(`${runtime}/native-workers.json`,JSON.stringify({probePreparations:595,probeResultHashes:probes.results,probeStartedMessages:probes.started,probeWorkersCreated:2,probeWorkersTerminated:2,malformedRequestRejected:probes.malformedRejected,controllers,typingMilliseconds,actualBrowserWorkersVerified:true,productionNextAssetsVerified:false,errors},null,2)+'\n');
    console.log(JSON.stringify({preparations:595,controllerWorkers:controllers.workersCreated,terminated:controllers.workersTerminated,heartbeat:controllers.maximumPendingHeartbeatGapMilliseconds,typingMilliseconds}));
    await context.close();
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
