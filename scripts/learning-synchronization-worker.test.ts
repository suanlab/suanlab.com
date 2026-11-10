import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {Worker} from 'node:worker_threads';
import {evaluateSynchronizationRequest as evaluate} from '../src/lib/learning-synchronization-evaluator';
import {SynchronizationCalculationController as Controller,type SynchronizationWorkerPort} from '../src/lib/learning-synchronization-controller';
import {synchronizationKinds as kinds,snapshotSynchronizationInput as snapshot,synchronizationInputKey as key,isSynchronizationInput as valid,isSynchronizationSuccessForInput as success,parseSynchronizationInputDraft as draft,parseSynchronizationStoredInput as stored,type SynchronizationInput} from '../src/lib/learning-synchronization-protocol';
const families=['hopf-amplitude','pair-phase-locking','finite-kuramoto'].map((name,i)=>({kind:kinds[i],cases:JSON.parse(readFileSync(`scripts/fixtures/${name}-reference.json`,'utf8')).cases as{label:string;input:SynchronizationInput['parameters']}[]}));
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input} as SynchronizationInput));
for(const family of families)for(const c of family.cases)test(`synchronization transport: ${family.kind}: ${c.label}`,()=>{
 const input=snapshot({kind:family.kind,parameters:c.input} as SynchronizationInput),response=evaluate({version:1,requestId:7,input});assert(response.ok);assert(success(structuredClone(response),input,7),`${family.kind} ${c.label}`);assert(!success(response,input,8));
});
test('canonical owned snapshots preserve integer lexical evidence, including population arrays',()=>{
 for(const input of inputs){
  const reverse={kind:input.kind,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())} as SynchronizationInput;
  assert.equal(key(reverse),key(input));assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);
  for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){
   const raw=JSON.stringify(input).replace(/"timeNumerator":\d+/,`"timeNumerator":${token}`);assert.equal(draft(raw),null);assert.equal(stored(`{"version":1,"value":${raw}}`),null);
  }
  assert(!valid({...input,extra:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('extra')]:1}}));assert.equal(stored(JSON.stringify({version:2,value:input})),null);
 }
 const p=structuredClone(inputs[2]);assert(p.kind===kinds[2]);const own=snapshot(p);assert(own.kind===kinds[2]);p.parameters.omegaNumerators=[0];assert.equal(own.parameters.omegaNumerators.length,4);assert(Object.isFrozen(own.parameters.omegaNumerators));assert(Object.isFrozen(own.parameters.phasePiNumerators));
 for(const token of ['1e-9999','1.00000000000000001'])assert.equal(draft(JSON.stringify(inputs[2]).replace('[-1,0,1,2]',`[${token},0,1,2]`)),null);
});
test('complete schemas reject missing, extra, nonfinite, malformed and misattributed result data',()=>{
 for(const input of inputs){
  const base=evaluate({version:1,requestId:1,input});assert(base.ok);
  const mutate=(fn:(x:any)=>void)=>{const x=structuredClone(base);fn(x);assert(!success(x,input,1));};
  mutate(x=>{delete x.result.current;});mutate(x=>{x.result.extra=1;});mutate(x=>{x.inputKey+=' ';});mutate(x=>{x.kind='other';});mutate(x=>{x.result.parameters.timeNumerator++;});mutate(x=>{x.result.history.reverse();});mutate(x=>{x.result.current=structuredClone(x.result.history[0]);});mutate(x=>{x.result.history[0].step=1;});
  mutate(x=>{x.result.history[0].time=NaN;});mutate(x=>{x.result.history[0].extra=1;});
  if(input.kind===kinds[0]){mutate(x=>{x.result.current.amplitude.value=NaN;});mutate(x=>{x.result.classification.originInitial=!x.result.classification.originInitial;});mutate(x=>{x.result.current.squaredAmplitude.bounds.upper.denominator.coefficient='0';});}
  if(input.kind===kinds[1]){mutate(x=>{x.result.current.phaseBranch.signedCount=null;x.result.history.at(-1).phaseBranch.signedCount=null;});mutate(x=>{x.result.flowMatrix.pop();});mutate(x=>{x.result.fixedPhases[0].kind='other';});}
  if(input.kind===kinds[2]){mutate(x=>{x.result.current.phases.pop();});mutate(x=>{x.result.numerics.errorEstimateIsCertified=true;});mutate(x=>{x.result.continuumReference.fittedFromFiniteFrequencies=true;});mutate(x=>{x.result.classification.finitePopulationCriticalValueInferred=true;});mutate(x=>{x.result.history[1].gridIndex++;});mutate(x=>{x.result.history[0].finiteWindowFrequencies=[0,0,0,0];});}
 }
});
class Port implements SynchronizationWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
test('replacement, stale callbacks, cancellation and final completion release exactly one worker',()=>{
 const ports:Port[]=[],states:string[]=[];const c=new Controller(()=>{const p=new Port();ports.push(p);return p;},s=>states.push(s.status));const id=c.run(inputs[0])!,stale=ports[0].onmessage!;c.run(inputs[2]);assert.equal(ports[0].terminated,1);assert.equal(ports[0].onmessage,null);stale({data:evaluate({version:1,requestId:id,input:inputs[0]})}as MessageEvent);assert.equal(c.getSnapshot().status,'pending');ports[1].emit({version:1,requestId:id,phase:'started',kind:inputs[0].kind});assert.equal(c.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);c.cancel();assert.equal(c.getSnapshot().status,'idle');c.dispose();assert.equal(c.run(inputs[0]),null);assert.deepEqual(states,['pending','pending','ready','idle']);
});
test('unavailable, clone failure, malformed progress, errors and retry never calculate synchronously',()=>{
 const unavailable=new Controller(()=>{throw Error('unavailable');},()=>{});unavailable.run(inputs[0]);assert.equal(unavailable.getSnapshot().status,'error');unavailable.dispose();
 for(const action of ['clone','malformed','progress','error','messageerror','cancel','dispose','invalid']){
  const port=new Port();if(action==='clone')port.postMessage=()=>{throw Error('clone');};const c=new Controller(()=>port,()=>{});c.run(inputs[0]);
  if(action==='malformed')port.emit({version:1,requestId:1,ok:true});if(action==='progress')port.emit({version:1,requestId:1,phase:'started',kind:kinds[1]});if(action==='error')port.onerror?.({preventDefault(){}}as ErrorEvent);if(action==='messageerror')port.onmessageerror?.({}as MessageEvent);if(action==='cancel')c.cancel();if(action==='dispose')c.dispose();if(action==='invalid')c.run({kind:'invalid'});assert.equal(port.terminated,1);assert.equal(port.onmessage,null);if(!['cancel','dispose'].includes(action))assert.equal(c.getSnapshot().status,'error');c.dispose();assert.equal(port.terminated,1);
 }
 const ports:Port[]=[],c=new Controller(()=>{const p=new Port();ports.push(p);return p;},()=>{});c.run(inputs[0]);ports[0].onerror?.({preventDefault(){}}as ErrorEvent);c.run(inputs[1]);ports[1].emit(evaluate(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert(ports.every(p=>p.terminated===1));c.dispose();
});
test('reentrant pending subscribers and worker factories cannot strand a worker',()=>{
 let c:Controller,created=0;const port=new Port();c=new Controller(()=>{created++;return port;},s=>{if(s.status==='pending')c.cancel();});c.run(inputs[0]);assert.equal(created,0);c.dispose();
 c=new Controller(()=>{c.cancel();return port;},()=>{});c.run(inputs[0]);assert.equal(port.terminated,1);assert.equal(c.getSnapshot().status,'idle');c.dispose();
});
test('envelopes and mixed families fail before calculation',()=>{
 for(const value of [null,{}, {version:2,requestId:1,input:inputs[0]}, {version:1,requestId:0,input:inputs[0]}, {version:1,requestId:1,input:inputs[0],extra:1}])assert.equal(evaluate(value).ok,false);
 for(const input of [{kind:kinds[0],parameters:inputs[1].parameters},{kind:'unknown',parameters:inputs[0].parameters}])assert.deepEqual(evaluate({version:1,requestId:1,input}),{version:1,requestId:1,ok:false,error:'invalid-input'});
});
test('reader dependency graph omits calculation engines; actual worker entry emits started and validated results',async()=>{
 const reader=await build({entryPoints:['src/lib/learning-synchronization-controller.ts'],bundle:true,write:false,metafile:true,platform:'browser'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(hopf-amplitude|pair-phase-locking|pair-phase-math|finite-kuramoto)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-synchronization.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});
test('scientific branch metadata stays tied to exact input rather than rounded displays',()=>{
 for(const input of inputs){
  const base=evaluate({version:1,requestId:9,input});assert(base.ok);const x:any=structuredClone(base);
  if(input.kind===kinds[0])x.result.classification.positiveMuCircleIsEquilibria=!x.result.classification.positiveMuCircleIsEquilibria;
  if(input.kind===kinds[1])x.result.classification.stableLockingExists=!x.result.classification.stableLockingExists;
  if(input.kind===kinds[2])x.result.continuumReference.regime='critical';
  assert(!success(x,input,9));
 }
 for(const label of ['continuum rounded equality above','continuum rounded equality below']){
  const p=families[2].cases.find(c=>c.label===label)!.input,input={kind:kinds[2],parameters:p} as SynchronizationInput,response=evaluate({version:1,requestId:1,input});assert(response.ok);assert(success(response,input,1));const x:any=structuredClone(response);x.result.continuumReference.regime='critical';assert(!success(x,input,1));
 }
});
