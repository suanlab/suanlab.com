import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {Worker as ThreadWorker} from 'node:worker_threads';
import {evaluateCorrelationRequest as evaluate} from '../src/lib/learning-correlations-evaluator';
import {correlationKinds as kinds,isCorrelationInput,snapshotCorrelationInput as snapshot,correlationInputKey as key,isCorrelationSuccessForInput as valid,isCorrelationObservation as observation,type CorrelationInput} from '../src/lib/learning-correlations-protocol';
import {CorrelationCalculationController as Controller,type CorrelationWorkerPort,type CorrelationCalculationState} from '../src/lib/learning-correlations-controller';
import {useCorrelationCalculation} from '../src/components/learning/use-correlations-calculation';
const pair:CorrelationInput={kind:kinds[0],parameters:{couplingK:0,fieldOne:0,fieldTwo:0,probeOne:1,probeTwo:1}};
const ring:CorrelationInput={kind:kinds[1],parameters:{spinCount:3,couplingK:-.5}};
const ar:CorrelationInput={kind:kinds[2],parameters:{sampleCount:4,correlationRho:-.5}};
class Port implements CorrelationWorkerPort {
 onmessage:Worker['onmessage']=null;onerror:Worker['onerror']=null;onmessageerror:Worker['onmessageerror']=null;
 messages:unknown[]=[];terminated=0;postFailure=false;
 postMessage(v:unknown){if(this.postFailure)throw Error('clone');this.messages.push(structuredClone(v));}
 terminate(){this.terminated++;}
 emit(data:unknown){this.onmessage?.call(this as unknown as Worker,new MessageEvent('message',{data}));}
}
function setup(){const ports:Port[]=[],states:CorrelationCalculationState[]=[];const controller=new Controller(()=>{const p=new Port();ports.push(p);return p;},state=>states.push(state));return {ports,states,controller};}
for(const [file,kind] of [['binary-correlations',kinds[0]],['periodic-ising-ring',kinds[1]],['ar1-average',kinds[2]]] as const)test(`${kind}: every independent preparation passes real dispatch and transport`,()=>{
 for(const c of JSON.parse(readFileSync(`scripts/fixtures/${file}-oracle.json`,'utf8')).cases){const input={kind,parameters:c.parameters} as CorrelationInput;const r=evaluate({version:1,requestId:3,input});assert.equal(r.ok,true,c.id);assert.ok(valid(r,input,3),c.id);assert.doesNotThrow(()=>structuredClone(r));assert.doesNotThrow(()=>JSON.stringify(r));}
});
test('invalid dimensions, extra fields, endpoints and precision budgets fail before calculation',()=>{
 const bad=[null,{}, {...pair,parameters:{...pair.parameters,probeOne:5}},{...pair,parameters:{...pair.parameters,couplingK:NaN}},{...ring,parameters:{spinCount:2,couplingK:0}},{...ar,parameters:{sampleCount:513,correlationRho:0}},{...ar,parameters:{sampleCount:4,correlationRho:1.1}},{...ring,parameters:{...ring.parameters,fieldB:0}}, {...ar,extra:true}, {...ar,options:null},{...ar,options:{maximumPrecision:129}},{...ar,options:{minimumPrecision:null}},{...ar,options:{minimumPrecision:256,maximumPrecision:128}},{...ar,options:{typo:128}}];
 for(const input of bad){assert.equal(isCorrelationInput(input),false);assert.equal(evaluate({version:1,requestId:1,input}).ok,false);}
 for(const value of [null,{}, {version:2,requestId:1,input:pair},{version:1,requestId:0,input:pair}])assert.equal(evaluate(value).ok,false);
});
test('canonical input keys and immutable snapshots preserve explicit precision and own their parameters',()=>{
 const input={...pair,parameters:{...pair.parameters}},s=snapshot(input),before=key(s);input.parameters.fieldOne=1;assert.equal(key(s),before);assert.ok(Object.isFrozen(s.parameters));assert.ok(Object.isFrozen(s.options));
 assert.equal(key(pair),key({...pair,options:{minimumPrecision:128,maximumPrecision:8192}}));assert.notEqual(key(pair),key({...pair,options:{maximumPrecision:128}}));
});
test('replacement, old callbacks and wrong-generation replies cannot overwrite the latest preparation',()=>{
 const {controller,ports}=setup();controller.run(pair);const old=ports[0].onmessage!;controller.run(ar);assert.equal(ports[0].terminated,1);old.call(ports[0] as unknown as Worker,new MessageEvent('message',{data:evaluate(ports[0].messages[0])}));assert.equal(controller.getSnapshot().status,'pending');
 ports[1].emit({version:1,requestId:1,phase:'started',kind:kinds[2]});assert.equal(controller.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].messages[0]));assert.equal(controller.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);
});
test('input identity, omitted observables, malformed rationals and cycles reject a response and detach its worker',()=>{
 for(const mode of ['key','kind','input','missing','bounds','cycle','resolution','version']){
  const {controller,ports}=setup();controller.run(pair);const r=structuredClone(evaluate(ports[0].messages[0]));assert.ok(r.ok&&r.kind===kinds[0]);const v=r as unknown as Record<string,unknown>;
  if(mode==='key')v.inputKey='wrong';if(mode==='kind')v.kind=kinds[1];if(mode==='input')r.input.parameters.fieldOne=1;if(mode==='missing')delete (r.result as unknown as Record<string,unknown>).probeVariance;
  if(mode==='bounds')r.result.states[0].probability.bounds!.lower.denominator.coefficient='0';if(mode==='cycle')(r.result as unknown as Record<string,unknown>).extra=r.result;if(mode==='resolution')r.result.fullyResolved=false;if(mode==='version')v.version=2;
  ports[0].emit(r);const state=controller.getSnapshot();assert.equal(state.status,'error',mode);if(state.status==='error')assert.equal(state.error,'invalid-response');assert.equal(ports[0].terminated,1);
 }
});
test('precision-capped nonzero correlations and exact degenerate infinity remain valid transport outcomes',()=>{
 for(const input of [{...pair,parameters:{...pair.parameters,couplingK:Number.MIN_VALUE},options:{maximumPrecision:128}},{...ring,parameters:{spinCount:128,couplingK:Number.MIN_VALUE},options:{maximumPrecision:128}}] as CorrelationInput[]){const r=evaluate({version:1,requestId:1,input});assert.ok(r.ok);assert.equal(r.result.fullyResolved,false);assert.ok(valid(r,input,1));}
 const input={...ar,parameters:{sampleCount:4,correlationRho:-1}} as CorrelationInput;const r=evaluate({version:1,requestId:1,input});assert.ok(r.ok&&r.kind===kinds[2]);assert.equal(r.result.varianceEquivalentSampleCount.resolution,'mathematical-infinity');assert.ok(valid(r,input,1));
});
test('creation, clone, execution and deserialization failures have no synchronous fallback',()=>{
 let calls=0;const c=new Controller(()=>{calls++;throw Error('blocked');},()=>{});c.run(pair);assert.equal(calls,1);assert.equal(c.getSnapshot().status,'error');
 for(const mode of ['post','error','messageerror','model']){const port=new Port();port.postFailure=mode==='post';const c=new Controller(()=>port,()=>{});c.run(pair);if(mode==='error')port.onerror!.call(port as unknown as Worker,new Event('error') as ErrorEvent);if(mode==='messageerror')port.onmessageerror!.call(port as unknown as Worker,new MessageEvent('messageerror'));if(mode==='model')port.emit({version:1,requestId:1,ok:false,error:'calculation-failed'});assert.equal(c.getSnapshot().status,'error');assert.equal(port.terminated,1);}
});
test('cancellation, disposal and reentrant notifications terminate every owned port once',()=>{
 const {controller,ports,states}=setup();controller.run(pair);controller.cancel();controller.run(ar);const count=states.length;controller.dispose();controller.dispose();controller.cancel();assert.equal(controller.run(ring),null);assert.equal(states.length,count);assert.deepEqual(ports.map(p=>p.terminated),[1,1]);
 let calls=0;const c=new Controller(()=>{calls++;return new Port();},state=>{if(state.status==='pending')c.cancel();});c.run(pair);assert.equal(calls,0);
 const old=new Port(),fresh=new Port();let created=0;const replacement=new Controller(()=>{if(++created===1){replacement.run(ring);return old;}return fresh;},()=>{});replacement.run(pair);assert.equal(old.terminated,1);fresh.emit(evaluate(fresh.messages[0]));assert.equal(replacement.getSnapshot().status,'ready');assert.equal(fresh.terminated,1);
});
test('SSR hook never reads Worker or executes a numerical calculation',()=>{
 const original=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR');}});
 function Probe(){const r=useCorrelationCalculation(pair);assert.equal(r.response,null);return React.createElement('output',null,r.status);}
 try{assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');assert.equal(reads,0);}finally{if(original)Object.defineProperty(globalThis,'Worker',original);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('mathematical infinity, outside-range nonzero and malformed zero are distinct observations',()=>{
 assert.ok(observation({value:null,sign:1,logMagnitude:null,logResolution:'not-applicable',resolution:'mathematical-infinity',bounds:null}));assert.equal(observation({value:0,sign:1,logMagnitude:-800,logResolution:'finite',resolution:'interval-resolved',bounds:null}),false);
});
test('each real numerical branch posts started and is terminated during heavy work; a fresh thread recovers',async()=>{
 const built=await build({entryPoints:['src/lib/learning-correlations.worker.ts'],bundle:true,write:false,format:'cjs',platform:'node',target:'es2022'});
 const code=`const {parentPort}=require('node:worker_threads'); globalThis.self={postMessage(v){parentPort.postMessage(v)},set onmessage(fn){parentPort.on('message',data=>fn({data}))}};\n${built.outputFiles[0].text}`;
 const options={minimumPrecision:8192,maximumPrecision:8192};
 for(const input of [{...pair,parameters:{...pair.parameters,couplingK:Number.MIN_VALUE,fieldOne:.8},options},{...ring,parameters:{spinCount:128,couplingK:Number.MIN_VALUE},options},{...ar,parameters:{sampleCount:512,correlationRho:.9999999999999999},options}] as CorrelationInput[]){
  const heavy=new ThreadWorker(code,{eval:true});let successes=0;
  try{await new Promise<void>((resolve,reject)=>{heavy.on('error',reject);heavy.on('message',v=>{if(v.phase==='started')resolve();if(v.ok)successes++;});heavy.postMessage({version:1,requestId:1,input});});await heavy.terminate();assert.equal(successes,0,input.kind);}finally{await heavy.terminate();}
 }
 const fresh=new ThreadWorker(code,{eval:true});try{const response=await new Promise<unknown>((resolve,reject)=>{fresh.on('error',reject);fresh.on('message',v=>{if(v.ok!==undefined)resolve(v);});fresh.postMessage({version:1,requestId:2,input:pair});});assert.ok(valid(response,pair,2));}finally{await fresh.terminate();}
});

test('family-specific missing data, malformed rows and endpoint labels fail transport',()=>{
 for(const input of [pair,ring,ar])for(const mode of ['missing','row','attempts','endpoint']){
  const r=structuredClone(evaluate({version:1,requestId:1,input}));assert.ok(r.ok);const result=r.result as unknown as Record<string,unknown>;
  if(mode==='missing')delete result[input.kind===kinds[0]?'probeVariance':input.kind===kinds[1]?'infiniteChainCorrelationLength':'standardError'];
  if(mode==='row'){if(input.kind===kinds[0])result.means=[null,0];else if(input.kind===kinds[1])result.correlations=[null];else result.autocovariances=[null];}
  if(mode==='attempts')result.attempts=[128,512];
  if(mode==='endpoint'){if(input.kind===kinds[0])result.states=[];else if(input.kind===kinds[1])result.oddAntiferromagnetic=false;else result.asymptoticStatus='not-applicable-nonmixing-endpoint';}
  assert.equal(valid(r,input,1),false,`${input.kind}.${mode}`);
 }
});
test('browser hook dependency graph excludes all numerical models and exact arithmetic helpers',async()=>{
 const r=await build({entryPoints:['src/components/learning/use-correlations-calculation.ts'],bundle:true,write:false,format:'esm',platform:'browser',metafile:true,external:['react'],logLevel:'silent'});
 const paths=Object.keys(r.metafile!.inputs);for(const file of ['learning-binary-correlations.ts','learning-periodic-ising-ring.ts','learning-ar1-average.ts','learning-binary-products.ts','learning-interval-transcendentals.ts'])assert.ok(!paths.some(p=>p.endsWith(file)),file);
 assert.ok(paths.some(p=>p.endsWith('learning-correlations-worker-factory.ts')));
});
