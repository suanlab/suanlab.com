import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {Worker as ThreadWorker} from 'node:worker_threads';
import {evaluatePhaseRequest as evaluate} from '../src/lib/learning-phase-transitions-evaluator';
import {phaseKinds as kinds,isPhaseInput,snapshotPhaseInput as snapshot,phaseInputKey as key,isPhaseSuccessForInput as valid,isPhaseObservation as observation,type PhaseInput} from '../src/lib/learning-phase-transitions-protocol';
import {PhaseCalculationController as Controller,type PhaseWorkerPort,type PhaseCalculationState} from '../src/lib/learning-phase-transitions-controller';
import {usePhaseCalculation} from '../src/components/learning/use-phase-transitions-calculation';
const finite:PhaseInput={kind:kinds[0],parameters:{spinCount:2,couplingK:0,fieldB:0}};
const variational:PhaseInput={kind:kinds[1],parameters:{couplingK:2,fieldB:0}};
const landau:PhaseInput={kind:kinds[2],parameters:{quadraticA:3/16,quarticB:-1,sexticC:1,fieldH:0}};
class Port implements PhaseWorkerPort {
 onmessage:Worker['onmessage']=null;onerror:Worker['onerror']=null;onmessageerror:Worker['onmessageerror']=null;
 messages:unknown[]=[];terminated=0;postFailure=false;
 postMessage(v:unknown){if(this.postFailure)throw Error('clone');this.messages.push(structuredClone(v));}
 terminate(){this.terminated++;}
 emit(data:unknown){this.onmessage?.call(this as unknown as Worker,new MessageEvent('message',{data}));}
}
function setup(){const ports:Port[]=[],states:PhaseCalculationState[]=[];const controller=new Controller(()=>{const p=new Port();ports.push(p);return p;},state=>states.push(state));return {ports,states,controller};}
for(const [file,kind] of [['curie-weiss',kinds[0]],['curie-weiss-variational',kinds[1]],['landau-sixth-order',kinds[2]]] as const)test(`${kind}: every independent reference preparation passes its real dispatch and transport contract`,()=>{
 for(const c of JSON.parse(readFileSync(`scripts/fixtures/${file}-oracle.json`,'utf8')).cases){const input={kind,parameters:c.parameters} as PhaseInput;const r=evaluate({version:1,requestId:3,input});assert.equal(r.ok,true,c.id);assert.ok(valid(r,input,3),c.id);assert.doesNotThrow(()=>structuredClone(r));assert.doesNotThrow(()=>JSON.stringify(r));}
});
test('invalid field, confinement, finite size, precision and request envelopes fail before numerical work',()=>{
 for(const input of [null,{}, {...finite,parameters:{...finite.parameters,spinCount:0}},{...finite,parameters:{...finite.parameters,couplingK:NaN}},{...variational,parameters:{couplingK:2,fieldB:5}},{...landau,parameters:{...landau.parameters,sexticC:0}}, {...landau,options:null},{...landau,options:{maximumPrecision:129}},{...landau,options:{minimumPrecision:null}},{...landau,options:{minimumPrecision:256,maximumPrecision:128}}]){assert.equal(isPhaseInput(input),false);assert.equal(evaluate({version:1,requestId:1,input}).ok,false);}
 for(const value of [null,{}, {version:2,requestId:1,input:finite},{version:1,requestId:0,input:finite}])assert.equal(evaluate(value).ok,false);
});
test('canonical input keys and immutable snapshots preserve explicit precision and own their parameters',()=>{
 const input={...finite,parameters:{...finite.parameters}},s=snapshot(input),before=key(s);input.parameters.fieldB=1;assert.equal(key(s),before);assert.ok(Object.isFrozen(s.parameters));assert.ok(Object.isFrozen(s.options));
 assert.equal(key(finite),key({...finite,options:{minimumPrecision:128,maximumPrecision:8192}}));assert.notEqual(key(finite),key({...finite,options:{maximumPrecision:128}}));
});
test('replacement, old callbacks and wrong-generation replies cannot overwrite the latest preparation',()=>{
 const {controller,ports}=setup();controller.run(finite);const old=ports[0].onmessage!;controller.run(landau);assert.equal(ports[0].terminated,1);old.call(ports[0] as unknown as Worker,new MessageEvent('message',{data:evaluate(ports[0].messages[0])}));assert.equal(controller.getSnapshot().status,'pending');
 ports[1].emit({version:1,requestId:1,phase:'started',kind:kinds[2]});assert.equal(controller.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].messages[0]));assert.equal(controller.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);
});
test('input identity, omitted observables, malformed rationals and cycles reject a response and detach its worker',()=>{
 for(const mode of ['key','kind','input','missing','bounds','cycle','resolution','version']){
  const {controller,ports}=setup();controller.run(finite);const r=structuredClone(evaluate(ports[0].messages[0]));assert.ok(r.ok&&r.kind===kinds[0]);const v=r as unknown as Record<string,unknown>;
  if(mode==='key')v.inputKey='wrong';if(mode==='kind')v.kind=kinds[1];if(mode==='input')r.input.parameters.fieldB=1;if(mode==='missing')delete (r.result as unknown as Record<string,unknown>).fieldSusceptibility;
  if(mode==='bounds')r.result.states[0].probability.bounds!.lower.denominator.coefficient='0';if(mode==='cycle')(r.result as unknown as Record<string,unknown>).extra=r.result;if(mode==='resolution')r.result.fullyResolved=false;if(mode==='version')v.version=2;
  ports[0].emit(r);const state=controller.getSnapshot();assert.equal(state.status,'error',mode);if(state.status==='error')assert.equal(state.error,'invalid-response');assert.equal(ports[0].terminated,1);
 }
});
test('unresolved variational spinodal comparisons and algebraic Landau ties are valid transport outcomes',()=>{
 for(const input of [{...variational,parameters:{couplingK:9/8,fieldB:.028426409720027343},options:{maximumPrecision:128}},{...landau,parameters:{quadraticA:59/64,quarticB:-13/4,sexticC:3,fieldH:1/8},options:{maximumPrecision:128}}] as PhaseInput[]){const r=evaluate({version:1,requestId:1,input});assert.ok(r.ok);assert.equal(r.result.fullyResolved,false);assert.ok(valid(r,input,1));}
});
test('creation, clone, execution and deserialization failures have no synchronous fallback',()=>{
 let calls=0;const c=new Controller(()=>{calls++;throw Error('blocked');},()=>{});c.run(finite);assert.equal(calls,1);assert.equal(c.getSnapshot().status,'error');
 for(const mode of ['post','error','messageerror','model']){const port=new Port();port.postFailure=mode==='post';const c=new Controller(()=>port,()=>{});c.run(finite);if(mode==='error')port.onerror!.call(port as unknown as Worker,new Event('error') as ErrorEvent);if(mode==='messageerror')port.onmessageerror!.call(port as unknown as Worker,new MessageEvent('messageerror'));if(mode==='model')port.emit({version:1,requestId:1,ok:false,error:'calculation-failed'});assert.equal(c.getSnapshot().status,'error');assert.equal(port.terminated,1);}
});
test('cancellation, disposal and reentrant notifications terminate every owned port once',()=>{
 const {controller,ports,states}=setup();controller.run(finite);controller.cancel();controller.run(landau);const count=states.length;controller.dispose();controller.dispose();controller.cancel();assert.equal(controller.run(variational),null);assert.equal(states.length,count);assert.deepEqual(ports.map(p=>p.terminated),[1,1]);
 let calls=0;const c=new Controller(()=>{calls++;return new Port();},state=>{if(state.status==='pending')c.cancel();});c.run(finite);assert.equal(calls,0);
 const old=new Port(),fresh=new Port();let created=0;const replacement=new Controller(()=>{if(++created===1){replacement.run(variational);return old;}return fresh;},()=>{});replacement.run(finite);assert.equal(old.terminated,1);fresh.emit(evaluate(fresh.messages[0]));assert.equal(replacement.getSnapshot().status,'ready');assert.equal(fresh.terminated,1);
});
test('SSR hook never reads Worker or executes a numerical calculation',()=>{
 const original=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR');}});
 function Probe(){const r=usePhaseCalculation(finite);assert.equal(r.response,null);return React.createElement('output',null,r.status);}
 try{assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');assert.equal(reads,0);}finally{if(original)Object.defineProperty(globalThis,'Worker',original);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('mathematical infinity, outside-range nonzero and malformed zero are distinct observations',()=>{
 assert.ok(observation({value:null,sign:1,logMagnitude:null,logResolution:'not-applicable',resolution:'mathematical-infinity',bounds:null}));assert.equal(observation({value:0,sign:1,logMagnitude:-800,logResolution:'finite',resolution:'interval-resolved',bounds:null}),false);
});
test('a real thread posts started, can be terminated during heavy work, and a fresh thread completes',async()=>{
 const built=await build({entryPoints:['src/lib/learning-phase-transitions.worker.ts'],bundle:true,write:false,format:'cjs',platform:'node',target:'es2022'});
 const code=`const {parentPort}=require('node:worker_threads'); globalThis.self={postMessage(v){parentPort.postMessage(v)},set onmessage(fn){parentPort.on('message',data=>fn({data}))}};\n${built.outputFiles[0].text}`;
 const heavy=new ThreadWorker(code,{eval:true});let successes=0;
 try{
  await new Promise<void>((resolve,reject)=>{heavy.on('error',reject);heavy.on('message',v=>{if(v.phase==='started')resolve();if(v.ok)successes++;});heavy.postMessage({version:1,requestId:1,input:{kind:kinds[0],parameters:{spinCount:128,couplingK:2,fieldB:Number.MIN_VALUE}}});});
  await heavy.terminate();assert.equal(successes,0);
 }finally{await heavy.terminate();}
 const fresh=new ThreadWorker(code,{eval:true});
 try{const response=await new Promise<unknown>((resolve,reject)=>{fresh.on('error',reject);fresh.on('message',v=>{if(v.ok!==undefined)resolve(v);});fresh.postMessage({version:1,requestId:2,input:finite});});assert.ok(valid(response,finite,2));}finally{await fresh.terminate();}
});

test('root-family transport rejects missing stability data, altered selection and absent fold labels',()=>{
 for(const input of [variational,landau])for(const mode of ['missing','selection','residual','role']){
  const r=structuredClone(evaluate({version:1,requestId:1,input}));assert.ok(r.ok&&r.kind!==kinds[0]);const result=r.result as unknown as Record<string,unknown>,roots=result.roots as Record<string,unknown>[];
  if(mode==='missing')delete roots[0].curvature;if(mode==='selection')result.globalMinimumIndices=[99];if(mode==='residual')delete roots[0].residualEnclosure;if(mode==='role')roots[0].role='unstable-maximum';
  assert.equal(valid(r,input,1),false,`${input.kind}.${mode}`);
 }
});

test('null or primitive root rows fail validation without leaving an active worker',()=>{
 for(const row of [null,4,'root']){const {controller,ports}=setup();controller.run(landau);const r=structuredClone(evaluate(ports[0].messages[0]));assert.ok(r.ok&&r.kind===kinds[2]);(r.result.roots as unknown[])[0]=row;assert.doesNotThrow(()=>ports[0].emit(r));assert.equal(controller.getSnapshot().status,'error');assert.equal(ports[0].terminated,1);}
});
test('browser hook dependency graph excludes all numerical models and exact arithmetic helpers',async()=>{
 const r=await build({entryPoints:['src/components/learning/use-phase-transitions-calculation.ts'],bundle:true,write:false,format:'esm',platform:'browser',metafile:true,external:['react'],logLevel:'silent'});
 const paths=Object.keys(r.metafile!.inputs);for(const file of ['learning-curie-weiss.ts','learning-curie-weiss-variational.ts','learning-landau-sixth-order.ts','learning-exact-polynomial-roots.ts','learning-interval-transcendentals.ts'])assert.ok(!paths.some(p=>p.endsWith(file)),file);
 assert.ok(paths.some(p=>p.endsWith('learning-phase-transitions-worker-factory.ts')));
});
