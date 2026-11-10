import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdirSync,writeFileSync} from 'node:fs';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {build} from 'esbuild';
import {Worker as ThreadWorker} from 'node:worker_threads';
import {evaluateDetailedBalanceRequest as evaluate} from '../src/lib/learning-detailed-balance-evaluator';
import {detailedBalanceKinds as kinds,isDetailedBalanceInput,snapshotDetailedBalanceInput as snapshot,detailedBalanceInputKey as key,isDetailedBalanceSuccessForInput as valid,isDetailedBalanceObservation as observation,type DetailedBalanceInput} from '../src/lib/learning-detailed-balance-protocol';
import {DetailedBalanceCalculationController as Controller,type DetailedBalanceWorkerPort,type DetailedBalanceCalculationState} from '../src/lib/learning-detailed-balance-controller';
import {useDetailedBalanceCalculation} from '../src/components/learning/use-detailed-balance-calculation';
const cycle:DetailedBalanceInput={kind:kinds[0],parameters:{forwardWeight:2,backwardWeight:1,stayWeight:1,initialWeights:[1,0,0],stepCount:12}};
const hastings:DetailedBalanceInput={kind:kinds[1],parameters:{targetWeights:[1,2,4],forwardWeight:2,backwardWeight:1,stayWeight:1}};
const binary:DetailedBalanceInput={kind:kinds[2],parameters:{forwardNumerator:1,backwardNumerator:2,transitionDenominator:4,initialOneNumerator:0,initialDenominator:1,stepCount:12}};
const normal=[cycle,hastings,binary];
const heavy:DetailedBalanceInput[]=[
 {kind:kinds[0],parameters:{forwardWeight:1000000,backwardWeight:999999,stayWeight:1000000,initialWeights:[1,0,0],stepCount:512}},
 {kind:kinds[1],parameters:{targetWeights:[1000000,999999,999983],forwardWeight:1000000,backwardWeight:999999,stayWeight:1}},
 {kind:kinds[2],parameters:{forwardNumerator:500000,backwardNumerator:500001,transitionDenominator:1000000,initialOneNumerator:0,initialDenominator:1,stepCount:512}},
];
class Port implements DetailedBalanceWorkerPort {
 onmessage:Worker['onmessage']=null;onerror:Worker['onerror']=null;onmessageerror:Worker['onmessageerror']=null;
 messages:unknown[]=[];terminated=0;postFailure=false;
 postMessage(value:unknown){if(this.postFailure)throw Error('clone');this.messages.push(structuredClone(value));}
 terminate(){this.terminated++;}
 emit(data:unknown){this.onmessage?.call(this as unknown as Worker,new MessageEvent('message',{data}));}
}
function setup(){const ports:Port[]=[],states:DetailedBalanceCalculationState[]=[];const controller=new Controller(()=>{const p=new Port();ports.push(p);return p;},s=>states.push(s));return {ports,states,controller};}
for(const [i,file] of ['stationary-cycle','hastings-proposal','two-state-mixing'].entries())test(`${kinds[i]}: every independent preparation passes real model dispatch and full transport structure`,()=>{
 let cases=0;for(const c of JSON.parse(readFileSync(`scripts/fixtures/${file}-oracle.json`,'utf8')).cases){const input={kind:kinds[i],parameters:c.parameters} as DetailedBalanceInput;const r=evaluate({version:1,requestId:1,input});assert.ok(r.ok,c.id);assert.ok(valid(r,input,1),c.id);const cloned=structuredClone(r);assert.ok(valid(cloned,input,1),c.id);assert.doesNotThrow(()=>JSON.stringify(r));cases++;}assert.equal(cases,[24,42,30][i]);
});
test('input validation rejects malformed integers, sparse arrays, denominator mismatches and inherited/extra envelopes',()=>{
 const bad=[null,{}, {...cycle,parameters:{...cycle.parameters,initialWeights:new Array(3)}},{...cycle,parameters:{...cycle.parameters,initialWeights:[0,0,0]}}, {...cycle,parameters:{...cycle.parameters,forwardWeight:NaN}}, {...hastings,parameters:{...hastings.parameters,targetWeights:[0,1,2]}}, {...binary,parameters:{...binary.parameters,transitionDenominator:0}},{...binary,parameters:{...binary.parameters,initialOneNumerator:2}},{...binary,parameters:{...binary.parameters,stepCount:513}}, {...cycle,options:{}},{...hastings,parameters:{...hastings.parameters,stepCount:1}},{...cycle,parameters:{...cycle.parameters,forwardWeight:.5}}];
 for(const input of bad){assert.equal(isDetailedBalanceInput(input),false);const r=evaluate({version:1,requestId:1,input});assert.deepEqual(r,{version:1,requestId:1,ok:false,error:'invalid-input'});}
 for(const value of [null,{}, {version:2,requestId:1,input:cycle},{version:1,requestId:0,input:cycle},{version:1,requestId:1,input:cycle,extra:0},Object.assign(Object.create({version:1}),{requestId:1,input:cycle,extra:0})]){const r=evaluate(value);assert.ok(!r.ok);assert.equal(r.error,'invalid-request');}
});
test('canonical snapshots own arrays, normalize signed zero and ignore incoming property order',()=>{
 const input={...cycle,parameters:{...cycle.parameters,initialWeights:[1,-0,0]}},s=snapshot(input),before=key(s);input.parameters.initialWeights[0]=2;assert.equal(key(s),before);assert.ok(Object.isFrozen(s));assert.ok(Object.isFrozen(s.parameters));assert.ok(s.kind===kinds[0]);assert.ok(Object.isFrozen(s.parameters.initialWeights));
 const reordered={parameters:{stepCount:12,initialWeights:[1,0,0],stayWeight:1,backwardWeight:1,forwardWeight:2},kind:kinds[0]};assert.ok(isDetailedBalanceInput(reordered));assert.equal(key(reordered),key(cycle));assert.equal(key({...cycle,parameters:{...cycle.parameters,initialWeights:[1,-0,0]}}),key(cycle));assert.notEqual(key(cycle),key({...cycle,parameters:{...cycle.parameters,stepCount:0}}));
});
test('started acknowledgement, ready publication and termination belong to the current request',()=>{
 const {controller,ports,states}=setup();controller.run(cycle);assert.equal(controller.getSnapshot().status,'pending');ports[0].emit({version:1,requestId:1,phase:'started',kind:kinds[0]});const pending=controller.getSnapshot();assert.ok(pending.status==='pending');assert.equal(pending.phase,'calculating');ports[0].emit(evaluate(ports[0].messages[0]));assert.equal(controller.getSnapshot().status,'ready');assert.equal(ports[0].terminated,1);assert.equal(ports[0].onmessage,null);assert.equal(states.filter(s=>s.status==='ready').length,1);
});
test('replacement and saved late callbacks cannot publish old generations or mutate the newest preparation',()=>{
 const {controller,ports}=setup();controller.run(cycle);const late=ports[0].onmessage!;controller.run(binary);assert.equal(ports[0].terminated,1);late.call(ports[0] as unknown as Worker,new MessageEvent('message',{data:evaluate(ports[0].messages[0])}));ports[1].emit({version:1,requestId:1,phase:'started',kind:kinds[2]});assert.equal(controller.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].messages[0]));const state=controller.getSnapshot();assert.ok(state.status==='ready');assert.equal(state.response.kind,kinds[2]);assert.equal(ports[1].terminated,1);
});
test('current identity, complete arrays and point rational shapes reject corrupt responses',()=>{
 for(const input of normal)for(const mode of ['key','kind','input','missing','row','bounds','cycle','resolution','version','current']){
  const {controller,ports}=setup();controller.run(input);const r=structuredClone(evaluate(ports[0].messages[0]));assert.ok(r.ok);const v=r as unknown as Record<string,unknown>,result=r.result as unknown as Record<string,unknown>;
  if(mode==='key')v.inputKey='wrong';if(mode==='kind')v.kind='unknown';if(mode==='input')v.input=normal[(normal.indexOf(input)+1)%3];if(mode==='missing')delete result.approximationContract;
  const matrices=input.kind===kinds[1]?result.corrected as Record<string,unknown>:result;
  if(mode==='row')matrices.transition=[null];if(mode==='bounds')((matrices.transition as {bounds:{lower:{denominator:{coefficient:string}}}}[][])[0][0]).bounds.lower.denominator.coefficient='0';if(mode==='cycle')result.extra=result;if(mode==='resolution')result.fullyResolved=false;if(mode==='version')v.version=2;
  if(mode==='current'){if(input.kind===kinds[1])delete (result.corrected as Record<string,unknown>).jumpTargetDistance;else result.current={};}
  ports[0].emit(r);const state=controller.getSnapshot();assert.ok(state.status==='error',`${input.kind}.${mode}`);assert.equal(state.error,'invalid-response');assert.equal(ports[0].terminated,1);
 }
});
test('exact point observations do not admit precision ceilings, infinity, signed zero or mismatched endpoints',()=>{
 const r=evaluate({version:1,requestId:1,input:cycle});assert.ok(r.ok&&r.kind===kinds[0]);const q=r.result.stationaryReference[0];assert.ok(observation(q));
 for(const mode of ['resolution','infinity','denominator','digits','exponent','sign','negative-zero','point']){const p=structuredClone(q);if(mode==='resolution')p.resolution='unresolved-at-precision-ceiling';if(mode==='infinity')p.value=Infinity;if(mode==='denominator')p.bounds!.lower.denominator.coefficient='0';if(mode==='digits')p.bounds!.lower.numerator.coefficient='1'.repeat(4097);if(mode==='exponent')p.bounds!.lower.numerator.exponent=1;if(mode==='sign')p.sign=-1;if(mode==='negative-zero')p.bounds!.lower.numerator.coefficient='-0';if(mode==='point')p.bounds!.upper={...p.bounds!.upper,numerator:{coefficient:'2',exponent:0}};assert.equal(observation(p),false,mode);}
 const invalid=structuredClone(r);invalid.requestId=0;assert.equal(valid(invalid,cycle,0),false);
});
test('absent acceptance, unselected equilibrium and zero-support reverse rows retain their null meaning',()=>{
 const oneWay:DetailedBalanceInput={...hastings,parameters:{...hastings.parameters,forwardWeight:1,backwardWeight:0,stayWeight:0}};const r=evaluate({version:1,requestId:1,input:oneWay});assert.ok(r.ok&&r.kind===kinds[1]);assert.ok(valid(r,oneWay,1));const q=r.result.corrected.totalHoldingByState[0];r.result.corrected.acceptance[0][2]=q;assert.equal(valid(r,oneWay,1),false);
 const identity:DetailedBalanceInput={...binary,parameters:{...binary.parameters,forwardNumerator:0,backwardNumerator:0}};const a=evaluate({version:1,requestId:1,input:identity});assert.ok(a.ok&&a.kind===kinds[2]);assert.ok(valid(a,identity,1));a.result.stationaryDistribution=a.result.initial;assert.equal(valid(a,identity,1),false);
 const absorbing:DetailedBalanceInput={...binary,parameters:{...binary.parameters,forwardNumerator:0,backwardNumerator:1}};const b=evaluate({version:1,requestId:1,input:absorbing});assert.ok(b.ok&&b.kind===kinds[2]);assert.ok(valid(b,absorbing,1));b.result.stationaryTimeReversal![1]=b.result.transition[1];assert.equal(valid(b,absorbing,1),false);
});
test('record-order and mixing-status structure must agree with the requested scope',()=>{
 for(const mode of ['index','copy','status','crossing']){const r=evaluate({version:1,requestId:1,input:binary});assert.ok(r.ok&&r.kind===kinds[2]);if(mode==='index')r.result.history[0].step=1;if(mode==='copy')r.result.current=structuredClone(r.result.history[0]);if(mode==='status')r.result.mixingAtQuarter.status='never-periodic';if(mode==='crossing')r.result.mixingAtQuarter.firstStep=null;assert.equal(valid(r,binary,1),false,mode);}
});
test('creation, posting, execution, malformed acknowledgements and deserialization failures terminate without fallback',()=>{
 const controller=new Controller(()=>{throw Error('blocked');},()=>{});controller.run(cycle);assert.equal(controller.getSnapshot().status,'error');
 for(const mode of ['post','error','messageerror','model','ack']){const port=new Port();port.postFailure=mode==='post';const c=new Controller(()=>port,()=>{});c.run(cycle);if(mode==='error')port.onerror!.call(port as unknown as Worker,new Event('error') as ErrorEvent);if(mode==='messageerror')port.onmessageerror!.call(port as unknown as Worker,new MessageEvent('messageerror'));if(mode==='model')port.emit({version:1,requestId:1,ok:false,error:'calculation-failed'});if(mode==='ack')port.emit({version:1,requestId:1,phase:'started',kind:kinds[0],extra:true});assert.equal(c.getSnapshot().status,'error');assert.equal(port.terminated,1);}
});
test('cancellation, disposal and reentrant publication own each port exactly once',()=>{
 const {controller,ports,states}=setup();controller.run(cycle);controller.cancel();assert.equal(controller.getSnapshot().status,'idle');assert.equal(ports[0].terminated,1);controller.run(hastings);controller.dispose();controller.dispose();assert.equal(ports[1].terminated,1);const count=states.length;controller.run(binary);controller.cancel();assert.equal(states.length,count);
 let factories=0;let c:Controller;c=new Controller(()=>{factories++;return new Port();},s=>{if(s.status==='pending')c.cancel();});c.run(cycle);assert.equal(factories,0);
});
test('SSR of valid, null and disabled hook inputs never accesses Worker or performs numerical dispatch',()=>{
 const original=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('SSR worker');}});
 function Probe({input,enabled=true}:{input:DetailedBalanceInput|null;enabled?:boolean}){const state=useDetailedBalanceCalculation(input,enabled);assert.equal(state.status,'idle');assert.equal(state.response,null);return React.createElement('output',null,state.status);}
 try{for(const input of [...normal,null])renderToStaticMarkup(React.createElement(Probe,{input}));renderToStaticMarkup(React.createElement(Probe,{input:cycle,enabled:false}));assert.equal(reads,0);}finally{if(original)Object.defineProperty(globalThis,'Worker',original);else Reflect.deleteProperty(globalThis,'Worker');}
});
test('main hook graph contains controller/factory but no numerical models or exact arithmetic',async()=>{
 const r=await build({entryPoints:['src/components/learning/use-detailed-balance-calculation.ts'],bundle:true,write:false,format:'esm',platform:'browser',metafile:true,external:['react'],logLevel:'silent'});
 const paths=Object.keys(r.metafile!.inputs);for(const name of ['learning-stationary-cycle.ts','learning-hastings-proposal.ts','learning-two-state-mixing.ts','learning-finite-markov-exact.ts','learning-binary-products.ts','learning-rational-gcd.ts','learning-detailed-balance-evaluator.ts'])assert.ok(!paths.some(p=>p.endsWith(name)),name);assert.ok(paths.some(p=>p.endsWith('learning-detailed-balance-worker-factory.ts')));
 mkdirSync('.runtime/learning/detailed-balance-worker',{recursive:true});writeFileSync('.runtime/learning/detailed-balance-worker/hook-metafile.json',JSON.stringify(r.metafile,null,2)+'\n');
});
test('native Node threads cancel on genuine started ACKs and fresh requests recover in every family',async()=>{
 const built=await build({entryPoints:['src/lib/learning-detailed-balance.worker.ts'],bundle:true,write:false,format:'cjs',platform:'node',target:'es2022',logLevel:'silent'});
 const code=`const {parentPort}=require('node:worker_threads');globalThis.self={postMessage(v){parentPort.postMessage(v)},set onmessage(fn){parentPort.on('message',data=>fn({data}))}};\n${built.outputFiles[0].text}`;
 const records:unknown[]=[];
 for(let i=0;i<3;i++){
  let cancelAtStart=true;const ports:{port:DetailedBalanceWorkerPort;thread:ThreadWorker;ended:Promise<number>;terminated:number}[]=[],states:DetailedBalanceCalculationState[]=[];
  let successResolve:()=>void=()=>{},cancelResolve:()=>void=()=>{};
  const cancelled=new Promise<void>(resolve=>{cancelResolve=resolve;});
  const c=new Controller(()=>{const thread=new ThreadWorker(code,{eval:true});const item={port:null as unknown as DetailedBalanceWorkerPort,thread,ended:Promise.resolve(0),terminated:0};item.ended=new Promise(resolve=>thread.once('exit',resolve));const port:DetailedBalanceWorkerPort={onmessage:null,onerror:null,onmessageerror:null,postMessage:v=>thread.postMessage(v),terminate(){item.terminated++;void thread.terminate();}};item.port=port;thread.on('message',data=>port.onmessage?.call(port as unknown as Worker,new MessageEvent('message',{data})));thread.on('error',()=>port.onerror?.call(port as unknown as Worker,new Event('error') as ErrorEvent));ports.push(item);return port;},state=>{states.push(state);if(cancelAtStart&&state.status==='pending'&&state.phase==='calculating'){c.cancel();cancelResolve();}if(state.status==='ready')successResolve();});
  try{
   c.run(heavy[i]);await Promise.race([cancelled,new Promise((_,reject)=>{const t=setTimeout(()=>reject(Error('started timeout')),30000);t.unref();})]);await ports[0].ended;assert.equal(c.getSnapshot().status,'idle');assert.equal(states.filter(s=>s.status==='ready').length,0);
   cancelAtStart=false;const success=new Promise<void>(resolve=>{successResolve=resolve;});c.run(normal[i]);await Promise.race([success,new Promise((_,reject)=>{const t=setTimeout(()=>reject(Error('recovery timeout')),30000);t.unref();})]);await ports[1].ended;const s=c.getSnapshot();assert.ok(s.status==='ready');assert.ok(valid(s.response,normal[i],2));assert.ok(ports.every(p=>p.terminated===1));records.push({kind:kinds[i],realStartedAckCancellation:true,readyBeforeCancellation:0,freshThreadRecovery:true,ownedPorts:ports.length});
  }finally{c.dispose();await Promise.all(ports.map(async p=>{await p.thread.terminate();await p.ended;}));}
 }
 writeFileSync('.runtime/learning/detailed-balance-worker/node-verification.json',JSON.stringify({scope:'actual Node worker entry with started-ACK-triggered controller cancellation; no claim of CPU interruption timing for the short Hastings calculation',records},null,2)+'\n');
});
