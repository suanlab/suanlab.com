import {observe,f,q as exactObservation} from '../src/lib/learning-quantum-hilbert-core';
import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {build} from 'esbuild';import {Worker} from 'node:worker_threads';
import {evaluateSpinRequest as evaluate} from '../src/lib/learning-spin-evaluator';
import {SpinCalculationController as Controller,type SpinWorkerPort} from '../src/lib/learning-spin-controller';
import {spinKinds as kinds,snapshotSpinInput as snapshot,spinInputKey as key,isSpinInput as valid,isSpinSuccessForInput as success,parseSpinInputDraft as draft,parseSpinStoredInput as stored,type SpinInput} from '../src/lib/learning-spin-protocol';
const families=kinds.map((kind,i)=>({kind,cases:JSON.parse(readFileSync(`scripts/fixtures/spin-${i===0?'half':'ladder'}-reference.json`,'utf8')).cases})) as {kind:typeof kinds[number];cases:{label:string;input:SpinInput['parameters']}[]}[];
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input} as SpinInput));
for(const family of families)for(const c of family.cases)test(`Spin transport: ${family.kind}: ${c.label}`,()=>{const input=snapshot({kind:family.kind,parameters:c.input} as SpinInput),r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(success(structuredClone(r),input,7),c.label);assert(!success(r,input,8));});
class Port implements SpinWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
test('caller mutations cannot alter the active immutable preparation or its request key',()=>{
 const value=structuredClone(inputs[0]),before=key(value),port=new Port(),c=new Controller(()=>port,()=>{});c.run(value);value.parameters.stateReal0=17;assert.equal(c.getSnapshot().inputKey,before);assert(Object.isFrozen(c.getSnapshot().input!.parameters));port.emit(evaluate(port.sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(c.getSnapshot().inputKey,before);assert.equal(port.terminated,1);c.dispose();
});
test('owned failure frames map to explicit errors and malformed current frames terminate the port',()=>{
 for(const[frame,error]of [[{version:1,requestId:1,ok:false,error:'invalid-input'},'invalid-input'],[{version:1,requestId:1,ok:false,error:'calculation-failed'},'worker-failed'],[{version:1,requestId:1,ok:false,error:'unknown'},'invalid-response'],[{version:1,requestId:1,ok:false,error:'invalid-input',extra:true},'invalid-response'],[{version:2,requestId:1,ok:true},'invalid-response'],[null,'invalid-response']]as const){const p=new Port(),c=new Controller(()=>p,()=>{});c.run(inputs[0]);p.emit(frame);const state=c.getSnapshot();assert.equal(state.status,'error');assert(state.status==='error');assert.equal(state.error,error);assert.equal(p.terminated,1);assert.equal(p.onmessage,null);assert.equal(p.onerror,null);assert.equal(p.onmessageerror,null);c.dispose();}
});
test('replacement, stale callbacks, cancellation and final completion release exactly one worker',()=>{
 const ports:Port[]=[],states:string[]=[];const c=new Controller(()=>{const p=new Port();ports.push(p);return p;},s=>states.push(s.status));const id=c.run(inputs[0])!,stale=ports[0].onmessage!;c.run(inputs[1]);assert.equal(ports[0].terminated,1);assert.equal(ports[0].onmessage,null);stale({data:evaluate({version:1,requestId:id,input:inputs[0]})}as MessageEvent);assert.equal(c.getSnapshot().status,'pending');ports[1].emit({version:1,requestId:id,phase:'started',kind:inputs[0].kind});assert.equal(c.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(ports[1].terminated,1);c.cancel();assert.equal(c.getSnapshot().status,'idle');c.dispose();assert.equal(c.run(inputs[0]),null);assert.deepEqual(states,['pending','pending','ready','idle']);
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
test('reentrant ready notification can launch the next request without reviving a completed port',()=>{
 const ports:Port[]=[];let c:Controller,replaced=false;c=new Controller(()=>{const p=new Port();ports.push(p);return p;},s=>{if(s.status==='ready'&&!replaced){replaced=true;c.run(inputs[1]);}});c.run(inputs[0]);const late=ports[0].onmessage!;ports[0].emit(evaluate(ports[0].sent[0]));assert.equal(ports[0].terminated,1);assert.equal(c.getSnapshot().status,'pending');late({data:evaluate(ports[0].sent[0])}as MessageEvent);assert.equal(c.getSnapshot().status,'pending');ports[1].emit(evaluate(ports[1].sent[0]));assert.equal(c.getSnapshot().status,'ready');assert(ports.every(p=>p.terminated===1));c.dispose();
});
test('envelopes and mixed families fail before calculation',()=>{
 for(const value of [null,{}, {version:2,requestId:1,input:inputs[0]}, {version:1,requestId:0,input:inputs[0]}, {version:1,requestId:1,input:inputs[0],extra:1}])assert.equal(evaluate(value).ok,false);
 for(const input of [{kind:kinds[0],parameters:inputs[1].parameters},{kind:'unknown',parameters:inputs[0].parameters}])assert.deepEqual(evaluate({version:1,requestId:1,input}),{version:1,requestId:1,ok:false,error:'invalid-input'});
});

test('reader dependency graph omits calculation engines; actual worker entry emits started and validated results',async()=>{
 const reader=await build({entryPoints:['src/components/learning/use-spin-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(spin-half|spin-ladder|spin-angle|quantum-evolution-interval|observable-matrix|quantum-hilbert-core)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-spin.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});
test('React server rendering starts no Worker and exposes no response before mounting',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useSpinCalculation}=await import('../src/components/learning/use-spin-calculation');
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('Worker unavailable during SSR');}});
 try{for(const input of [...inputs,null]){const Probe=()=>{const state=useSpinCalculation(input);assert.equal(state.response,null);assert.equal(state.requestId,null);return React.createElement('output',null,state.status);};assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');}assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});


test('spin input identity, lexical integers and storage reject rounded or foreign preparations',()=>{
 for(const input of inputs){
  assert.equal(key(input),key({...input,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())} as SpinInput));
  assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);
  assert.equal(stored(JSON.stringify({version:2,value:input})),null);assert.equal(stored(JSON.stringify({version:1,value:input,extra:1})),null);
  assert(Object.isFrozen(snapshot(input)));assert(Object.isFrozen(snapshot(input).parameters));
  for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){const raw=JSON.stringify(input).replace(/"hbarNumerator":\d+/,'"hbarNumerator":'+token);assert.equal(draft(raw),null);assert.equal(stored('{"version":1,"value":'+raw+'}'),null);}
  assert(!valid({...input,extra:1}));assert(!valid({...input,[Symbol('foreign')]:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('foreign')]:1}}));assert(!valid(Object.assign(Object.create(input),{})));
 }
 const p=structuredClone(inputs[1]);assert(p.kind===kinds[1]);p.parameters.stateReal8=1;assert(!valid(p));
 assert.equal(draft(' '.repeat(32769)),null);assert.equal(draft('not-json'),null);assert.equal(stored('{}'),null);
});
const responses=inputs.map(input=>{const r=evaluate({version:1,requestId:1,input});assert(r.ok);return r;});
function rejectMutation(family:number,change:(r:any)=>void){const x=structuredClone(responses[family]);change(x);assert(!success(x,inputs[family],1));}
test('spin frames require every root field and owned complete nested collections',()=>{
 for(const[i,r]of responses.entries()){
  for(const k of Object.keys(r))rejectMutation(i,x=>{delete x[k];});
  for(const k of Object.keys(r.result))rejectMutation(i,x=>{delete x.result[k];});
  rejectMutation(i,x=>{x.extra=1;});rejectMutation(i,x=>{x.result.extra=1;});rejectMutation(i,x=>{x.result[Symbol('foreign')]=1;});rejectMutation(i,x=>{x.inputKey+=' ';});rejectMutation(i,x=>{x.result.parameters.stateReal0++;});rejectMutation(i,x=>{x.result.precisionBits=53;});rejectMutation(i,x=>{x.kind=kinds[1-i];});
 }
 for(const k of Object.keys((responses[0].result as any).records[1]))rejectMutation(0,x=>{delete x.result.records[1][k];});
 rejectMutation(0,x=>{delete x.result.records[0];});rejectMutation(0,x=>{x.result.records.extra=true;});rejectMutation(0,x=>{delete x.result.records[0].propagator[0][0];});rejectMutation(0,x=>{x.result.records[0].amplitudes[Symbol('foreign')]=1;});
 for(const k of Object.keys((responses[1].result as any).state))rejectMutation(1,x=>{delete x.result.state[k];});
 for(const k of Object.keys((responses[1].result as any).raising))rejectMutation(1,x=>{delete x.result.raising[k];});
 rejectMutation(1,x=>{delete x.result.physicalMatrices.x[0][0];});rejectMutation(1,x=>{x.result.raisingChain.extra=1;});rejectMutation(1,x=>{delete x.result.raisingChain[0];});
});
test('spin observation certificates reject false coordinates, malformed bounds and hidden fields',()=>{
 for(let i=0;i<2;i++)for(const change of [(q:any)=>{q.value=123;},(q:any)=>{q.value=NaN;},(q:any)=>{q.bounds.upper.denominator.coefficient='0';},(q:any)=>{q.bounds.lower.numerator.coefficient='1'.repeat(4097);},(q:any)=>{q.bounds.upper.numerator.exponent=1;},(q:any)=>{q.bounds.extra=1;},(q:any)=>{q.bounds.upper.numerator[Symbol('foreign')]=1;},(q:any)=>{q.logMagnitude=9999;},(q:any)=>{q.extra=1;}])rejectMutation(i,x=>change(i===0?x.result.records[1].amplitudes[0].real:x.result.state.physicalAmplitudes[0].real));
});
test('spin-half attribution retains signed angle order, pure-spin units and initial Born accounts',()=>{
 rejectMutation(0,x=>{x.result.records.reverse();});rejectMutation(0,x=>{x.result.records[1].turns=x.result.records[0].turns;});rejectMutation(0,x=>{x.result.records[1].fraction=x.result.records[0].fraction;});
 rejectMutation(0,x=>{x.result.hbar=exactObservation(f(BigInt(2)));});rejectMutation(0,x=>{x.result.rawNormSquared=exactObservation(f(BigInt(2)));});rejectMutation(0,x=>{x.result.casimirEigenvalue=x.result.hbar;});
 rejectMutation(0,x=>{x.result.initialMeasurement.outcomes.reverse();});rejectMutation(0,x=>{x.result.initialMeasurement.outcomes[1].conditionalAmplitudes=null;});rejectMutation(0,x=>{x.result.initialMeasurement.outcomes[0].probability=x.result.initialMeasurement.outcomes[1].probability;});
 rejectMutation(0,x=>{x.result.initialBloch.reverse();});rejectMutation(0,x=>{x.result.classification.physicalTimeSpecified=true;});rejectMutation(0,x=>{x.result.classification.integerEndpointSpinorSign=1;});rejectMutation(0,x=>{x.result.records[0].residuals.norm=x.result.rawNormSquared;});
});
test('ladder attribution requires the selected metric, Casimir, spectrum and nonprobabilistic chain units',()=>{
 rejectMutation(1,x=>{x.result.dimension++;});rejectMutation(1,x=>{x.result.j=exactObservation(f(BigInt(2)));});rejectMutation(1,x=>{x.result.metricWeights[0]=exactObservation(f(BigInt(2)));});rejectMutation(1,x=>{x.result.binomialCoefficients[1]=exactObservation(f(BigInt(1)));});
 rejectMutation(1,x=>{x.result.state.zProbabilities=x.result.state.naiveEuclideanZProbabilities;});rejectMutation(1,x=>{x.result.state.spectrum.reverse();});rejectMutation(1,x=>{x.result.casimirEigenvalue=x.result.hbar;});rejectMutation(1,x=>{x.result.state.moments.x.variance=x.result.hbar;});
 rejectMutation(1,x=>{x.result.raising.normalizedPhysicalAmplitudes=null;});rejectMutation(1,x=>{x.result.raising.actionNormSquared=x.result.hbar;});rejectMutation(1,x=>{x.result.raisingChain.reverse();});rejectMutation(1,x=>{x.result.raisingChain[1].normActionPower=0;});rejectMutation(1,x=>{x.result.raisingChain[0].weightedNumerators[0].real=exactObservation(f(BigInt(2)));});
 rejectMutation(1,x=>{x.result.ladderCoefficients[1].raisingSquaredCoefficient=x.result.hbar;});rejectMutation(1,x=>{x.result.classification.ladderActionNormIsProbability=true;});rejectMutation(1,x=>{x.result.classification.rawLadderChainRenormalized=true;});rejectMutation(1,x=>{x.result.physicalResiduals.norm=x.result.hbar;});
});

test('spin wire intervals retain the numerical width contract rather than accepting a broad residual enclosure',()=>{
 const broad=observe({lower:f(BigInt(-1)),upper:f(BigInt(1))});
 rejectMutation(0,x=>{x.result.records[0].residuals.norm=broad;});
 rejectMutation(1,x=>{x.result.physicalResiduals.norm=broad;});
});
