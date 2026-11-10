import {observe,f,q as exactObservation} from '../src/lib/learning-quantum-hilbert-core';
import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {build} from 'esbuild';import {Worker} from 'node:worker_threads';
import {evaluateDensityRequest as evaluate} from '../src/lib/learning-density-evaluator';
import {DensityCalculationController as Controller,type DensityWorkerPort} from '../src/lib/learning-density-controller';
import {densityKinds as kinds,snapshotDensityInput as snapshot,densityInputKey as key,isDensityInput as valid,isDensitySuccessForInput as success,parseDensityInputDraft as draft,parseDensityStoredInput as stored,type DensityInput} from '../src/lib/learning-density-protocol';
const families=kinds.map((kind,i)=>({kind,cases:JSON.parse(readFileSync(`scripts/fixtures/density-${i===0?'ensemble':'spectrum'}-reference.json`,'utf8')).cases})) as {kind:typeof kinds[number];cases:{label:string;input:DensityInput['parameters']}[]}[];
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input} as DensityInput));
for(const family of families)for(const c of family.cases)test(`Density transport: ${family.kind}: ${c.label}`,()=>{const input=snapshot({kind:family.kind,parameters:c.input} as DensityInput),r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(success(structuredClone(r),input,7),c.label);assert(!success(r,input,8));});
class Port implements DensityWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
test('caller mutations cannot alter the active immutable preparation or its request key',()=>{
 const value=structuredClone(inputs[0]),before=key(value),port=new Port(),c=new Controller(()=>port,()=>{});c.run(value);if(value.kind===kinds[0])value.parameters.component0Real0=17;assert.equal(c.getSnapshot().inputKey,before);assert(Object.isFrozen(c.getSnapshot().input!.parameters));port.emit(evaluate(port.sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(c.getSnapshot().inputKey,before);assert.equal(port.terminated,1);c.dispose();
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
 const reader=await build({entryPoints:['src/components/learning/use-density-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(density-ensemble|density-spectrum|density-spectrum-math|quantum-evolution-interval|observable-matrix|quantum-hilbert-core)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-density.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});
test('React server rendering starts no Worker and exposes no response before mounting',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useDensityCalculation}=await import('../src/components/learning/use-density-calculation');
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('Worker unavailable during SSR');}});
 try{for(const input of [...inputs,null]){const Probe=()=>{const state=useDensityCalculation(input);assert.equal(state.response,null);assert.equal(state.requestId,null);return React.createElement('output',null,state.status);};assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');}assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});


test('density input identity, lexical integers and storage reject rounded or foreign preparations',()=>{
 for(const input of inputs){
  assert.equal(key(input),key({...input,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())} as DensityInput));
  assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);
  assert.equal(stored(JSON.stringify({version:2,value:input})),null);assert.equal(stored(JSON.stringify({version:1,value:input,extra:1})),null);
  assert(Object.isFrozen(snapshot(input)));assert(Object.isFrozen(snapshot(input).parameters));
  for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){const raw=JSON.stringify(input).replace(/"probeReal0":-?\d+/,'"probeReal0":'+token);assert.equal(draft(raw),null);assert.equal(stored('{"version":1,"value":'+raw+'}'),null);}
  assert(!valid({...input,extra:1}));assert(!valid({...input,[Symbol('foreign')]:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('foreign')]:1}}));assert(!valid(Object.assign(Object.create(input),{})));
 }
 const p=structuredClone(inputs[1]);assert(p.kind===kinds[1]);p.parameters.blochX=1000000;assert(!valid(p));
 assert.equal(draft(' '.repeat(32769)),null);assert.equal(draft('not-json'),null);assert.equal(stored('{}'),null);
});
const attributedInputs=[snapshot({kind:kinds[0],parameters:families[0].cases.find(c=>c.label==='unequal nonorthogonal mixture')!.input}as DensityInput),snapshot({kind:kinds[1],parameters:families[1].cases.find(c=>c.label==='phase flip endpoint')!.input}as DensityInput)];
const responses=attributedInputs.map(input=>{const r=evaluate({version:1,requestId:1,input});assert(r.ok);return r;});
function rejectMutation(family:number,change:(r:any)=>void){const x=structuredClone(responses[family]);change(x);assert(!success(x,attributedInputs[family],1));}

test('density frames require every owned root and nested field, complete arrays and matching family',()=>{
 for(const[i,r]of responses.entries()){
  for(const k of Object.keys(r))rejectMutation(i,x=>{delete x[k];});
  for(const k of Object.keys(r.result))rejectMutation(i,x=>{delete x.result[k];});
  rejectMutation(i,x=>{x.extra=1;});rejectMutation(i,x=>{x.result.extra=1;});rejectMutation(i,x=>{x.result[Symbol('foreign')]=1;});rejectMutation(i,x=>{x.inputKey+=' ';});rejectMutation(i,x=>{x.result.parameters.probeReal0++;});rejectMutation(i,x=>{x.result.precisionBits=53;});rejectMutation(i,x=>{x.kind=kinds[1-i];});
 }
 for(const k of Object.keys((responses[0].result as any).components[0]))rejectMutation(0,x=>{delete x.result.components[0][k];});
 for(const k of Object.keys((responses[0].result as any).observable))rejectMutation(0,x=>{delete x.result.observable[k];});
 for(const k of Object.keys((responses[1].result as any).records[0]))rejectMutation(1,x=>{delete x.result.records[0][k];});
 rejectMutation(0,x=>{delete x.result.density[0][0];});rejectMutation(0,x=>{x.result.components.extra=1;});rejectMutation(0,x=>{x.result.weightedColumns[0][Symbol('foreign')]=1;});
 rejectMutation(1,x=>{delete x.result.records[0];});rejectMutation(1,x=>{x.result.records.extra=true;});rejectMutation(1,x=>{x.result.records[0].channel.choi[0][Symbol('foreign')]=1;});
});
test('density observation certificates reject malformed fractions, forged drawing values and foreign fields',()=>{
 for(let i=0;i<2;i++)for(const change of [(q:any)=>{q.value=123;},(q:any)=>{q.value=NaN;},(q:any)=>{q.bounds.upper.denominator.coefficient='0';},(q:any)=>{q.bounds.lower.numerator.coefficient='1'.repeat(4097);},(q:any)=>{q.bounds.upper.numerator.exponent=1;},(q:any)=>{q.bounds.extra=1;},(q:any)=>{q.bounds.upper.numerator[Symbol('foreign')]=1;},(q:any)=>{q.logMagnitude=9999;},(q:any)=>{q.extra=1;}])rejectMutation(i,x=>change(i===0?x.result.density[0][0].real:x.result.records[0].spectrum[1].eigenvalue));
 const broad=observe({lower:f(BigInt(-1)),upper:f(BigInt(1))});rejectMutation(0,x=>{x.result.directedResiduals.gramTrace.real=broad;});rejectMutation(1,x=>{x.result.records[0].directedResiduals.coherenceEntropyAccount=broad;});
});
test('ensemble attribution rejects a different density, raw norm, purity, Gram rank and tomography sign',()=>{
 const changed=exactObservation(f(BigInt(1),BigInt(2)));
 rejectMutation(0,x=>{x.result.density[0][0].real=changed;});rejectMutation(0,x=>{x.result.weights.reverse();});rejectMutation(0,x=>{x.result.components[0].rawNormSquared=changed;});rejectMutation(0,x=>{x.result.components[0].occupied=false;});rejectMutation(0,x=>{x.result.components[0].pureDensity[0][0].real=changed;});
 rejectMutation(0,x=>{x.result.purity=x.result.standardBasisDephasing.purity;});rejectMutation(0,x=>{x.result.rank=1;});rejectMutation(0,x=>{x.result.gramRank=1;});rejectMutation(0,x=>{x.result.classification.allOccupiedComponentsSameRay=true;});rejectMutation(0,x=>{x.result.standardProbabilities.reverse();});
 rejectMutation(0,x=>{x.result.tomography.pairs[0].probabilities.xPlus=x.result.tomography.pairs[0].probabilities.xMinus;});rejectMutation(0,x=>{x.result.tomography.pairs[0].i=1;});rejectMutation(0,x=>{x.result.tomography.reconstructedDensity[0][1].real=changed;});rejectMutation(0,x=>{x.result.characteristicPolynomial.gram[4]=changed;});
 rejectMutation(0,x=>{x.result.observable.mean=changed;});rejectMutation(0,x=>{x.result.observable.matrix[0][0].real=changed;});rejectMutation(0,x=>{x.result.observable.outcomes.reverse();});rejectMutation(0,x=>{x.result.observable.withinEnsembleVariance=changed;});rejectMutation(0,x=>{x.result.probeComparison.pureDensity[0][0].real=changed;});
});
test('ensemble structural nulls and merged outcomes retain their mathematical meanings',()=>{
 for(const label of ['zero outcome','dimension 1 pure','dimension 3 scalar measurement','degenerate mixed conditional','zero weight nonzero component']){
  const input=snapshot({kind:kinds[0],parameters:families[0].cases.find(c=>c.label===label)!.input}as DensityInput),response=evaluate({version:1,requestId:1,input});assert(response.ok&&response.kind===kinds[0]);assert(success(response,input,1));
  const reject=(change:(x:any)=>void)=>{const x=structuredClone(response);change(x);assert(!success(x,input,1));};
  if(label==='zero outcome')reject(x=>{const o=x.result.observable.outcomes.find((o:any)=>o.probability.sign===0);o.conditionalDensity=x.result.density;});
  if(label==='dimension 1 pure'||label==='dimension 3 scalar measurement')reject(x=>{x.result.observable.outcomes.push(structuredClone(x.result.observable.outcomes[0]));});
  if(label==='degenerate mixed conditional')reject(x=>{x.result.observable.outcomes[0].rank=1;});
  if(label==='zero weight nonzero component')reject(x=>{x.result.components[0].pureDensity=null;});
 }
});
test('spectrum attribution preserves signed eta order, exact Choi convention and input-bound populations',()=>{
 const changed=exactObservation(f(BigInt(1),BigInt(2)));
 rejectMutation(1,x=>{x.result.records.reverse();});rejectMutation(1,x=>{x.result.records[1].eta=x.result.records[0].eta;});rejectMutation(1,x=>{x.result.records[1].fraction=x.result.records[0].fraction;});rejectMutation(1,x=>{x.result.initialBloch[0]=changed;});
 rejectMutation(1,x=>{x.result.records[0].density[0][0].real=changed;x.result.records[0].density[0][1].imaginary=changed;});rejectMutation(1,x=>{x.result.records[0].purity=changed;});rejectMutation(1,x=>{x.result.records[0].measurement.matrix[0][0].real=changed;});rejectMutation(1,x=>{x.result.records[0].measurement.probabilities[0]=exactObservation(f(BigInt(1)));});
 rejectMutation(1,x=>{x.result.records[0].channel.choiNormalization='trace=1';});rejectMutation(1,x=>{x.result.records[0].channel.choi[0][0].real=changed;});rejectMutation(1,x=>{x.result.records[0].channel.identityWeight=changed;});rejectMutation(1,x=>{x.result.records[0].channel.choiSpectrum.reverse();});rejectMutation(1,x=>{x.result.records[0].channel.minimalKrausCount=2;});
 rejectMutation(1,x=>{x.result.classification.parameterPathIsTimeEvolution=true;});rejectMutation(1,x=>{x.result.classification.continuousParameterPathEntropyNondecreasing=true;});rejectMutation(1,x=>{x.result.records.at(-1).classification.selectedStateUnchanged=true;});
});
test('spectrum exact zero, degenerate eigenspaces and offgrid crossing cannot be relabeled by transport',()=>{
 for(const label of ['maximally mixed','offgrid zero crossing','minimal positive determinant','X measurement basis']){
  const input=snapshot({kind:kinds[1],parameters:families[1].cases.find(c=>c.label===label)!.input}as DensityInput),response=evaluate({version:1,requestId:1,input});assert(response.ok&&response.kind===kinds[1]);assert(success(response,input,1));
  const reject=(change:(x:any)=>void)=>{const x=structuredClone(response);change(x);assert(!success(x,input,1));};
  if(label==='maximally mixed'){reject(x=>{x.result.records[0].spectralDirection=[x.result.endpointEta,x.result.endpointEta,x.result.endpointEta];});reject(x=>{x.result.records[0].spectrum[0].rank=1;});}
  if(label==='offgrid zero crossing')reject(x=>{x.result.records=x.result.records.filter((r:any)=>r.eta.sign!==0);});
  if(label==='minimal positive determinant')reject(x=>{x.result.records[0].spectrum[0].logEigenvalue=null;});
  if(label==='X measurement basis')reject(x=>{x.result.records[0].spectrum[0].logEigenvalue=x.result.lnTwo;});
 }
});
