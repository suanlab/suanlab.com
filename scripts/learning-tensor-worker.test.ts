import {observe,f,q as exactObservation} from '../src/lib/learning-quantum-hilbert-core';
import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {build} from 'esbuild';import {Worker} from 'node:worker_threads';
import {evaluateTensorRequest as evaluate} from '../src/lib/learning-tensor-evaluator';
import {TensorCalculationController as Controller,type TensorWorkerPort} from '../src/lib/learning-tensor-controller';
import {tensorKinds as kinds,snapshotTensorInput as snapshot,tensorInputKey as key,isTensorInput as valid,isTensorSuccessForInput as success,parseTensorInputDraft as draft,parseTensorStoredInput as stored,type TensorInput} from '../src/lib/learning-tensor-protocol';
import {tensorMixedKeys,isTensorMixedParameters} from '../src/lib/learning-tensor-input';
import {mixedEntanglementKeys,isMixedEntanglementParameters} from '../src/lib/learning-mixed-entanglement-input';
const families=kinds.map((kind,i)=>({kind,cases:JSON.parse(readFileSync(`scripts/fixtures/${i===0?'tensor-pure':'mixed-entanglement'}-reference.json`,'utf8')).cases})) as {kind:typeof kinds[number];cases:{label:string;input:TensorInput['parameters']}[]}[];
const inputs=families.map(f=>snapshot({kind:f.kind,parameters:f.cases[0].input} as TensorInput));
for(const family of families)for(const c of family.cases)test(`Tensor transport: ${family.kind}: ${c.label}`,()=>{const input=snapshot({kind:family.kind,parameters:c.input} as TensorInput),r=evaluate({version:1,requestId:7,input});assert(r.ok);assert(success(structuredClone(r),input,7),c.label);assert(!success(r,input,8));});
class Port implements TensorWorkerPort{
 onmessage:((e:MessageEvent)=>void)|null=null;onerror:((e:ErrorEvent)=>void)|null=null;onmessageerror:((e:MessageEvent)=>void)|null=null;sent:unknown[]=[];terminated=0;
 postMessage(x:unknown){this.sent.push(x);}terminate(){this.terminated++;}emit(data:unknown){this.onmessage?.({data} as MessageEvent);}
}
test('caller mutations cannot alter the active immutable preparation or its request key',()=>{
 const value=structuredClone(inputs[0]),before=key(value),port=new Port(),c=new Controller(()=>port,()=>{});c.run(value);if(value.kind===kinds[0])value.parameters.stateReal00=17;assert.equal(c.getSnapshot().inputKey,before);assert(Object.isFrozen(c.getSnapshot().input!.parameters));port.emit(evaluate(port.sent[0]));assert.equal(c.getSnapshot().status,'ready');assert.equal(c.getSnapshot().inputKey,before);assert.equal(port.terminated,1);c.dispose();
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
 const reader=await build({entryPoints:['src/components/learning/use-tensor-calculation.ts'],bundle:true,write:false,metafile:true,platform:'browser',format:'esm',target:'es2020'}),bundled=Object.keys(reader.metafile!.inputs);
 assert(!bundled.some(p=>/learning-(tensor-pure|tensor-roots|mixed-entanglement(?:-input|-algebra)?|quantum-evolution-interval|observable-matrix|quantum-hilbert-core)\.ts|entropy-production-math|interval-transcendentals/.test(p)),bundled.join('\n'));
 const code=await build({entryPoints:['src/lib/learning-tensor.worker.ts'],bundle:true,write:false,platform:'browser',format:'iife',target:'es2020'});
 const worker=new Worker(`const{parentPort}=require('node:worker_threads');globalThis.self={postMessage:x=>parentPort.postMessage(x)};${code.outputFiles[0].text}\nparentPort.on('message',data=>self.onmessage({data}));`,{eval:true});
 try{for(let i=0;i<inputs.length;i++){
  const messages:unknown[]=[];const done=new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(Error('worker timeout')),15000);const error=(e:Error)=>{clearTimeout(timer);reject(e);};const receive=(message:unknown)=>{messages.push(message);if(messages.length===2){clearTimeout(timer);worker.off('message',receive);worker.off('error',error);resolve();}};worker.on('message',receive);worker.once('error',error);});worker.postMessage({version:1,requestId:i+1,input:inputs[i]});await done;assert.deepEqual(messages[0],{version:1,requestId:i+1,phase:'started',kind:inputs[i].kind});assert(success(messages[1],inputs[i],i+1));
 }}finally{await worker.terminate();}
});
test('React server rendering starts no Worker and exposes no response before mounting',async()=>{
 const React=await import('react'),{renderToStaticMarkup}=await import('react-dom/server'),{useTensorCalculation}=await import('../src/components/learning/use-tensor-calculation');
 const previous=Object.getOwnPropertyDescriptor(globalThis,'Worker');let reads=0;Object.defineProperty(globalThis,'Worker',{configurable:true,get(){reads++;throw Error('Worker unavailable during SSR');}});
 try{for(const input of [...inputs,null]){const Probe=()=>{const state=useTensorCalculation(input);assert.equal(state.response,null);assert.equal(state.requestId,null);return React.createElement('output',null,state.status);};assert.equal(renderToStaticMarkup(React.createElement(Probe)),'<output>idle</output>');}assert.equal(reads,0);}finally{if(previous)Object.defineProperty(globalThis,'Worker',previous);else Reflect.deleteProperty(globalThis,'Worker');}
});


test('tensor input identity, lexical integers and storage reject rounded or foreign preparations',()=>{
 for(const input of inputs){
  assert.equal(key(input),key({...input,parameters:Object.fromEntries(Object.entries(input.parameters).reverse())}as TensorInput));assert.deepEqual(draft(JSON.stringify(input)),input);assert.deepEqual(stored(JSON.stringify({version:1,value:input})),input);
  assert.equal(stored(JSON.stringify({version:2,value:input})),null);assert.equal(stored(JSON.stringify({version:1,value:input,extra:1})),null);assert(Object.isFrozen(snapshot(input)));assert(Object.isFrozen(snapshot(input).parameters));
  const field=input.kind===kinds[0]?'stateReal00':'hReal00';for(const token of ['1e-9999','1.00000000000000001','9007199254740993','1e9999']){const raw=JSON.stringify(input).replace(new RegExp('"'+field+'":-?\\d+'),'"'+field+'":'+token);assert.equal(draft(raw),null);assert.equal(stored('{"version":1,"value":'+raw+'}'),null);}
  assert(!valid({...input,extra:1}));assert(!valid({...input,[Symbol('foreign')]:1}));assert(!valid({...input,parameters:{...input.parameters,[Symbol('foreign')]:1}}));assert(!valid(Object.assign(Object.create(input),{})));
 }
 assert.equal(draft(' '.repeat(32769)),null);assert.equal(draft('not-json'),null);assert.equal(stored('{}'),null);
});

const attributedInputs=[snapshot({kind:kinds[0],parameters:families[0].cases.find(c=>c.label==='2x3 complex generic')!.input}as TensorInput),snapshot({kind:kinds[1],parameters:families[1].cases.find(c=>c.label==='general complex rank4 case0')!.input}as TensorInput)];
const responses=attributedInputs.map(input=>{const r=evaluate({version:1,requestId:1,input});assert(r.ok);assert(success(r,input,1));return r;});
function rejectMutation(i:number,change:(r:any)=>void){const r=structuredClone(responses[i]);change(r);assert.notDeepEqual(r,responses[i],'Mutation must alter the message');assert(!success(r,attributedInputs[i],1));}
test('every tensor result root and representative nested object requires owned fields',()=>{
 for(const[i,r]of responses.entries()){
  for(const k of Object.keys(r))rejectMutation(i,v=>{delete v[k];});for(const k of Object.keys(r.result))rejectMutation(i,v=>{delete v.result[k];});
  rejectMutation(i,v=>{v.result.extra=1;});rejectMutation(i,v=>{v.extra=1;});rejectMutation(i,v=>{v.result[Symbol('foreign')]=1;});rejectMutation(i,v=>{v.kind=kinds[1-i];});rejectMutation(i,v=>{v.inputKey+=' ';});rejectMutation(i,v=>{v.result.precisionBits=53;});
 }
 const groups=[['reducedA','projectorTests','localReflection','basisDephasedReference','partialTranspose','classification'],['joint','reducedA','localTests','partialTranspose','chsh','classification']];
 for(const[i,names]of groups.entries())for(const name of names)for(const field of Object.keys((responses[i].result as any)[name]))rejectMutation(i,v=>{delete v.result[name][field];});
 rejectMutation(0,v=>{delete v.result.schmidtSpectrum[0].coefficient;});rejectMutation(0,v=>{v.result.rawCoefficients.extra=1;});rejectMutation(0,v=>{delete v.result.reducedB.density[0][0];});rejectMutation(1,v=>{v.result.localTests.pairs[0][0].extra=1;});rejectMutation(1,v=>{v.result.chsh.gramSpectrum[Symbol('foreign')]=1;});
});
test('tensor observations reject malformed fractions,nonfinite/forged projections and wide enclosures',()=>{
 for(let i=0;i<2;i++)for(const change of [(q:any)=>{q.value=123;},(q:any)=>{q.value=NaN;},(q:any)=>{q.bounds.upper.denominator.coefficient='0';},(q:any)=>{q.bounds.lower.numerator.coefficient='1'.repeat(4097);},(q:any)=>{q.bounds.upper.numerator.exponent=1;},(q:any)=>{q.bounds.extra=1;},(q:any)=>{q.bounds.upper.numerator[Symbol('foreign')]=1;},(q:any)=>{q.logMagnitude=9999;},(q:any)=>{q.extra=1;}])rejectMutation(i,v=>change(i===0?v.result.rawNormSquared:v.result.normalizationTrace));
 rejectMutation(0,v=>{v.result.directedResiduals.schmidtWeightMass=observe({lower:f(BigInt(-1)),upper:f(BigInt(1))});});rejectMutation(1,v=>{v.result.chsh.topTwoEigenvalueSum=observe({lower:f(BigInt(-1)),upper:f(BigInt(1))});});
});
test('pure attribution binds amplitudes,densities,Gram transpose,minors and dephased states',()=>{
 const q=exactObservation(f(BigInt(1),BigInt(3)));
 for(const mutate of [(r:any)=>{r.rawNormSquared=q;},(r:any)=>{r.rawCoefficients[0][0].real=q;},(r:any)=>{r.normalizedAmplitudes[0].real=q;},(r:any)=>{r.jointDensity[0][0].real=q;},(r:any)=>{r.reducedA.purity=q;},(r:any)=>{r.coefficientGramB[0][1].imaginary=q;},(r:any)=>{r.allTwoByTwoMinors[0].value.real=q;},(r:any)=>{r.entanglementWitnessMinor=null;},(r:any)=>{r.standardBasisProbabilities[0][0]=q;},(r:any)=>{r.basisDephasedReference.sameReducedStates=!r.basisDephasedReference.sameReducedStates;},(r:any)=>{r.basisDephasedReference.density[0][0].real=q;}])rejectMutation(0,v=>mutate(v.result));
});
test('pure attribution binds spectral multiplicities,root intervals,pair indexing and entropy accounts',()=>{
 const q=exactObservation(f(BigInt(1),BigInt(3)));
 rejectMutation(0,v=>{v.result.schmidtSpectrum[0].multiplicity++;});rejectMutation(0,v=>{v.result.schmidtSpectrum.reverse();});rejectMutation(0,v=>{v.result.schmidtSpectrum[0].isolatingLower=q;});rejectMutation(0,v=>{v.result.schmidtSpectrum[0].exactRationalRootKnown=!v.result.schmidtSpectrum[0].exactRationalRootKnown;});rejectMutation(0,v=>{v.result.schmidtSpectrum[0].coefficient=q;});rejectMutation(0,v=>{v.result.entropyBits=q;});rejectMutation(0,v=>{v.result.entropyNats=q;});rejectMutation(0,v=>{v.result.partialTranspose.pairs[0].i=1;});rejectMutation(0,v=>{v.result.partialTranspose.negativity=q;});rejectMutation(0,v=>{v.result.classification.productState=true;});
});
test('pure conditional,reflection and product-factor accounts preserve dimensions and impossible events',()=>{
 const q=exactObservation(f(BigInt(1),BigInt(3)));
 rejectMutation(0,v=>{v.result.projectorTests.conditionBOnA.rows[0].conditionalPurity=q;});rejectMutation(0,v=>{v.result.projectorTests.conditionAOnB.rows[1].projectorRank=0;});rejectMutation(0,v=>{v.result.projectorTests.jointProbabilities[0][0]=q;});rejectMutation(0,v=>{v.result.localReflection.transformedRawCoefficients[0][0].real=q;});
 for(const label of ['zero A event','3x4 complex product','1x1 complex generic']){
  const input=snapshot({kind:kinds[0],parameters:families[0].cases.find(c=>c.label===label)!.input}as TensorInput),r=evaluate({version:1,requestId:1,input});assert(r.ok&&r.kind===kinds[0]);assert(success(r,input,1));const v:any=structuredClone(r);
  if(label==='zero A event'){const row=v.result.projectorTests.conditionBOnA.rows.find((row:any)=>row.probability.sign===0);assert(row);row.conditionalDensity=v.result.reducedB.density;}else{v.result.productFactorization.rawA[0].real=q;}
  assert(!success(v,input,1));
 }
});
test('mixed attribution binds general state,Pauli expansion,partial transpose and exact CHSH boundary',()=>{
 const q=exactObservation(f(BigInt(1),BigInt(3)));
 for(const mutate of [(r:any)=>{r.joint.density[0][0].real=q;},(r:any)=>{r.reducedA.blochVector[1]=q;},(r:any)=>{r.correlationTensor[0][0]=q;},(r:any)=>{r.partialTranspose.matrix[0][1].imaginary=q;},(r:any)=>{r.partialTranspose.positiveSemidefinite=!r.partialTranspose.positiveSemidefinite;},(r:any)=>{r.chsh.gramMatrix[0][0].real=q;},(r:any)=>{r.chsh.exactBoundaryCertificate.threshold=q;},(r:any)=>{r.chsh.exactBoundaryCertificate.determinant=q;},(r:any)=>{r.chsh.optimizedViolates=!r.chsh.optimizedViolates;},(r:any)=>{r.chsh.optimizedMaximum=q;},(r:any)=>{r.chsh.selectedSigned=q;},(r:any)=>{r.classification.separableTwoQubit=!r.classification.separableTwoQubit;}])rejectMutation(1,v=>mutate(v.result));
});
test('mixed attribution binds conditional nulls,joint laws,spectral order and information differences',()=>{
 const q=exactObservation(f(BigInt(1),BigInt(3)));
 rejectMutation(1,v=>{v.result.localTests.outcomeOrder.reverse();});rejectMutation(1,v=>{v.result.localTests.settings.reverse();});rejectMutation(1,v=>{v.result.localTests.pairs[0][0].jointProbabilities[0][0]=q;});rejectMutation(1,v=>{v.result.localTests.conditionals[0].rows[0].conditionalPurity=q;});rejectMutation(1,v=>{v.result.joint.spectrum.reverse();});rejectMutation(1,v=>{v.result.joint.spectrum[0].multiplicity++;});rejectMutation(1,v=>{v.result.mutualInformationNats=q;});rejectMutation(1,v=>{v.result.conditionalEntropyAGivenBNats=q;});
 const input=snapshot({kind:kinds[1],parameters:families[1].cases.find(c=>c.label==='pure product zero outcomes')!.input}as TensorInput),r=evaluate({version:1,requestId:1,input});assert(r.ok&&r.kind===kinds[1]);const v:any=structuredClone(r);v.result.localTests.conditionals[0].rows[1].conditionalDensity=v.result.reducedB.density;assert(!success(v,input,1));
});
test('reader-safe mixed guards match engine keys and exact PSD decisions for bounded general perturbations',()=>{
 assert.deepEqual(tensorMixedKeys,mixedEntanglementKeys);let seed=78131;
 for(let i=0;i<120;i++){
  const base=families[1].cases[i%families[1].cases.length].input,p={...base};seed=(seed*48271)%2147483647;const field=tensorMixedKeys[seed%16];p[field]+=seed%11-5;
  assert.equal(isTensorMixedParameters(p),isMixedEntanglementParameters(p),`perturbation${i}:${field}`);
 }
 for(const key of tensorMixedKeys){const p={...families[1].cases[0].input,[key]:1000001};assert(!isTensorMixedParameters(p));assert(!isMixedEntanglementParameters(p));}
 const p={...families[1].cases.find(c=>c.label==='pure product zero outcomes')!.input,hReal00:1,hReal11:0,hReal22:-1,hReal33:1};assert(!isTensorMixedParameters(p));assert(!isMixedEntanglementParameters(p));
});
test('all reference preparations retain canonical identity through text,storage and negative-zero normalization',()=>{
 for(const f of families)for(const c of f.cases){const input=snapshot({kind:f.kind,parameters:c.input}as TensorInput);assert.equal(key(draft(JSON.stringify(input))!),key(input));assert.equal(key(stored(JSON.stringify({version:1,value:input}))!),key(input));}
 for(const input of inputs)for(const field of Object.keys(input.parameters)){
  const raw=JSON.stringify(input).replace(new RegExp('"'+field+'":-?\\d+'),'"'+field+'":1.00000000000000001');assert.equal(draft(raw),null);
 }
 const p={...inputs[0].parameters,stateImag33:-0},input=snapshot({kind:kinds[0],parameters:p});assert(!Object.is(input.parameters.stateImag33,-0));assert.equal(stored('{"version":1.00000000000000001,"value":'+JSON.stringify(input)+'}'),null);
});
test('repeated spectral weights retain one distinct root row with its multiplicity',()=>{
 for(const[family,label]of [[0,'3x3 maximal Schmidt'],[1,'independent maximally mixed']]as const){
  const input=snapshot({kind:kinds[family],parameters:families[family].cases.find(c=>c.label===label)!.input}as TensorInput),r=evaluate({version:1,requestId:1,input});assert(r.ok);assert(success(r,input,1));const v:any=structuredClone(r),rows=family===0?v.result.schmidtSpectrum:v.result.joint.spectrum;assert.equal(rows.length,1);const n=rows[0].multiplicity;rows[0].multiplicity=n-1;rows.push({...structuredClone(rows[0]),multiplicity:1});assert(!success(v,input,1));
 }
});
